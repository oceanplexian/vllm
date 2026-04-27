# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for heterogeneous-page-size KV cache groups (KANB-91).

Pre-KANB-91, ``unify_kv_cache_spec_page_size`` raised ``NotImplementedError``
when two attention groups had non-divisible ``page_size_bytes`` — the load-
bearing case being a TurboQuant target layer set + a non-TQ (or differently
configured) speculative-decoding drafter, where the head-config mismatch
produces page sizes a few percent apart.

These tests exercise the new behavior: groups with different page sizes
coexist via per-group ``BlockPool`` instances, each with its own
``num_blocks`` derived from that group's share of the memory budget.
"""

import pytest
import torch

from vllm.config import ModelConfig, SchedulerConfig, VllmConfig
from vllm.utils.mem_constants import GiB_bytes
from vllm.v1.core.kv_cache_utils import (
    get_kv_cache_config_from_groups,
    get_kv_cache_groups,
)
from vllm.v1.kv_cache_interface import (
    KVCacheGroupSpec,
    MambaSpec,
    TQFullAttentionSpec,
)

pytestmark = pytest.mark.cpu_test


def _make_vllm_config(max_model_len: int = 16384) -> VllmConfig:
    return VllmConfig(
        model_config=ModelConfig(max_model_len=max_model_len),
        scheduler_config=SchedulerConfig(
            max_model_len=max_model_len, is_encoder_decoder=False
        ),
    )


def _tq_spec(num_kv_heads: int, head_size: int, tq_slot_size: int):
    """Mirror of the Qwen3.6 TQ shapes that produce the page-size mismatch."""
    return TQFullAttentionSpec(
        block_size=3168,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        dtype=torch.bfloat16,
        tq_slot_size=tq_slot_size,
    )


def _mamba_spec(block_size: int = 3168):
    """A small MambaSpec to force `_get_kv_cache_groups_heterogeneous_page_size`
    rather than `UniformTypeKVCacheSpecs.from_specs` (which would lump
    all-AttentionSpec layers into a single uniform-type group)."""
    return MambaSpec(
        block_size=block_size,
        shapes=((2, 64), (3, 8, 8)),
        dtypes=(torch.float32, torch.float32),
        mamba_cache_mode="none",
        num_speculative_blocks=0,
    )


def test_hybrid_mamba_plus_two_tq_specs_form_distinct_groups():
    """The Qwen3.6 + DFlash production failure: mamba + target TQ full-attn +
    drafter TQ full-attn. Without mamba in the mix, the two TQ specs are both
    AttentionSpec subclasses and would be lumped into a single
    UniformTypeKVCacheSpecs group (already supported). Mamba breaks that
    short-circuit and forces the path that pre-KANB-91 raised
    NotImplementedError on the non-divisible TQ page sizes."""
    target_spec = _tq_spec(num_kv_heads=2, head_size=256, tq_slot_size=262)
    drafter_spec = _tq_spec(num_kv_heads=4, head_size=128, tq_slot_size=134)
    mamba_spec = _mamba_spec()
    assert target_spec.page_size_bytes != drafter_spec.page_size_bytes
    larger = max(target_spec.page_size_bytes, drafter_spec.page_size_bytes)
    smaller = min(target_spec.page_size_bytes, drafter_spec.page_size_bytes)
    # Non-divisible — the exact shape that pre-KANB-91 broke on.
    assert larger % smaller != 0

    kv_cache_spec = {f"target.layer_{i}": target_spec for i in range(4)}
    kv_cache_spec.update({f"drafter.layer_{i}": drafter_spec for i in range(2)})
    kv_cache_spec.update({f"mamba.layer_{i}": mamba_spec for i in range(6)})

    vllm_config = _make_vllm_config()
    groups = get_kv_cache_groups(vllm_config, kv_cache_spec)
    # Each spec class winds up in at least one group; layer-count balancing
    # may split a type across multiple groups but no group mixes spec types.
    page_sizes = {g.kv_cache_spec.page_size_bytes for g in groups}
    assert target_spec.page_size_bytes in page_sizes
    assert drafter_spec.page_size_bytes in page_sizes
    assert mamba_spec.page_size_bytes in page_sizes
    for g in groups:
        # Within a group every layer is a single canonical spec.
        spec_classes = {type(g.kv_cache_spec).__name__}
        assert len(spec_classes) == 1


def test_per_group_tensor_sizes_reflect_each_group_page_size():
    """KVCacheTensors built from heterogeneous groups must be sized for their
    own group's page_size, not coerced to a single max-page slab."""
    target_spec = _tq_spec(num_kv_heads=2, head_size=256, tq_slot_size=262)
    drafter_spec = _tq_spec(num_kv_heads=4, head_size=128, tq_slot_size=134)

    target_group = KVCacheGroupSpec(
        layer_names=[f"target.layer_{i}" for i in range(4)],
        kv_cache_spec=target_spec,
    )
    drafter_group = KVCacheGroupSpec(
        layer_names=[f"drafter.layer_{i}" for i in range(2)],
        kv_cache_spec=drafter_spec,
    )
    # `get_kv_cache_config_from_groups` general-case branch is taken when
    # there is more than one group OR the single group is not
    # UniformTypeKVCacheSpecs. Two distinct AttentionSpec groups suffice.
    available_memory = 4 * GiB_bytes
    cfg = get_kv_cache_config_from_groups(
        _make_vllm_config(),
        [target_group, drafter_group],
        available_memory=available_memory,
        suppress_log=True,
    )

    # Each layer gets its own KVCacheTensor, sized for its group's page_size.
    target_tensors = [
        t for t in cfg.kv_cache_tensors if any(n.startswith("target.") for n in t.shared_by)
    ]
    drafter_tensors = [
        t for t in cfg.kv_cache_tensors if any(n.startswith("drafter.") for n in t.shared_by)
    ]
    assert len(target_tensors) == 4
    assert len(drafter_tensors) == 2

    expected_target_size = target_spec.page_size_bytes * cfg.get_num_blocks(0)
    expected_drafter_size = drafter_spec.page_size_bytes * cfg.get_num_blocks(1)
    for t in target_tensors:
        assert t.size == expected_target_size
    for t in drafter_tensors:
        assert t.size == expected_drafter_size

    # No cross-group sharing — each tensor's `shared_by` is exactly one layer.
    for t in cfg.kv_cache_tensors:
        assert len(t.shared_by) == 1


def test_heterogeneous_config_respects_memory_budget():
    """Total allocated bytes (Σ tensor.size) must not exceed available_memory."""
    target_spec = _tq_spec(num_kv_heads=2, head_size=256, tq_slot_size=262)
    drafter_spec = _tq_spec(num_kv_heads=4, head_size=128, tq_slot_size=134)
    target_group = KVCacheGroupSpec(
        layer_names=[f"target.layer_{i}" for i in range(16)],
        kv_cache_spec=target_spec,
    )
    drafter_group = KVCacheGroupSpec(
        layer_names=[f"drafter.layer_{i}" for i in range(5)],
        kv_cache_spec=drafter_spec,
    )

    available_memory = 4 * GiB_bytes
    cfg = get_kv_cache_config_from_groups(
        _make_vllm_config(),
        [target_group, drafter_group],
        available_memory=available_memory,
        suppress_log=True,
    )

    total_bytes = sum(t.size for t in cfg.kv_cache_tensors)
    assert total_bytes <= available_memory
    # And we used "most" of the budget — within one block of fitting
    # exactly. Each block costs Σ_g(page_size_g × len_g) bytes.
    bytes_per_block = sum(
        g.kv_cache_spec.page_size_bytes * len(g.layer_names)
        for g in cfg.kv_cache_groups
    )
    assert available_memory - total_bytes < bytes_per_block


def test_num_blocks_per_group_is_populated():
    """num_blocks_per_group is parallel to kv_cache_groups and consulted by
    the per-group BlockPool wiring."""
    target_spec = _tq_spec(num_kv_heads=2, head_size=256, tq_slot_size=262)
    drafter_spec = _tq_spec(num_kv_heads=4, head_size=128, tq_slot_size=134)
    groups = [
        KVCacheGroupSpec(
            layer_names=[f"target.layer_{i}" for i in range(4)],
            kv_cache_spec=target_spec,
        ),
        KVCacheGroupSpec(
            layer_names=[f"drafter.layer_{i}" for i in range(2)],
            kv_cache_spec=drafter_spec,
        ),
    ]
    cfg = get_kv_cache_config_from_groups(
        _make_vllm_config(), groups, available_memory=2 * GiB_bytes, suppress_log=True
    )
    assert cfg.num_blocks_per_group is not None
    assert len(cfg.num_blocks_per_group) == len(groups)
    assert all(n > 0 for n in cfg.num_blocks_per_group)
    assert cfg.num_blocks == max(cfg.num_blocks_per_group)
