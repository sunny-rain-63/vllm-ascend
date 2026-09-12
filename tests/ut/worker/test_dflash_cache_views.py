# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Exercise mixed DFlash's real CPU tensor views without NPU kernels."""

from types import SimpleNamespace

import pytest
import torch
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
    SlidingWindowSpec,
)
from vllm.v1.worker.utils import AttentionGroup

from vllm_ascend.attention.attention_v1 import (
    AscendAttentionBackend,
    AscendAttentionBackendImpl,
)
from vllm_ascend.core.dflash_cache import (
    align_dflash_cache_specs,
    validate_dflash_cache_views,
)
from vllm_ascend.worker.v2 import attn_utils

NUM_BLOCKS = 12
CONV_BYTES = 102400
SSM_BYTES = 1572864
PAGE_BYTES = CONV_BYTES + 2 * SSM_BYTES
STORAGE_BLOCK_SIZE = 1536
KERNEL_BLOCK_SIZE = 128


@pytest.fixture
def mixed_cache(monkeypatch):
    config = SimpleNamespace(
        use_v2_model_runner=True,
        speculative_config=SimpleNamespace(
            method="dflash",
            draft_model_config=SimpleNamespace(
                hf_config=SimpleNamespace(layer_types=["sliding_attention"] * 4 + ["full_attention"])
            ),
        ),
        parallel_config=SimpleNamespace(pipeline_parallel_size=1),
        model_config=SimpleNamespace(hf_config=SimpleNamespace()),
    )
    original_specs = {
        "target.full": FullAttentionSpec(
            block_size=STORAGE_BLOCK_SIZE,
            num_kv_heads=2,
            head_size=256,
            dtype=torch.bfloat16,
            page_size_padded=PAGE_BYTES,
        ),
        "draft.swa": SlidingWindowSpec(
            block_size=KERNEL_BLOCK_SIZE,
            num_kv_heads=4,
            head_size=128,
            dtype=torch.bfloat16,
            sliding_window=2048,
            page_size_padded=PAGE_BYTES,
        ),
        "target.mamba": MambaSpec(
            block_size=STORAGE_BLOCK_SIZE,
            shapes=((10, 5120), (24, 128, 128)),
            dtypes=(torch.bfloat16, torch.float32),
            page_size_padded=PAGE_BYTES,
            num_speculative_blocks=7,
        ),
    }
    specs = align_dflash_cache_specs(config, original_specs)
    groups = [KVCacheGroupSpec(layer_names=[name], kv_cache_spec=spec) for name, spec in specs.items()]
    cache_config = KVCacheConfig(num_blocks=NUM_BLOCKS, kv_cache_tensors=[], kv_cache_groups=groups)
    # Different cache groups deliberately alias one approximately 39 MB pool.
    # Correctness must follow from physical block ownership, not separate tensors.
    backing = torch.zeros(NUM_BLOCKS * PAGE_BYTES, dtype=torch.int8)
    raw_caches = dict.fromkeys(specs, backing)
    layers = {name: SimpleNamespace(impl=SimpleNamespace()) for name in specs}
    monkeypatch.setattr(attn_utils, "get_current_vllm_config", lambda: config)
    monkeypatch.setattr(attn_utils, "get_layers_from_vllm_config", lambda *_args, **_kwargs: layers)
    monkeypatch.setattr(attn_utils, "enable_sfa", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(attn_utils, "enable_fa_quant", lambda *_args, **_kwargs: False)
    attention_groups = [
        AttentionGroup(
            backend=AscendAttentionBackend,
            layer_names=group.layer_names,
            kv_cache_spec=group.kv_cache_spec,
            kv_cache_group_id=group_id,
        )
        for group_id, group in enumerate(groups)
    ]
    caches = attn_utils._reshape_kv_cache_v2(
        attn_groups=attention_groups,
        kv_cache_raw_tensors=raw_caches,
        cache_dtype="auto",
        kernel_block_sizes=[KERNEL_BLOCK_SIZE] * len(groups),
        shared_kv_cache_layers={},
        kv_cache_config=cache_config,
    )
    return SimpleNamespace(
        config=config,
        original_specs=original_specs,
        specs=specs,
        cache_config=cache_config,
        backing=backing,
        raw_caches=raw_caches,
        layers=layers,
        caches=caches,
    )


def _physical_block(cache, block_id):
    blocks_per_physical = STORAGE_BLOCK_SIZE // KERNEL_BLOCK_SIZE
    return cache[block_id * blocks_per_physical : (block_id + 1) * blocks_per_physical]


def test_alignment_changes_storage_geometry_without_disabling_sliding_window(mixed_cache):
    original = mixed_cache.original_specs["draft.swa"]
    aligned = mixed_cache.specs["draft.swa"]
    assert original.block_size == KERNEL_BLOCK_SIZE
    assert type(aligned) is SlidingWindowSpec
    assert aligned.block_size == STORAGE_BLOCK_SIZE
    assert aligned.sliding_window == original.sliding_window == 2048
    assert aligned.page_size_bytes == PAGE_BYTES
    assert type(mixed_cache.specs["target.full"]) is FullAttentionSpec
    assert mixed_cache.specs["target.full"].block_size == STORAGE_BLOCK_SIZE

    full_key, full_value = mixed_cache.caches["target.full"]
    swa_key, swa_value = mixed_cache.caches["draft.swa"]
    conv, state = mixed_cache.caches["target.mamba"]
    assert full_key.shape == (144, 128, 2, 256)
    assert swa_key.shape == (144, 128, 4, 128)
    assert conv.shape == (NUM_BLOCKS, 10, 5120)
    assert state.shape == (NUM_BLOCKS, 24, 128, 128)
    assert full_key.data_ptr() == swa_key.data_ptr() == state.data_ptr()
    assert full_value.data_ptr() == swa_value.data_ptr()
    assert state.data_ptr() - conv.data_ptr() == NUM_BLOCKS * CONV_BYTES


def test_swa_block_one_cannot_overwrite_full_blocks_ten_eleven_or_other_ssm_blocks(mixed_cache):
    full_key, full_value = mixed_cache.caches["target.full"]
    swa_key, swa_value = mixed_cache.caches["draft.swa"]
    conv, state = mixed_cache.caches["target.mamba"]
    conv.fill_(2)
    state.fill_(1.25)
    protected_full = []
    for block_id in (10, 11):
        for plane, value in ((full_key, 3), (full_value, 5)):
            view = _physical_block(plane, block_id)
            view.fill_(value)
            protected_full.append((view, view.clone()))
    # Full K and SSM intentionally share SAME-ID bytes; compare raw bytes so
    # floating-point reinterpretation cannot mask corruption with NaNs.
    protected_state = [
        (state[i].view(torch.uint8), state[i].view(torch.uint8).clone())
        for i in range(NUM_BLOCKS)
        if i != 1
    ]
    original_conv = conv.clone()

    _physical_block(swa_key, 1).fill_(7)
    _physical_block(swa_value, 1).fill_(9)

    for view, expected in protected_full + protected_state:
        assert torch.equal(view, expected)
    assert torch.equal(conv, original_conv)
    assert torch.all(_physical_block(swa_key, 1) == 7)
    assert torch.all(_physical_block(swa_value, 1) == 9)


@pytest.mark.parametrize("defect", ["dtype", "offset", "noncontiguous"])
def test_materialized_cache_layout_mismatch_fails_closed(mixed_cache, defect):
    caches = dict(mixed_cache.caches)
    key, value = caches["draft.swa"]
    if defect == "dtype":
        key = key.view(torch.float16)
    elif defect == "offset":
        start = NUM_BLOCKS * CONV_BYTES + key.element_size()
        size = key.numel() * key.element_size()
        key = mixed_cache.backing[start : start + size].view(key.dtype).view(key.shape)
    else:
        key = key.transpose(1, 2)
    caches["draft.swa"] = (key, value)
    with pytest.raises(ValueError, match="aligned contiguous-plane layout"):
        validate_dflash_cache_views(
            mixed_cache.config, mixed_cache.cache_config, mixed_cache.raw_caches, caches
        )


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("layer_name", ["target.full", "draft.swa"])
def test_materialization_installs_guard_for_all_null_subblocks_and_out_of_bounds(mixed_cache, dtype, layer_name):
    impl = mixed_cache.layers[layer_name].impl
    limit = NUM_BLOCKS * STORAGE_BLOCK_SIZE
    assert impl._dflash_null_block_size == STORAGE_BLOCK_SIZE
    assert impl._dflash_cache_slot_limit == limit
    slots = torch.tensor([-1, 0, 127, 128, 1535, 1536, limit - 1, limit], dtype=dtype)
    expected = torch.tensor([-1, -1, -1, -1, -1, 1536, limit - 1, -1], dtype=dtype)
    original = slots.clone()
    result = AscendAttentionBackendImpl._mask_dflash_cache_slots(impl, slots)
    assert result.dtype == dtype
    assert torch.equal(result, expected)
    assert torch.equal(slots, original)


def test_cache_slot_guard_is_noop_for_unaffected_models():
    slots = torch.tensor([-1, 0, 128, 4096], dtype=torch.int32)
    assert AscendAttentionBackendImpl._mask_dflash_cache_slots(SimpleNamespace(), slots) is slots
