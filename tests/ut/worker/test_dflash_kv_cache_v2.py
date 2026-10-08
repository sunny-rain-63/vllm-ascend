# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

import pytest
import torch

from vllm_ascend.worker.v2.kv_cache_utils import reshape_paged_attention_kv_cache


@pytest.mark.parametrize("splits", [1, 2, 9])
@pytest.mark.parametrize("padding", [0, 16])
@pytest.mark.parametrize("storage_offset", [0, 32])
@pytest.mark.parametrize("head_size_v", [2, 4])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_dflash_pages_preserve_other_blocks_and_padding(splits, padding, storage_offset, head_size_v, dtype):
    num_pages, block_size, heads, head_size = 3, 4, 2, 4
    key_elements = block_size * heads * head_size
    value_elements = block_size * heads * head_size_v
    kernel_stride = key_elements + value_elements + padding
    page_elements = splits * kernel_stride
    sentinel = -7
    storage = torch.full((storage_offset + num_pages * page_elements + 32,), sentinel, dtype=dtype)
    raw = storage[storage_offset : storage_offset + num_pages * page_elements].view(torch.int8)
    key, value = reshape_paged_attention_kv_cache(
        raw,
        (2, num_pages * splits, block_size, heads, head_size),
        dtype,
        page_elements * storage.element_size(),
        splits,
        head_size_v=head_size_v,
    )
    assert key.untyped_storage().data_ptr() == value.untyped_storage().data_ptr() == storage.data_ptr()
    assert key.stride(0) == value.stride(0) == kernel_stride
    assert not key.is_contiguous() and not value.is_contiguous()

    # Context and query slots cross a kernel-block boundary inside one owned
    # scheduler page. Compare the ENTIRE allocation to catch damage to another
    # group's page, K/V overlap, prefix/suffix guards, and page padding.
    slots = [splits * block_size, 2 * splits * block_size - 1]
    expected = storage.clone()
    for token, slot in enumerate(slots):
        block, offset = divmod(slot, block_size)
        key[block, offset].fill_(token + 1)
        value[block, offset].fill_(token + 11)
        base = storage_offset + block * kernel_stride
        k_begin = base + offset * heads * head_size
        v_begin = base + key_elements + offset * heads * head_size_v
        expected[k_begin : k_begin + heads * head_size] = token + 1
        expected[v_begin : v_begin + heads * head_size_v] = token + 11
    torch.testing.assert_close(storage, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "shape,page_bytes,splits,raw_bytes,message",
    [
        ((3, 2, 4, 1, 2), 32, 1, 64, "shape"),
        ((2, 2, 4, 1, 2), 33, 1, 66, "aligned to its dtype"),
        ((2, 2, 4, 1, 2), 32, 0, 64, "positive"),
        ((2, 2, 4, 1, 2), 66, 2, 132, "dtype-aligned"),
        ((2, 2, 4, 1, 2), 16, 1, 64, "too small"),
        ((2, 2, 4, 1, 2), 32, 1, 62, "backing allocation"),
    ],
)
def test_dflash_rejects_invalid_page_geometry(shape, page_bytes, splits, raw_bytes, message):
    # Extra underlying storage must not allow a view to escape its layer slice.
    raw = torch.zeros(raw_bytes + 128, dtype=torch.int8)[:raw_bytes]
    with pytest.raises(ValueError, match=message):
        reshape_paged_attention_kv_cache(raw, shape, torch.float16, page_bytes, splits)
