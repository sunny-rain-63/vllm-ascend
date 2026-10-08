import unittest

import pytest
import torch
import torch_npu
from vllm.v1.attention.backend import AttentionType

from vllm_ascend.attention.attention_v1 import AscendAttentionBackendImpl
from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.worker.v2.kv_cache_utils import reshape_paged_attention_kv_cache


class TestPaKvCacheOps(unittest.TestCase):
    def test_scatter_pa_kv_cache_slot_mapping_zero_and_minus_one(self):
        torch.manual_seed(20260709)

        dtype = torch.float16
        block_size = 4
        num_blocks = 2
        num_heads = 1
        head_dim = 8
        slot_mapping = torch.tensor([0, -1, 3], dtype=torch.int32, device="npu")
        key = torch.arange(3 * num_heads * head_dim, dtype=dtype, device="npu").view(3, num_heads, head_dim)
        value = key + 100
        key_cache = torch.randn(num_blocks, block_size, num_heads, head_dim, dtype=dtype, device="npu")
        value_cache = torch.randn_like(key_cache)

        expected_key_cache = key_cache.clone()
        expected_value_cache = value_cache.clone()
        for token_idx, slot in enumerate(slot_mapping.cpu().tolist()):
            if slot < 0:
                continue
            expected_key_cache[slot // block_size, slot % block_size] = key[token_idx]
            expected_value_cache[slot // block_size, slot % block_size] = value[token_idx]

        torch_npu.npu_scatter_pa_kv_cache(
            key=key,
            value=value,
            key_cache=key_cache,
            value_cache=value_cache,
            slot_mapping=slot_mapping,
            cache_mode="Norm",
        )
        torch.npu.synchronize()

        torch.testing.assert_close(key_cache, expected_key_cache, atol=0, rtol=0)
        torch.testing.assert_close(value_cache, expected_value_cache, atol=0, rtol=0)

    def test_scatter_pa_kv_cache_all_minus_one_leaves_cache_unchanged(self):
        dtype = torch.float16
        key = torch.randn(2, 1, 8, dtype=dtype, device="npu")
        value = torch.randn_like(key)
        key_cache = torch.randn(2, 4, 1, 8, dtype=dtype, device="npu")
        value_cache = torch.randn_like(key_cache)
        expected_key_cache = key_cache.clone()
        expected_value_cache = value_cache.clone()
        slot_mapping = torch.tensor([-1, -1], dtype=torch.int32, device="npu")

        torch_npu.npu_scatter_pa_kv_cache(
            key=key,
            value=value,
            key_cache=key_cache,
            value_cache=value_cache,
            slot_mapping=slot_mapping,
            cache_mode="Norm",
        )
        torch.npu.synchronize()

        torch.testing.assert_close(key_cache, expected_key_cache, atol=0, rtol=0)
        torch.testing.assert_close(value_cache, expected_value_cache, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("splits", [1, 4, 9])
@pytest.mark.parametrize("graph_mode", [False, True])
def test_dflash_context_and_query_writes_to_strided_pages(dtype, splits, graph_mode):
    """DFlash's eager context writer and captured query writer must agree."""
    num_pages, block_size, heads, dim = 4, 128, 2, 64
    block_elements = block_size * heads * dim
    kernel_stride = 2 * block_elements + 128
    page_elements = splits * kernel_stride
    offset = 128
    storage = torch.full((offset + num_pages * page_elements + offset,), -7, dtype=dtype, device="npu")
    raw = storage[offset : offset + num_pages * page_elements].view(torch.int8)
    key_cache, value_cache = reshape_paged_attention_kv_cache(
        raw,
        (2, num_pages * splits, block_size, heads, dim),
        dtype,
        page_elements * storage.element_size(),
        splits,
    )
    impl = AscendAttentionBackendImpl.__new__(AscendAttentionBackendImpl)
    impl.attn_type = AttentionType.DECODER
    impl.key_cache = impl.value_cache = None
    impl.use_bnsd_kv_cache = False
    context_slots = [splits * block_size, (splits + 1) * block_size - 1, -1]
    query_slots = [2 * splits * block_size, 3 * splits * block_size - 1, -1]
    key = torch.arange(3 * heads * dim, dtype=torch.float32).reshape(3, heads, dim).to(dtype).to("npu")
    value = -key
    context_mapping = torch.tensor(context_slots, dtype=torch.int32, device="npu")
    query_mapping = torch.tensor(query_slots, dtype=torch.int32, device="npu")
    expected = storage.cpu()

    impl.do_kv_cache_update(None, key, value, (key_cache, value_cache), context_mapping)

    def query_write():
        DeviceOperator.reshape_and_cache(key, value, key_cache, value_cache, query_mapping)

    if graph_mode:
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            query_write()
        graph.replay()
    else:
        query_write()
    torch.npu.synchronize()

    for slots in (context_slots, query_slots):
        for token, slot in enumerate(slots):
            if slot < 0:
                continue
            block, token_offset = divmod(slot, block_size)
            begin = offset + block * kernel_stride + token_offset * heads * dim
            expected[begin : begin + heads * dim] = key[token].cpu().flatten()
            expected[begin + block_elements : begin + block_elements + heads * dim] = value[token].cpu().flatten()
    torch.testing.assert_close(storage.cpu(), expected, rtol=0, atol=0)
