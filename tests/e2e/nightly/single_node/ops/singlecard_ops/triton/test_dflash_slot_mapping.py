# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.ops.triton.v2.block_table.mask_dflash_slots import mask_dflash_slots


def _reference(slots):
    return torch.where((slots >= 1536) & (slots < 5461 * 1536), slots, -1)


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("num_slots", [0, 1, 255, 256, 257, 4096, 16384])
@torch.inference_mode()
def test_dflash_slot_mapping_eager(dtype, num_slots):
    limit = 5461 * 1536
    boundary = torch.tensor([-1, 0, 127, 128, 1535, 1536, limit - 1, limit], dtype=dtype, device="npu")
    slots = boundary.repeat((num_slots + 7) // 8)[:num_slots]
    before = slots.clone()
    result = mask_dflash_slots(slots, 1536, limit, -1)
    torch.testing.assert_close(result, _reference(slots), rtol=0, atol=0)
    torch.testing.assert_close(slots, before, rtol=0, atol=0)
    assert result.dtype == slots.dtype and result.shape == slots.shape


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@torch.inference_mode()
def test_dflash_slot_mapping_graph_replay_reads_changed_slots(dtype):
    slots = torch.arange(4096, dtype=dtype, device="npu")
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        for _ in range(3):
            mask_dflash_slots(slots, 1536, 5461 * 1536, -1)
    torch.npu.current_stream().wait_stream(stream)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        result = mask_dflash_slots(slots, 1536, 5461 * 1536, -1)
    for offset in (0, 1536, 5461 * 1536, -10000):
        slots.copy_(torch.arange(4096, dtype=dtype, device="npu") + offset)
        before = slots.clone()
        graph.replay()
        torch.testing.assert_close(result, _reference(slots), rtol=0, atol=0)
        torch.testing.assert_close(slots, before, rtol=0, atol=0)
