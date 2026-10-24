# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

"""QSA metadata must follow GPU boundaries when a captured batch shrinks."""

import pytest
import torch
import torch_npu  # noqa: F401
from vllm.v1.attention.backend import CommonAttentionMetadata

from vllm_ascend.models.qwen4_exp.qsa_metadata import build_qsa_metadata


def _common(query_start_loc, seq_lens, block_table, actual, padded, cpu_boundaries):
    metadata = CommonAttentionMetadata(
        query_start_loc=query_start_loc,
        query_start_loc_cpu=torch.tensor(cpu_boundaries, dtype=torch.int32),
        seq_lens=seq_lens,
        num_reqs=query_start_loc.numel() - 1,
        num_actual_tokens=actual,
        max_query_len=1,
        max_seq_len=300,
        block_table_tensor=block_table,
        slot_mapping=torch.full((padded,), -1, dtype=torch.int64, device="npu"),
    )
    metadata.num_input_tokens = padded
    return metadata


def _buffers(capacity):
    return (
        torch.empty(capacity, dtype=torch.int32, device="npu"),
        torch.empty(capacity, dtype=torch.int64, device="npu"),
        torch.empty(capacity, dtype=torch.int32, device="npu"),
        torch.empty(capacity, dtype=torch.int64, device="npu"),
    )


def _build(common, buffers):
    return build_qsa_metadata(common, *buffers, storage_block_size=8, compress_ratio=1, circular_buffer_size=8)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU required")
def test_qsa_raw_slots_follow_changing_gpu_boundaries_in_one_graph():
    query_start_loc = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int32, device="npu")
    seq_lens = torch.tensor([101, 201, 51, 61], dtype=torch.int32, device="npu")
    # Keep positive stale physical blocks in rows that become dummy requests.
    block_table = torch.tensor([[11], [12], [21], [22]], dtype=torch.int32, device="npu")
    common = _common(query_start_loc, seq_lens, block_table, 4, 4, [0, 1, 2, 3, 4])
    buffers = _buffers(4)
    _build(common, buffers)
    torch.npu.synchronize()
    # A new CommonAttentionMetadata is used for each capture pass in the runner.
    # Record the token-map producer, rather than reusing the warmup's cache.
    common._token_to_req_indices_cache = None
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        result = _build(common, buffers)

    cases = [
        ([0, 1, 2, 3, 4], [101, 201, 51, 61], [100, 200, 50, 60], [92, 96, 170, 180]),
        ([0, 1, 2, 2, 2], [101, 201, 0, 0], [100, 200, -1, -1], [92, 96, -1, -1]),
        ([0, 1, 2, 3, 4], [101, 201, 0, 0], [100, 200, -1, -1], [92, 96, -1, -1]),
        ([0, 1, 2, 3, 4], [101, 201, 51, 61], [100, 200, 50, 60], [92, 96, 170, 180]),
    ]
    for boundaries, lengths, expected_positions, expected_slots in cases:
        query_start_loc.copy_(torch.tensor(boundaries, dtype=torch.int32, device="npu"))
        seq_lens.copy_(torch.tensor(lengths, dtype=torch.int32, device="npu"))
        graph.replay()
        torch.npu.synchronize()
        assert result[1].cpu().tolist() == expected_positions
        assert result[3].cpu().tolist() == expected_slots
        if lengths[-1] == 0:
            assert result[0].cpu().tolist()[2:] == [0, 0]
            assert result[2].cpu().tolist()[2:] == [0, 0]


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU required")
def test_smaller_runtime_metadata_clears_persistent_captured_tail():
    buffers = _buffers(4)
    block_table = torch.tensor([[11], [12], [21], [22]], dtype=torch.int32, device="npu")
    large = _common(
        torch.tensor([0, 1, 2, 3, 4], dtype=torch.int32, device="npu"),
        torch.tensor([101, 201, 51, 61], dtype=torch.int32, device="npu"),
        block_table,
        4,
        4,
        [0, 1, 2, 3, 4],
    )
    captured_views = _build(large, buffers)
    small = _common(
        torch.tensor([0, 1, 2], dtype=torch.int32, device="npu"),
        torch.tensor([101, 201], dtype=torch.int32, device="npu"),
        block_table[:2],
        2,
        4,
        [0, 1, 2],
    )
    result = _build(small, buffers)
    torch.npu.synchronize()
    assert all(tensor.numel() == 2 for tensor in result)
    assert captured_views[0].cpu().tolist() == [0, 1, 0, 0]
    assert captured_views[1].cpu().tolist() == [100, 200, -1, -1]
    assert captured_views[2].cpu().tolist() == [101, 201, 0, 0]
    assert captured_views[3].cpu().tolist() == [92, 96, -1, -1]
