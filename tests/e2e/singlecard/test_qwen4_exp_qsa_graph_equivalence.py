# SPDX-License-Identifier: Apache-2.0
"""Replay the QSA LI/GBSA transaction with changing data, not capture dummies.

Run on an Ascend device supporting QSA LightningIndexer and GBSA:
    pytest -q tests/e2e/singlecard/test_qwen4_exp_qsa_graph_equivalence.py

The eager oracle starts from identical cache contents for each replay. This is
an operator/graph regression test, not a model acceptance or performance test.
"""

from dataclasses import dataclass

import pytest
import torch
import torch_npu  # noqa: F401
import vllm_ascend.vllm_ascend_C  # noqa: F401

from vllm_ascend.models.qwen4_exp.lightning_indexer import qsa_select_paged_tokens_lightning
from vllm_ascend.ops.gbsa_qsa import qsa_gbsa_attention
from vllm_ascend.ops.triton.qwen4_exp.qsa import qsa_store_cache_rows

NUM_REQUESTS = 8
MAIN_HEAD_DIM = 256
INDEX_HEAD_DIM = 128
MAIN_QUERY_HEADS = 4
INDEX_QUERY_HEADS = 8
# TP8 Qwen3.8 aligns its main K page with the GDN SSM page at 768
# tokens. This also exercises GBSA's six 128-token subpages per block.
MANAGER_BLOCK_SIZE = 768
COMPRESS_RATIO = 4
COMPRESSED_BLOCK_SIZE = MANAGER_BLOCK_SIZE // COMPRESS_RATIO
BLOCKS_PER_REQUEST = 6
NUM_BLOCKS = NUM_REQUESTS * BLOCKS_PER_REQUEST + 1
TOKEN_TOPK = 2048
GROUP_TOPK = TOKEN_TOPK // COMPRESS_RATIO
INDEX_WIDTH = TOKEN_TOPK + COMPRESS_RATIO - 1
BASE_CONTEXT = 2177
REPLAY_COUNT = 4
ATOL = 2e-2
RTOL = 2e-2


@dataclass
class QSAReplayCase:
    query: torch.Tensor
    index_query: torch.Tensor
    current_key: torch.Tensor
    current_value: torch.Tensor
    current_compressed_key: torch.Tensor
    block_table: torch.Tensor
    token_to_req: torch.Tensor
    logical_positions: torch.Tensor
    seq_lens: torch.Tensor
    query_start_loc: torch.Tensor
    sparse_indices: torch.Tensor
    output: torch.Tensor
    caches: dict[str, torch.Tensor]


def _initial_caches():
    generator = torch.Generator().manual_seed(20261024)
    return {
        "key": torch.randn(NUM_BLOCKS, MANAGER_BLOCK_SIZE, 1, MAIN_HEAD_DIM, generator=generator).to(torch.bfloat16),
        "value": torch.randn(NUM_BLOCKS, MANAGER_BLOCK_SIZE, 1, MAIN_HEAD_DIM, generator=generator).to(torch.bfloat16),
        "compressed": torch.randn(NUM_BLOCKS, COMPRESSED_BLOCK_SIZE, 1, INDEX_HEAD_DIM, generator=generator).to(
            torch.bfloat16
        ),
    }


def _make_case(rows, caches):
    assert rows in (8, 32)
    query_per_request = rows // NUM_REQUESTS
    device = caches["key"].device
    return QSAReplayCase(
        query=torch.empty((rows, MAIN_QUERY_HEADS, MAIN_HEAD_DIM), dtype=torch.bfloat16, device=device),
        index_query=torch.empty((rows, INDEX_QUERY_HEADS, INDEX_HEAD_DIM), dtype=torch.bfloat16, device=device),
        current_key=torch.empty((rows, 1, MAIN_HEAD_DIM), dtype=torch.bfloat16, device=device),
        current_value=torch.empty((rows, 1, MAIN_HEAD_DIM), dtype=torch.bfloat16, device=device),
        current_compressed_key=torch.empty((rows, 1, INDEX_HEAD_DIM), dtype=torch.bfloat16, device=device),
        block_table=torch.empty((NUM_REQUESTS, BLOCKS_PER_REQUEST), dtype=torch.int32, device=device),
        token_to_req=torch.arange(NUM_REQUESTS, dtype=torch.int32, device=device).repeat_interleave(query_per_request),
        logical_positions=torch.empty(rows, dtype=torch.int64, device=device),
        seq_lens=torch.empty(NUM_REQUESTS, dtype=torch.int32, device=device),
        query_start_loc=torch.arange(NUM_REQUESTS + 1, dtype=torch.int32, device=device) * query_per_request,
        sparse_indices=torch.empty((rows, INDEX_WIDTH), dtype=torch.int32, device=device),
        output=torch.empty((rows, MAIN_QUERY_HEADS, MAIN_HEAD_DIM), dtype=torch.bfloat16, device=device),
        caches=caches,
    )


def _payload(rows, iteration):
    """Build a different valid batch while preserving every captured address."""
    generator = torch.Generator().manual_seed(20261024 + rows + iteration * 17)
    per_request = rows // NUM_REQUESTS
    owners = torch.arange(rows) // per_request
    seq_lens = BASE_CONTEXT + torch.arange(NUM_REQUESTS, dtype=torch.int32) * 11 + iteration * 7
    logical_positions = seq_lens[owners].long() - per_request + torch.arange(rows) % per_request
    assert logical_positions.min() > TOKEN_TOPK
    assert seq_lens.max() < MANAGER_BLOCK_SIZE * BLOCKS_PER_REQUEST
    block_ids = torch.arange(NUM_REQUESTS * BLOCKS_PER_REQUEST, dtype=torch.int32)
    block_table = ((block_ids + iteration * 7) % block_ids.numel() + 1).reshape(NUM_REQUESTS, BLOCKS_PER_REQUEST)
    # The reuse phase consumes these changed indices as graph inputs. Each
    # row selects 512 of >512 visible groups, rather than selecting all keys.
    visible_groups = (logical_positions + 1) // COMPRESS_RATIO
    first_group = (iteration * 13 + owners * 3) % (visible_groups - GROUP_TOPK + 1)
    groups = (first_group[:, None] + torch.arange(GROUP_TOPK)).to(torch.int32)
    groups = groups.roll(iteration, dims=1)
    sparse_indices = torch.full((rows, INDEX_WIDTH), -1, dtype=torch.int32)
    sparse_indices[:, :GROUP_TOPK] = groups

    def random_rows(*shape):
        return torch.randn(*shape, generator=generator).to(torch.bfloat16)

    return {
        "query": random_rows(rows, MAIN_QUERY_HEADS, MAIN_HEAD_DIM),
        "index_query": random_rows(rows, INDEX_QUERY_HEADS, INDEX_HEAD_DIM),
        "current_key": random_rows(rows, 1, MAIN_HEAD_DIM),
        "current_value": random_rows(rows, 1, MAIN_HEAD_DIM),
        "current_compressed_key": random_rows(rows, 1, INDEX_HEAD_DIM),
        "logical_positions": logical_positions,
        "seq_lens": seq_lens,
        "block_table": block_table,
        "sparse_indices": sparse_indices,
    }


def _reset(case, initial_caches, payload):
    for name, initial in initial_caches.items():
        case.caches[name].copy_(initial)
    for name, value in payload.items():
        getattr(case, name).copy_(value)
    case.output.fill_(float("nan"))


def _execute(case, select_indices):
    # Include current main-K/V and completed compressed-group writes in the
    # same captured transaction. Pooling raw-ring keys is deliberately outside
    # this test: these supplied compressed rows are already prepared keys.
    requests = case.token_to_req.long()
    logical = case.logical_positions
    blocks = case.block_table[requests, logical // MANAGER_BLOCK_SIZE].long()
    slots = blocks * MANAGER_BLOCK_SIZE + logical % MANAGER_BLOCK_SIZE
    qsa_store_cache_rows(case.caches["key"], slots, case.current_key)
    qsa_store_cache_rows(case.caches["value"], slots, case.current_value)
    group_positions = logical // COMPRESS_RATIO
    compressed_slots = blocks * COMPRESSED_BLOCK_SIZE + group_positions % COMPRESSED_BLOCK_SIZE
    compressed_slots = torch.where((logical + 1) % COMPRESS_RATIO == 0, compressed_slots, -1)
    qsa_store_cache_rows(case.caches["compressed"], compressed_slots, case.current_compressed_key)

    if select_indices:
        qsa_select_paged_tokens_lightning(
            case.index_query,
            case.caches["compressed"],
            case.block_table,
            case.token_to_req,
            case.logical_positions,
            case.seq_lens,
            case.query_start_loc,
            TOKEN_TOPK,
            COMPRESS_RATIO,
            case.sparse_indices,
        )
    qsa_gbsa_attention(
        case.query,
        case.caches["key"],
        case.caches["value"],
        case.sparse_indices[:, : GROUP_TOPK + COMPRESS_RATIO - 1],
        case.block_table,
        case.token_to_req,
        case.logical_positions,
        case.seq_lens,
        case.output,
    )


def _capture(case, initial_caches, select_indices, pool=None):
    payload = _payload(case.query.shape[0], iteration=0)
    _reset(case, initial_caches, payload)
    _execute(case, select_indices)
    torch.npu.synchronize()
    _reset(case, initial_caches, payload)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, pool=pool, capture_error_mode="thread_local", auto_dispatch_capture=True):
        _execute(case, select_indices)
    return graph


def _check_replay(eager, captured, graph, initial_caches, select_indices, iteration):
    payload = _payload(eager.query.shape[0], iteration)
    _reset(eager, initial_caches, payload)
    _reset(captured, initial_caches, payload)
    _execute(eager, select_indices)
    graph.replay()
    torch.npu.synchronize()
    assert torch.isfinite(captured.output).all(), f"nonfinite output at rows={eager.query.shape[0]}, replay={iteration}"
    torch.testing.assert_close(captured.output, eager.output, atol=ATOL, rtol=RTOL)
    if select_indices:
        # LI may permute equal-score groups. Compare the selected sets rather
        # than requiring the same order within those sets.
        eager_groups = eager.sparse_indices[:, :GROUP_TOPK].sort(dim=1).values
        captured_groups = captured.sparse_indices[:, :GROUP_TOPK].sort(dim=1).values
        torch.testing.assert_close(captured_groups, eager_groups, atol=0, rtol=0)
    else:
        torch.testing.assert_close(captured.sparse_indices, eager.sparse_indices, atol=0, rtol=0)
    for name in initial_caches:
        torch.testing.assert_close(captured.caches[name], eager.caches[name], atol=0, rtol=0)


@pytest.mark.parametrize("rows", [8, 32])
@pytest.mark.parametrize("select_indices", [True, False], ids=["select", "reuse"])
@torch.inference_mode()
def test_qsa_eager_matches_replay_with_updated_inputs(rows, select_indices):
    initial = _initial_caches()
    eager = _make_case(rows, {name: value.to("npu") for name, value in initial.items()})
    captured = _make_case(rows, {name: value.to("npu") for name, value in initial.items()})
    graph = _capture(captured, initial, select_indices)
    # Replay zero checks the capture payload; later replays change queries,
    # positions, sequence lengths, physical pages, sparse inputs and KV writes.
    for iteration in range(REPLAY_COUNT):
        _check_replay(eager, captured, graph, initial, select_indices, iteration)


@pytest.mark.parametrize("select_indices", [True, False], ids=["select", "reuse"])
@torch.inference_mode()
def test_qsa_interleaved_graphs_share_pool_and_persistent_caches(select_indices):
    initial = _initial_caches()
    eager_caches = {name: value.to("npu") for name, value in initial.items()}
    captured_caches = {name: value.to("npu") for name, value in initial.items()}
    pool = torch.npu.graph_pool_handle()
    cases = {}
    # Largest first, matching model graph capture. Both buckets retain the
    # same persistent cache pointers while their temporary allocations share a
    # graph pool. Replay also runs in the opposite order below.
    for rows in (32, 8):
        eager, captured = _make_case(rows, eager_caches), _make_case(rows, captured_caches)
        cases[rows] = (eager, captured, _capture(captured, initial, select_indices, pool=pool))
    for iteration, rows in enumerate((8, 32, 8, 32, 32, 8, 32, 8), start=1):
        eager, captured, graph = cases[rows]
        _check_replay(eager, captured, graph, initial, select_indices, iteration)
