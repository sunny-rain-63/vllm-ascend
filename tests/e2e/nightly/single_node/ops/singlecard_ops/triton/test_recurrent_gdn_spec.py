# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

import pytest
import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401

from vllm_ascend.ops.triton.fla import recurrent_gdn
from vllm_ascend.ops.triton.fla.recurrent_gdn import recurrent_gated_delta_rule_spec
from vllm_ascend.ops.triton.fla.spec_state_io import SPEC_STATE_IO_BLOCK_SIZE, prepare_spec_states_kernel


@pytest.mark.parametrize("width", [1, 3, 8])
@pytest.mark.parametrize("graph_mode", [False, True])
@torch.inference_mode()
def test_packed_indices_cover_empty_requests_and_graph_padding(width, graph_mode):
    num_reqs, row_size = 3, 4
    tokens = num_reqs * width - 1
    state = torch.zeros(2, row_size, device="npu")
    workspace = torch.empty(tokens + 1, row_size, device="npu")
    table = torch.zeros(num_reqs, width, dtype=torch.int32, device="npu")
    accepted = torch.ones(num_reqs, dtype=torch.int32, device="npu")
    starts = torch.zeros(num_reqs + 1, dtype=torch.int32, device="npu")
    lengths = torch.empty_like(starts)
    active = torch.zeros(tokens, dtype=torch.bool, device="npu")
    packed = torch.empty(tokens, dtype=torch.int32, device="npu")

    def run():
        prepare_spec_states_kernel[(1, num_reqs)](
            state,
            workspace,
            table,
            accepted,
            starts,
            lengths,
            active,
            packed,
            state.stride(0),
            table.stride(0),
            table.stride(1),
            NUM_STATES=state.shape[0],
            TOKENS=tokens,
            WIDTH=width,
            ROW_SIZE=row_size,
            BLOCK=SPEC_STATE_IO_BLOCK_SIZE,
        )

    run()
    if graph_mode:
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            run()
    # Reinitialize on every replay, even when all requests have zero length.
    for offsets in ([0, 0, 0, 0], [0, 0, width, 2 * width]):
        starts.copy_(torch.tensor(offsets, dtype=torch.int32))
        packed.fill_(-11)
        if graph_mode:
            graph.replay()
        else:
            run()
        torch.npu.synchronize()
        torch.testing.assert_close(packed.cpu(), torch.arange(1, tokens + 1, dtype=torch.int32))
        torch.testing.assert_close(
            lengths.cpu(), torch.tensor([0, *torch.diff(starts.cpu()).tolist()], dtype=torch.int32)
        )


def reference_recurrent(query, key, value, state, g, beta, scale, starts, indices, accepted):
    """Independent scalar-token recurrence; state is [block, head, V, K]."""
    output = torch.zeros_like(value)
    grouped_heads = value.shape[1] // query.shape[1]
    for req, count in enumerate(accepted.tolist()):
        begin, end = starts[req : req + 2].tolist()
        if begin == end or count <= 0:
            continue
        initial_id = int(indices[req, count - 1])
        if initial_id <= 0:
            continue
        # Load once before any writes, including when the source is one of
        # this step's destinations. Accepted counts belong to the LAST step.
        h = state[initial_id].float().clone()
        for token in range(begin, end):
            q = query[token].float().repeat_interleave(grouped_heads, dim=0)
            k = key[token].float().repeat_interleave(grouped_heads, dim=0)
            h *= g[token].float().exp()[:, None, None]
            residual = value[token].float() - torch.matmul(h, k.unsqueeze(-1)).squeeze(-1)
            h += (residual * beta[token].float()[:, None]).unsqueeze(-1) * k.unsqueeze(-2)
            output[token] = (torch.matmul(h, q.unsqueeze(-1)).squeeze(-1) * scale).to(value.dtype)
            destination = int(indices[req, token - begin])
            if destination > 0:
                state[destination] = h.to(state.dtype)
    return output


@pytest.mark.parametrize("lengths", [(8, 8, 0), (2, 8, 0), (2, 2, 4)])
@pytest.mark.parametrize("padding", [0, 256])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("graph_mode", [False, True])
@pytest.mark.parametrize("token_padding", [0, 3])
@torch.inference_mode()
def test_spec_recurrence_preserves_rows_and_padding(lengths, padding, dtype, graph_mode, token_padding):
    torch.manual_seed(20261008)
    device = "npu"
    heads, value_heads, key_dim, value_dim = 2, 4, 128, 128
    tokens = sum(lengths) + token_padding
    payload = value_heads * value_dim * key_dim
    # A prefix/suffix guard, nonzero storage offset and unselected rows detect
    # writes outside the selected state pages, including shared-page padding.
    guard, num_blocks = 32, 27
    backing_cpu = torch.full((2 * guard + num_blocks * (payload + padding),), -73.0)

    def state_view(storage):
        return torch.as_strided(
            storage,
            (num_blocks, value_heads, value_dim, key_dim),
            (payload + padding, value_dim * key_dim, key_dim, 1),
            storage_offset=guard,
        )

    state_view(backing_cpu).copy_(torch.randn(num_blocks, value_heads, value_dim, key_dim) * 0.1)
    backing = backing_cpu.to(device)
    state = state_view(backing)
    index_backing = torch.full((3, 16), -1, dtype=torch.int32, device=device)
    indices = index_backing[:, ::2]  # Both table strides matter.
    table_cpu = torch.arange(1, 25, dtype=torch.int32).flip(0).reshape(3, 8)
    table_cpu[-1].zero_()  # Graph padding, with both zero and nonzero lengths.
    indices.copy_(table_cpu)
    starts = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32, device=device)
    accepted = torch.tensor([8, 5, 1], dtype=torch.int32, device=device)
    query = torch.empty(tokens, heads, key_dim, dtype=dtype, device=device)
    key = torch.empty_like(query)
    value = torch.empty(tokens, value_heads, value_dim, dtype=dtype, device=device)
    g = torch.empty(tokens, value_heads, dtype=torch.float32, device=device)
    beta = torch.empty(tokens, value_heads, dtype=dtype, device=device)

    def randomize_inputs():
        query.copy_(F.normalize(torch.randn(query.shape), dim=-1).to(dtype))
        key.copy_(F.normalize(torch.randn(key.shape), dim=-1).to(dtype))
        value.copy_(torch.randn(value.shape).to(dtype))
        g.copy_(-torch.rand(g.shape))
        beta.copy_(torch.rand(beta.shape).to(dtype))

    def run():
        return recurrent_gated_delta_rule_spec(
            query,
            key,
            value,
            state,
            g=g,
            beta=beta,
            scale=key_dim**-0.5,
            query_start_loc=starts,
            ssm_state_indices=indices,
            num_accepted_tokens=accepted,
        )

    randomize_inputs()
    run()  # Compile before capture; reset any warmup/capture state writes below.
    if graph_mode:
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            output = run()
    backing.copy_(backing_cpu)
    for step in range(3):
        randomize_inputs()
        step_lengths = lengths if step != 1 else tuple(reversed(lengths))
        starts.copy_(torch.tensor([0, *torch.tensor(step_lengths).cumsum(0).tolist()], dtype=torch.int32))
        # Acceptance changes without recapture, and may exceed this step's
        # query length. Swap physical row ownership between requests as well.
        accepted.copy_(torch.tensor(([8, 5, 1], [1, 8, 1], [2, 3, 1])[step], dtype=torch.int32))
        indices.copy_(table_cpu if step != 1 else table_cpu[[1, 0, 2]])
        expected_backing = backing.cpu().clone()
        expected = reference_recurrent(
            query.cpu(),
            key.cpu(),
            value.cpu(),
            state_view(expected_backing),
            g.cpu(),
            beta.cpu(),
            key_dim**-0.5,
            starts.cpu(),
            indices.cpu(),
            accepted.cpu(),
        )
        if graph_mode:
            graph.replay()
        else:
            output = run()
        torch.npu.synchronize()
        tolerance = 2e-2 if dtype == torch.bfloat16 else 2e-3
        torch.testing.assert_close(output.cpu(), expected, rtol=tolerance, atol=tolerance)
        torch.testing.assert_close(backing.cpu(), expected_backing, rtol=2e-4, atol=2e-4)
        # Sentinels must be EXACTLY preserved, even if numerical tolerances
        # above would allow an unintended small change to an unused page.
        untouched = expected_backing == -73
        assert torch.equal(backing.cpu()[untouched], expected_backing[untouched])


@pytest.mark.parametrize("num_reqs", [1, 4, 16])
@torch.inference_mode()
def test_spec_recurrence_graph_latency(num_reqs, monkeypatch, record_property):
    """Report fused-vs-previous-path latency on the same padded cache.

    Run with pytest -s to see timings. No hardware-independent speed threshold
    is assumed; this benchmark is also a numerical parity check before timing.
    """
    torch.manual_seed(17)
    heads, value_heads, dim, width = 16, 32, 128, 8
    tokens = num_reqs * width
    payload = value_heads * dim * dim
    storage = torch.randn(tokens + 1, payload + 256, device="npu") * 0.01
    state = storage[:, :payload].view(tokens + 1, value_heads, dim, dim)
    initial = storage.clone()
    query = F.normalize(torch.randn(tokens, heads, dim, device="npu"), dim=-1).bfloat16()
    key = F.normalize(torch.randn_like(query).float(), dim=-1).bfloat16()
    value = torch.randn(tokens, value_heads, dim, dtype=torch.bfloat16, device="npu")
    g = -torch.rand(tokens, value_heads, device="npu")
    beta = torch.rand(tokens, value_heads, dtype=torch.bfloat16, device="npu")
    starts = torch.arange(0, tokens + 1, width, dtype=torch.int32, device="npu")
    indices = torch.arange(1, tokens + 1, dtype=torch.int32, device="npu").view(num_reqs, width)
    accepted = torch.full((num_reqs,), width, dtype=torch.int32, device="npu")

    def run():
        return recurrent_gated_delta_rule_spec(
            query,
            key,
            value,
            state,
            g=g,
            beta=beta,
            scale=dim**-0.5,
            query_start_loc=starts,
            ssm_state_indices=indices,
            num_accepted_tokens=accepted,
        )

    results, timings = [], []
    for label, limit in (("previous_triton", 0), ("fused_fla", width)):
        monkeypatch.setattr(recurrent_gdn, "_FLA_MAX_QUERY_LEN", limit)
        storage.copy_(initial)
        output = run()
        torch.npu.synchronize()
        results.append((output.cpu(), storage.cpu()))
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            run()
        for _ in range(3):
            graph.replay()
        start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
        start.record()
        for _ in range(30):
            graph.replay()
        end.record()
        torch.npu.synchronize()
        elapsed_ms = start.elapsed_time(end) / 30
        record_property(f"{label}_ms", elapsed_ms)
        timings.append(elapsed_ms)
        del graph
    torch.testing.assert_close(results[0][0], results[1][0], rtol=2e-2, atol=2e-3)
    torch.testing.assert_close(results[0][1], results[1][1], rtol=2e-4, atol=2e-4)
    print(f"requests={num_reqs}: previous={timings[0]:.3f} ms, fused={timings[1]:.3f} ms")
