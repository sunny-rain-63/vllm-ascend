# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

import pytest
import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401

from vllm_ascend.ops.triton.fla import recurrent_gdn
from vllm_ascend.ops.triton.fla.recurrent_gdn import recurrent_gated_delta_rule_spec
from vllm_ascend.ops.triton.fla.spec_state_io import (
    SPEC_STATE_IO_BLOCK_SIZE,
    prepare_spec_states_kernel,
    scatter_spec_states_kernel,
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


@pytest.mark.parametrize("lengths", [(8, 8, 0), (2, 8, 0), (2, 2, 4), (0, 0, 0)])
@pytest.mark.parametrize("padding", [0, 256])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("graph_mode", [False, True])
@pytest.mark.parametrize("token_padding", [0, 3])
@pytest.mark.parametrize("input_stride", [1, 2])
@torch.inference_mode()
def test_spec_recurrence_preserves_rows_and_padding(lengths, padding, dtype, graph_mode, token_padding, input_stride):
    torch.manual_seed(20261008)
    device = "npu"
    heads, value_heads, key_dim, value_dim = 2, 4, 128, 128
    tokens = sum(lengths) + token_padding
    if tokens == 0:
        pytest.skip("The zero-token early return is covered by the unit test.")
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
    query = torch.empty(tokens, heads, key_dim * input_stride, dtype=dtype, device=device)[..., ::input_stride]
    key = torch.empty(tokens, heads, key_dim * input_stride, dtype=dtype, device=device)[..., ::input_stride]
    value = torch.empty(tokens, value_heads, value_dim * input_stride, dtype=dtype, device=device)[..., ::input_stride]
    g = torch.empty(tokens, value_heads * input_stride, dtype=torch.float32, device=device)[:, ::input_stride]
    beta = torch.empty(tokens, value_heads * input_stride, dtype=dtype, device=device)[:, ::input_stride]

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
def test_spec_recurrence_graph_latency(num_reqs, record_property):
    """Compare the 11682d1e3 full-width call, staged adapter and direct adapter.

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
    baseline_state = state.clone().contiguous()
    baseline_initial = baseline_state.clone()
    baseline_lengths = torch.tensor([0, *([width] * num_reqs)], dtype=torch.int32, device="npu")

    def baseline():
        # The old flattened ABI is valid for this full-width, unpadded batch.
        return recurrent_gdn.recurrent_gated_delta_rule(
            query,
            key,
            value,
            baseline_state,
            g=g,
            beta=beta,
            scale=dim**-0.5,
            actual_seq_lengths=baseline_lengths,
            ssm_state_indices=indices.flatten(),
            num_accepted_tokens=accepted,
        )

    def staged():
        # Reproduce the 984ccc8 adapter for an apples-to-apples state-IO cost.
        workspace = torch.empty(tokens + 1, value_heads, dim, dim, device="npu")
        lengths = torch.empty(num_reqs + 1, dtype=torch.int32, device="npu")
        packed_indices = torch.arange(1, tokens + 1, dtype=torch.int32, device="npu")
        active = torch.zeros(tokens, dtype=torch.bool, device="npu")
        tiles = (payload + SPEC_STATE_IO_BLOCK_SIZE - 1) // SPEC_STATE_IO_BLOCK_SIZE
        prepare_spec_states_kernel[(tiles, num_reqs)](
            state,
            workspace,
            indices,
            accepted,
            starts,
            lengths,
            active,
            state.stride(0),
            indices.stride(0),
            indices.stride(1),
            NUM_STATES=state.shape[0],
            WIDTH=width,
            ROW_SIZE=payload,
            BLOCK=SPEC_STATE_IO_BLOCK_SIZE,
        )
        result = recurrent_gdn.recurrent_gated_delta_rule(
            query,
            key,
            value,
            workspace,
            g=g,
            beta=beta,
            scale=dim**-0.5,
            actual_seq_lengths=lengths,
            ssm_state_indices=packed_indices,
            num_accepted_tokens=None,
        )
        scatter_spec_states_kernel[(tiles, num_reqs * width)](
            workspace,
            state,
            indices,
            starts,
            active,
            result,
            state.stride(0),
            indices.stride(0),
            indices.stride(1),
            NUM_STATES=state.shape[0],
            TOKENS=tokens,
            WIDTH=width,
            ROW_SIZE=payload,
            OUTPUT_ROW_SIZE=value_heads * dim,
            BLOCK=SPEC_STATE_IO_BLOCK_SIZE,
        )
        return result

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
    for label, operation in (("baseline_11682", baseline), ("staged_984ccc8", staged), ("direct_strided", run)):
        storage.copy_(initial)
        baseline_state.copy_(baseline_initial)
        output = operation()
        torch.npu.synchronize()
        results.append((output.cpu(), (baseline_state if label == "baseline_11682" else state).cpu()))
        assert torch.equal(storage[:, payload:].cpu(), initial[:, payload:].cpu())
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            operation()
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
    for result in results[1:]:
        torch.testing.assert_close(results[0][0], result[0], rtol=2e-2, atol=2e-3)
        torch.testing.assert_close(results[0][1], result[1], rtol=2e-4, atol=2e-4)
    record_property("direct_to_baseline_ratio", timings[2] / timings[0])
    print(f"requests={num_reqs}: baseline={timings[0]:.3f} ms, staged={timings[1]:.3f} ms, direct={timings[2]:.3f} ms")
