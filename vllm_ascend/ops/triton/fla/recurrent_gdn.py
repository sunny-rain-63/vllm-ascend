# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

import torch
from fla_npu.ops.ascendc import recurrent_gated_delta_rule
from vllm.triton_utils import triton

from vllm_ascend.ops.triton.fla.spec_state_io import (
    SPEC_STATE_IO_BLOCK_SIZE,
    prepare_spec_states_kernel,
    scatter_spec_states_kernel,
)
from vllm_ascend.ops.triton.kda.fused_recurrent_kda import fused_recurrent_gated_delta_rule_fwd_kernel

_FLA_MAX_QUERY_LEN = 8


def recurrent_gated_delta_rule_spec(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    state: torch.Tensor,
    *,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    query_start_loc: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
) -> torch.Tensor:
    """Verify ragged queries using a fixed-width table of speculative states.

    State r starts at table[r, accepted[r] - 1] from the previous step. The
    current token t writes table[r, t], even if a preceding request verifies
    fewer tokens or accepted[r] exceeds the current query length. Flattening
    the table and indexing it by packed token offsets violates this contract.

    Gather just the selected initial rows into a batch-sized dense workspace
    for FLA, then scatter the token states back using the original page stride.
    Neither the cache allocation nor its logical/physical page layout changes.
    The previous Triton path remains available for shapes/dtypes outside FLA's
    eight-token FP16/BF16 contract.
    """
    if query.ndim != 3 or query.shape != key.shape or value.ndim != 3:
        raise ValueError("Expected Q/K [tokens, heads, key_dim] and V [tokens, value_heads, value_dim].")
    tokens, heads, key_dim = query.shape
    value_heads, value_dim = value.shape[1:]
    if value.shape[0] != tokens or heads < 1 or value_heads < 1 or value_heads % heads or min(key_dim, value_dim) < 1:
        raise ValueError("Q/K and V must share tokens and have compatible grouped heads.")
    if state.ndim != 4 or state.shape[1:] != (value_heads, value_dim, key_dim):
        raise ValueError("Expected state [blocks, value_heads, value_dim, key_dim].")
    if state.shape[0] == 0 or not state[0].is_contiguous() or state.stride(0) < value_heads * value_dim * key_dim:
        raise ValueError("State rows must be dense and non-overlapping; only the block stride may be padded.")
    if query_start_loc.ndim != 1 or query_start_loc.numel() < 1:
        raise ValueError("query_start_loc must contain cumulative query offsets.")
    num_reqs = query_start_loc.numel() - 1
    if ssm_state_indices.ndim != 2 or ssm_state_indices.shape[0] != num_reqs or ssm_state_indices.shape[1] < 1:
        raise ValueError("Speculative state indices must be a fixed-width [requests, states] table.")
    if num_accepted_tokens.shape != (num_reqs,):
        raise ValueError("Expected one previous acceptance count per request.")
    if g.shape != (tokens, value_heads) or beta.shape != g.shape:
        raise ValueError("Expected scalar gates and beta [tokens, value_heads].")
    for tensor in (key, value, state, g, beta, query_start_loc, ssm_state_indices, num_accepted_tokens):
        if tensor.device != query.device:
            raise ValueError("All recurrent inputs must be on the same device.")
    for tensor in (query_start_loc, ssm_state_indices, num_accepted_tokens):
        if tensor.dtype not in (torch.int32, torch.int64):
            raise TypeError("Recurrent offsets, state indices and acceptance counts must be integers.")

    # Null/zero-length graph rows may be skipped by the kernel. Never expose
    # uninitialized output from those rows on graph capture or replay.
    if tokens == 0 or num_reqs == 0:
        return torch.zeros(value.shape, dtype=value.dtype, device=value.device)
    if (
        ssm_state_indices.shape[1] <= _FLA_MAX_QUERY_LEN
        and tokens <= num_reqs * ssm_state_indices.shape[1]
        and query.dtype in (torch.float16, torch.bfloat16)
        and key.dtype == value.dtype == beta.dtype == query.dtype
        and state.dtype == g.dtype == torch.float32
    ):
        width = ssm_state_indices.shape[1]
        row_size = value_heads * value_dim * key_dim
        # Only this batch's token states are staged, never the entire cache.
        # FLA initializes every live row, so only initial rows need gathering.
        workspace = torch.empty((tokens + 1, *state.shape[1:]), dtype=state.dtype, device=state.device)
        lengths = torch.empty(num_reqs + 1, dtype=torch.int32, device=state.device)
        packed_indices = torch.empty(tokens, dtype=torch.int32, device=state.device)
        active = torch.zeros(tokens, dtype=torch.bool, device=state.device)
        starts = query_start_loc.contiguous()
        prepare_spec_states_kernel[(triton.cdiv(row_size, SPEC_STATE_IO_BLOCK_SIZE), num_reqs)](
            state,
            workspace,
            ssm_state_indices,
            num_accepted_tokens.contiguous(),
            starts,
            lengths,
            active,
            packed_indices,
            state.stride(0),
            ssm_state_indices.stride(0),
            ssm_state_indices.stride(1),
            NUM_STATES=state.shape[0],
            TOKENS=tokens,
            WIDTH=width,
            ROW_SIZE=row_size,
            BLOCK=SPEC_STATE_IO_BLOCK_SIZE,
        )
        # Acceptance was resolved while gathering, not clamped to this step's
        # query length. FLA sees a genuine packed table and needs no acceptance.
        output = recurrent_gated_delta_rule(
            query,
            key,
            value,
            workspace,
            g=g,
            beta=beta,
            scale=scale,
            actual_seq_lengths=lengths,
            ssm_state_indices=packed_indices,
            num_accepted_tokens=None,
        )
        scatter_spec_states_kernel[(triton.cdiv(row_size, SPEC_STATE_IO_BLOCK_SIZE), num_reqs * width)](
            workspace,
            state,
            ssm_state_indices,
            starts,
            active,
            output,
            state.stride(0),
            ssm_state_indices.stride(0),
            ssm_state_indices.stride(1),
            NUM_STATES=state.shape[0],
            TOKENS=tokens,
            WIDTH=width,
            ROW_SIZE=row_size,
            OUTPUT_ROW_SIZE=value_heads * value_dim,
            BLOCK=SPEC_STATE_IO_BLOCK_SIZE,
        )
        return output
    output = torch.zeros(value.shape, dtype=value.dtype, device=value.device)
    block_v = min(triton.next_power_of_2(value_dim), 8)
    grid = (1, triton.cdiv(value_dim, block_v), num_reqs * value_heads)
    fused_recurrent_gated_delta_rule_fwd_kernel[grid](
        q=query.contiguous(),
        k=key.contiguous(),
        v=value.contiguous(),
        g=g.contiguous(),
        beta=beta.contiguous(),
        o=output,
        h0=state,
        ht=state,
        cu_seqlens=query_start_loc.contiguous(),
        ssm_state_indices=ssm_state_indices,
        num_accepted_tokens=num_accepted_tokens.contiguous(),
        scale=scale,
        N=num_reqs,
        T=tokens,
        B=1,
        H=heads,
        HV=value_heads,
        K=key_dim,
        V=value_dim,
        BK=triton.next_power_of_2(key_dim),
        BV=block_v,
        stride_init_state_token=state.stride(0),
        stride_final_state_token=state.stride(0),
        stride_indices_seq=ssm_state_indices.stride(0),
        stride_indices_tok=ssm_state_indices.stride(1),
        IS_BETA_HEADWISE=False,
        USE_QK_L2NORM_IN_KERNEL=False,
        INPLACE_FINAL_STATE=True,
        IS_KDA=False,
        SPEC_STATE_WIDTH=ssm_state_indices.shape[1],
        num_warps=1,
        num_stages=3,
    )
    return output
