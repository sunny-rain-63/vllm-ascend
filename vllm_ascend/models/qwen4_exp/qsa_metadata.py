# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

"""Keep persistent QSA metadata safe when a captured batch shrinks."""

import torch
from vllm.models.qwen4_exp.common import qsa_cache
from vllm.triton_utils import tl, triton

_SANITIZE_BLOCK_SIZE = 128


def _sanitize_qsa_metadata_torch(
    common, token_to_req_source, token_to_req, logical_positions, visible_blocks, slot_mapping_source, slot_mapping
):
    """CPU reference for the device kernel; no device scalar is read on host."""
    rows = torch.arange(token_to_req.numel(), device=token_to_req.device)
    num_reqs = common.query_start_loc.numel() - 1
    requests = torch.zeros_like(token_to_req, dtype=torch.int64)
    requests[: common.num_actual_tokens].copy_(token_to_req_source[: common.num_actual_tokens])
    slot_mapping[: common.num_actual_tokens].copy_(slot_mapping_source[: common.num_actual_tokens])
    safe_requests = requests.clamp(0, num_reqs - 1)
    starts = common.query_start_loc[safe_requests]
    ends = common.query_start_loc[safe_requests + 1]
    lengths = common.seq_lens[safe_requests]
    valid = (
        (rows < common.num_actual_tokens)
        & (rows < common.query_start_loc[-1])
        & (requests >= 0)
        & (requests < num_reqs)
        & (rows >= starts)
        & (rows < ends)
        & (logical_positions >= 0)
        & (logical_positions < lengths)
    )
    token_to_req.copy_(requests)
    token_to_req.masked_fill_(~valid, 0)
    logical_positions.masked_fill_(~valid, -1)
    visible_blocks.masked_fill_(~valid, 0)
    slot_mapping.masked_fill_(~valid, -1)


@triton.jit
def _sanitize_qsa_metadata_kernel(
    query_start_loc,
    seq_lens,
    token_to_req_source,
    token_to_req,
    logical_positions,
    visible_blocks,
    slot_mapping_source,
    slot_mapping,
    num_reqs,
    num_actual_tokens,
    num_input_tokens,
    BLOCK_SIZE: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    in_bounds = rows < num_input_tokens
    actual = in_bounds & (rows < num_actual_tokens)
    requests = tl.load(token_to_req_source + rows, actual, other=0)
    valid_request = (requests >= 0) & (requests < num_reqs)
    safe_requests = tl.minimum(tl.maximum(requests, 0), num_reqs - 1)
    starts = tl.load(query_start_loc + safe_requests, actual & valid_request, other=0)
    ends = tl.load(query_start_loc + safe_requests + 1, actual & valid_request, other=0)
    lengths = tl.load(seq_lens + safe_requests, actual & valid_request, other=0)
    positions = tl.load(logical_positions + rows, actual, other=-1)
    mapped_tokens = tl.load(query_start_loc + num_reqs)
    valid = (
        actual
        & valid_request
        & (rows < mapped_tokens)
        & (rows >= starts)
        & (rows < ends)
        & (positions >= 0)
        & (positions < lengths)
    )
    visible = tl.load(visible_blocks + rows, valid, other=0)
    slots = tl.load(slot_mapping_source + rows, valid, other=-1)
    tl.store(token_to_req + rows, tl.where(valid, requests, 0), in_bounds)
    tl.store(logical_positions + rows, tl.where(valid, positions, -1), in_bounds)
    tl.store(visible_blocks + rows, visible, in_bounds)
    tl.store(slot_mapping + rows, slots, in_bounds)


def build_qsa_metadata(
    common_attn_metadata,
    token_to_req_buffer,
    logical_positions_buffer,
    visible_blocks_buffer,
    slot_mapping_buffer,
    *,
    storage_block_size,
    compress_ratio,
    circular_buffer_size=0,
    k_work_metadata_buffer=None,
    request_capacity=None,
):
    """Preserve upstream metadata semantics and invalidate GPU-side padding.

    CPU query boundaries are fixed during graph capture. At replay the GPU
    boundaries and lengths can describe fewer tokens or zero-length requests.
    Also clear the padded model-input tail outside capture: an older, larger
    graph still holds views of these same persistent builder buffers.
    """
    num_actual_tokens = common_attn_metadata.num_actual_tokens
    num_input_tokens = getattr(common_attn_metadata, "num_input_tokens", num_actual_tokens)
    if num_input_tokens is None:
        num_input_tokens = num_actual_tokens
    if num_input_tokens < num_actual_tokens:
        raise ValueError("QSA model input cannot be shorter than its actual tokens")
    buffers = (token_to_req_buffer, logical_positions_buffer, visible_blocks_buffer, slot_mapping_buffer)
    if any(buffer.ndim != 1 or buffer.stride(0) != 1 or buffer.numel() < num_input_tokens for buffer in buffers):
        raise ValueError("QSA metadata buffer cannot hold the padded model input")
    token_to_req, logical_positions, visible_blocks, slot_mapping = qsa_cache._build_qsa_metadata_torch(
        common_attn_metadata,
        token_to_req_buffer,
        logical_positions_buffer,
        visible_blocks_buffer,
        slot_mapping_buffer,
        storage_block_size=storage_block_size,
        compress_ratio=compress_ratio,
        circular_buffer_size=circular_buffer_size,
        k_work_metadata_buffer=k_work_metadata_buffer,
        request_capacity=request_capacity,
    )
    if num_input_tokens == 0:
        return token_to_req, logical_positions, visible_blocks, slot_mapping
    if common_attn_metadata.query_start_loc.numel() < 2:
        raise ValueError("Nonempty QSA model inputs require at least one request")
    # The common token-map cache may belong to another builder and may be only
    # actual-sized. Never extend it or vary the captured output pointer: read
    # its actual rows and always write this builder's preallocated full buffer.
    token_to_req_full = token_to_req_buffer[:num_input_tokens]
    slot_mapping_full = slot_mapping_buffer[:num_input_tokens]
    logical_positions_full = logical_positions_buffer[:num_input_tokens]
    visible_blocks_full = visible_blocks_buffer[:num_input_tokens]
    if token_to_req.device.type == "cpu":
        _sanitize_qsa_metadata_torch(
            common_attn_metadata,
            token_to_req,
            token_to_req_full,
            logical_positions_full,
            visible_blocks_full,
            slot_mapping,
            slot_mapping_full,
        )
    else:
        _sanitize_qsa_metadata_kernel[(triton.cdiv(num_input_tokens, _SANITIZE_BLOCK_SIZE),)](
            common_attn_metadata.query_start_loc,
            common_attn_metadata.seq_lens,
            token_to_req,
            token_to_req_full,
            logical_positions_full,
            visible_blocks_full,
            slot_mapping,
            slot_mapping_full,
            common_attn_metadata.query_start_loc.numel() - 1,
            num_actual_tokens,
            num_input_tokens,
            BLOCK_SIZE=_SANITIZE_BLOCK_SIZE,
        )
    # Do not apply the circular group's common slot_mapping as a validity mask:
    # Ascend intentionally disables token slot mappings for ring-state groups.
    return (
        token_to_req_full[:num_actual_tokens],
        logical_positions,
        visible_blocks,
        slot_mapping_full[:num_actual_tokens],
    )
