# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from vllm.triton_utils import tl, triton

SPEC_TOKEN_BLOCK_SIZE = 512


@triton.jit
def prepare_spec_metadata_kernel(
    starts,
    table,
    accepted,
    lengths,
    packed_starts,
    packed_accepted,
    table_row_stride,
    table_col_stride,
    NUM_REQS: tl.constexpr,
    NUM_STATES: tl.constexpr,
    WIDTH: tl.constexpr,
    REQ_BLOCK: tl.constexpr,
    WIDTH_BLOCK: tl.constexpr,
):
    req = tl.arange(0, REQ_BLOCK)
    begin = tl.load(starts + req, req < NUM_REQS, other=0)
    end = tl.load(starts + req + 1, req < NUM_REQS, other=0)
    count = tl.load(accepted + req, req < NUM_REQS, other=0)
    valid = (req < NUM_REQS) & (end > begin) & (count > 0) & (count <= WIDTH)
    source = tl.load(table + req * table_row_stride + (count - 1) * table_col_stride, valid, other=0)
    valid &= (source > 0) & (source < NUM_STATES)
    col = tl.arange(0, WIDTH_BLOCK)
    ids = tl.load(
        table + req[:, None] * table_row_stride + col[None, :] * table_col_stride,
        (req[:, None] < NUM_REQS) & (col[None, :] < WIDTH),
        other=0,
    )
    # Real requests own every live output row; null graph rows own none.
    valid &= (
        tl.sum(((ids > 0) & (ids < NUM_STATES) & (col[None, :] < (end - begin)[:, None])).to(tl.int32), 1)
        == end - begin
    )
    size = tl.where(valid, end - begin, 0)
    prefix = tl.cumsum(size, 0)
    tl.store(lengths, 0)
    tl.store(lengths + req + 1, size, req < NUM_REQS)
    tl.store(packed_starts, 0)
    tl.store(packed_starts + req + 1, prefix, req < NUM_REQS)
    # FLA bounds acceptance by this step's length. Only out-of-range prior
    # acceptance needs an initial-state copy into the first destination row.
    tl.store(packed_accepted + req, tl.where(valid & (count <= size), count, 1), req < NUM_REQS)


@triton.jit
def pack_spec_inputs_kernel(
    query,
    key,
    value,
    g,
    beta,
    state,
    table,
    accepted,
    starts,
    lengths,
    packed_starts,
    packed_query,
    packed_key,
    packed_value,
    packed_g,
    packed_beta,
    packed_indices,
    state_stride: tl.int64,
    table_row_stride,
    table_col_stride,
    Q_STRIDES: tl.constexpr,
    K_STRIDES: tl.constexpr,
    V_STRIDES: tl.constexpr,
    G_STRIDES: tl.constexpr,
    B_STRIDES: tl.constexpr,
    WIDTH: tl.constexpr,
    Q_SIZE: tl.constexpr,
    V_SIZE: tl.constexpr,
    K_DIM: tl.constexpr,
    V_DIM: tl.constexpr,
    VALUE_HEADS: tl.constexpr,
    STATE_SIZE: tl.constexpr,
    NUM_TILES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    tile, slot = tl.program_id(0), tl.program_id(1)
    req, col = slot // WIDTH, slot % WIDTH
    size = tl.load(lengths + req + 1)
    if col < size:
        token = tl.load(starts + req).to(tl.int64) + col
        packed_token = tl.load(packed_starts + req).to(tl.int64) + col
        offsets = tile * BLOCK + tl.arange(0, BLOCK)
        q = tl.load(
            query + token * Q_STRIDES[0] + offsets // K_DIM * Q_STRIDES[1] + offsets % K_DIM * Q_STRIDES[2],
            offsets < Q_SIZE,
            other=0,
        )
        k = tl.load(
            key + token * K_STRIDES[0] + offsets // K_DIM * K_STRIDES[1] + offsets % K_DIM * K_STRIDES[2],
            offsets < Q_SIZE,
            other=0,
        )
        v = tl.load(
            value + token * V_STRIDES[0] + offsets // V_DIM * V_STRIDES[1] + offsets % V_DIM * V_STRIDES[2],
            offsets < V_SIZE,
            other=0,
        )
        tl.store(packed_query + packed_token * Q_SIZE + offsets, q, offsets < Q_SIZE)
        tl.store(packed_key + packed_token * Q_SIZE + offsets, k, offsets < Q_SIZE)
        tl.store(packed_value + packed_token * V_SIZE + offsets, v, offsets < V_SIZE)
        gate = tl.load(g + token * G_STRIDES[0] + offsets * G_STRIDES[1], offsets < VALUE_HEADS, other=0)
        b = tl.load(beta + token * B_STRIDES[0] + offsets * B_STRIDES[1], offsets < VALUE_HEADS, other=0)
        tl.store(packed_g + packed_token * VALUE_HEADS + offsets, gate, offsets < VALUE_HEADS)
        tl.store(packed_beta + packed_token * VALUE_HEADS + offsets, b, offsets < VALUE_HEADS)
        if tile == 0:
            destination = tl.load(table + req * table_row_stride + col * table_col_stride)
            tl.store(packed_indices + packed_token, destination)
        if col == 0:
            count = tl.load(accepted + req)
            if count > size:
                source = tl.load(table + req * table_row_stride + (count - 1) * table_col_stride).to(tl.int64)
                destination = tl.load(table + req * table_row_stride).to(tl.int64)
                # Request state rows are disjoint. The source lies beyond this
                # step's outputs, and each tile copies a disjoint row segment.
                for base in range(tile * BLOCK, STATE_SIZE, NUM_TILES * BLOCK):
                    state_offsets = base + tl.arange(0, BLOCK)
                    values = tl.load(state + source * state_stride + state_offsets, state_offsets < STATE_SIZE, other=0)
                    tl.store(state + destination * state_stride + state_offsets, values, state_offsets < STATE_SIZE)


@triton.jit
def unpack_spec_output_kernel(
    packed_output,
    output,
    starts,
    lengths,
    packed_starts,
    NUM_REQS: tl.constexpr,
    TOKENS: tl.constexpr,
    WIDTH: tl.constexpr,
    V_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    tile, slot = tl.program_id(0), tl.program_id(1)
    req, col = slot // WIDTH, slot % WIDTH
    begin = tl.load(starts + req).to(tl.int64)
    end = tl.load(starts + req + 1).to(tl.int64)
    size = tl.load(lengths + req + 1)
    packed_token = tl.load(packed_starts + req).to(tl.int64) + col
    offsets = tile * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(packed_output + packed_token * V_SIZE + offsets, (col < size) & (offsets < V_SIZE), other=0)
    tl.store(output + (begin + col) * V_SIZE + offsets, values, (begin + col < end) & (offsets < V_SIZE))
    # The fixed-width slots also partition all possible trailing graph tokens.
    total = tl.load(starts + NUM_REQS)
    tl.store(output + slot * V_SIZE + offsets, 0, (slot >= total) & (slot < TOKENS) & (offsets < V_SIZE))
