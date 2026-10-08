# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from vllm.triton_utils import tl, triton

# Match the selected-row IO granularity used by mamba/state_index.py.
SPEC_STATE_IO_BLOCK_SIZE = 1024


@triton.jit
def prepare_spec_states_kernel(
    state,
    workspace,
    table,
    accepted,
    starts,
    lengths,
    active,
    packed_indices,
    state_stride: tl.int64,
    table_row_stride,
    table_col_stride,
    NUM_STATES: tl.constexpr,
    TOKENS: tl.constexpr,
    WIDTH: tl.constexpr,
    ROW_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    tile, req = tl.program_id(0), tl.program_id(1)
    begin = tl.load(starts + req).to(tl.int64)
    end = tl.load(starts + req + 1).to(tl.int64)
    count = tl.load(accepted + req)
    valid = (end > begin) & (count > 0) & (count <= WIDTH)
    source = tl.load(table + req * table_row_stride + (count - 1) * table_col_stride, mask=valid, other=0).to(tl.int64)
    valid &= (source > 0) & (source < NUM_STATES)
    offsets = tile * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(state + source * state_stride + offsets, mask=(offsets < ROW_SIZE) & valid, other=0)
    # FLA reads the first packed token's row as its initial state. Row zero
    # stays reserved; all real scratch indices are token_index + 1.
    tl.store(workspace + (begin + 1) * ROW_SIZE + offsets, values, mask=(offsets < ROW_SIZE) & (end > begin))
    if tile == 0:
        tl.store(lengths + req + 1, end - begin)
        if req == 0:
            tl.store(lengths, 0)
        for col in range(WIDTH):
            tl.store(active + begin + col, valid, mask=begin + col < end)
            # Fixed-width slots partition all TOKENS, including graph padding
            # and empty requests, without depending on the live query lengths.
            slot = req * WIDTH + col
            tl.store(packed_indices + slot, slot + 1, mask=slot < TOKENS)


@triton.jit
def scatter_spec_states_kernel(
    workspace,
    state,
    table,
    starts,
    active,
    output,
    state_stride: tl.int64,
    table_row_stride,
    table_col_stride,
    NUM_STATES: tl.constexpr,
    TOKENS: tl.constexpr,
    WIDTH: tl.constexpr,
    ROW_SIZE: tl.constexpr,
    OUTPUT_ROW_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    tile, slot = tl.program_id(0), tl.program_id(1)
    req, col = slot // WIDTH, slot % WIDTH
    begin = tl.load(starts + req).to(tl.int64)
    end = tl.load(starts + req + 1).to(tl.int64)
    token = begin + col
    valid = tl.load(active + token, mask=token < end, other=0).to(tl.int1)
    destination = tl.load(table + req * table_row_stride + col * table_col_stride, mask=valid, other=0).to(tl.int64)
    valid &= (destination > 0) & (destination < NUM_STATES)
    offsets = tile * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(workspace + (token + 1) * ROW_SIZE + offsets, mask=(offsets < ROW_SIZE) & valid, other=0)
    tl.store(state + destination * state_stride + offsets, values, mask=(offsets < ROW_SIZE) & valid)
    # Also zero dummy requests and graph-token padding. FLA may leave trailing
    # output uninitialized. This small store shares the state-scatter launch.
    if tile * BLOCK < OUTPUT_ROW_SIZE:
        live = tl.load(active + slot, mask=slot < TOKENS, other=0).to(tl.int1)
        tl.store(
            output + slot * OUTPUT_ROW_SIZE + offsets, 0, mask=(slot < TOKENS) & ~live & (offsets < OUTPUT_ROW_SIZE)
        )
