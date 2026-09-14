# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _mask_dflash_slots_kernel(
    slots_ptr,
    output_ptr,
    num_slots,
    null_block_size,
    slot_limit,
    pad_slot_id,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    in_bounds = offsets < num_slots
    slots = tl.load(slots_ptr + offsets, mask=in_bounds, other=pad_slot_id)
    valid = (slots >= null_block_size) & (slots < slot_limit)
    tl.store(output_ptr + offsets, tl.where(valid, slots, pad_slot_id), mask=in_bounds)


def mask_dflash_slots(
    slots: torch.Tensor,
    null_block_size: int,
    slot_limit: int,
    pad_slot_id: int,
) -> torch.Tensor:
    """Preserve the mixed-cache write guard with one NPU kernel per write.

    Never mutate the source mapping or retain a result across invocations:
    graph replay and successive decoding steps reuse these input buffers.
    Keep the original PyTorch expression for CPU and nonstandard layouts.
    """
    if (
        slots.device.type != "npu"
        or slots.ndim != 1
        or not slots.is_contiguous()
        or slots.dtype not in (torch.int32, torch.int64)
    ):
        valid = (slots >= null_block_size) & (slots < slot_limit)
        return torch.where(valid, slots, pad_slot_id)

    output = torch.empty_like(slots)
    num_slots = slots.numel()
    if num_slots:
        block_size = 256
        _mask_dflash_slots_kernel[(triton.cdiv(num_slots, block_size),)](
            slots,
            output,
            num_slots,
            null_block_size,
            slot_limit,
            pad_slot_id,
            BLOCK_SIZE=block_size,
        )
    return output
