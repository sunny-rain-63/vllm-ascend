# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch

from vllm_ascend.ops.triton.mamba.state_index import (
    STATE_IO_BLOCK_SIZE,
    gather_ssm_states,
    scatter_ssm_states_,
)

pytestmark = pytest.mark.skipif(
    not hasattr(torch, "npu") or not torch.npu.is_available(),
    reason="Mamba state index tests require an Ascend NPU",
)


@pytest.mark.parametrize("padding", [0, 13])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize(
    "state_shape",
    [
        (1, 1, STATE_IO_BLOCK_SIZE - 1),
        (1, 1, STATE_IO_BLOCK_SIZE),
        (1, 1, STATE_IO_BLOCK_SIZE + 1),
        (8, 128, 128),
    ],
)
def test_gather_and_scatter_ssm_states(
    padding: int,
    index_dtype: torch.dtype,
    state_shape: tuple[int, ...],
) -> None:
    num_states = 5
    row_size = math.prod(state_shape)
    state_stride = row_size + padding
    storage_size = (num_states - 1) * state_stride + row_size
    inner_strides = []
    running_stride = 1
    for dim in reversed(state_shape):
        inner_strides.append(running_stride)
        running_stride *= dim
    inner_strides.reverse()

    storage = torch.full(
        (storage_size,),
        -1.0,
        dtype=torch.float32,
        device="npu",
    )
    state = torch.as_strided(
        storage,
        size=(num_states, *state_shape),
        stride=(state_stride, *inner_strides),
    )
    for state_idx in range(num_states):
        state[state_idx].fill_(state_idx + 0.25)

    indices = torch.tensor([3, 0, 2], dtype=index_dtype, device="npu")
    has_initial_state = torch.tensor([True, False, True], device="npu")

    gathered = gather_ssm_states(
        state,
        indices,
        has_initial_state,
        output_dtype=torch.bfloat16,
    )
    expected_gather = torch.stack([state[3], torch.zeros_like(state[0]), state[2]]).to(torch.bfloat16)
    torch.testing.assert_close(gathered, expected_gather, rtol=0, atol=0)

    source = (
        torch.arange(
            indices.numel() * row_size,
            dtype=torch.float32,
            device="npu",
        )
        .reshape(indices.numel(), *state_shape)
        .to(torch.bfloat16)
    )
    scatter_ssm_states_(state, indices, source)

    for source_idx, state_idx in enumerate((3, 0, 2)):
        torch.testing.assert_close(
            state[state_idx],
            source[source_idx].to(state.dtype),
            rtol=0,
            atol=0,
        )
    torch.testing.assert_close(
        state[1],
        torch.full_like(state[1], 1.25),
        rtol=0,
        atol=0,
    )

    if padding:
        storage_cpu = storage.cpu()
        for state_idx in range(num_states - 1):
            padding_start = state_idx * state_stride + row_size
            padding_end = (state_idx + 1) * state_stride
            torch.testing.assert_close(
                storage_cpu[padding_start:padding_end],
                torch.full((padding,), -1.0),
                rtol=0,
                atol=0,
            )


@pytest.mark.parametrize("padding", [0, 13])
def test_triton_prefill_state_roundtrip_preserves_shared_pages(padding: int) -> None:
    # Cache stores [H, V, K]; Triton prefill consumes [H, K, V]. Use
    # rectangular, nonuniform states to detect transpose and stride mistakes.
    shape = (2, 3, 5)
    row_size = math.prod(shape)
    stride = row_size + padding
    offset = 7
    storage = torch.full((offset + 5 * stride,), -17.0, device="npu")
    state = torch.as_strided(storage, (5, *shape), (stride, 15, 5, 1), storage_offset=offset)
    initial = torch.arange(5 * row_size, dtype=torch.float32).reshape(5, *shape)
    state.copy_(initial.to("npu"))
    expected_storage = storage.cpu()
    indices = torch.tensor([3, 0, 2], dtype=torch.int32, device="npu")
    has_initial_state = torch.tensor([True, False, True], device="npu")
    expected = initial[[3, 0, 2]].clone()
    expected[1].zero_()

    # Repeated prefill chunks must read the previous write, while neighboring
    # requests and page padding retain their original values.
    for step in range(3):
        gathered = gather_ssm_states(state, indices, has_initial_state)
        triton_initial = gathered.transpose(-1, -2).contiguous()
        torch.testing.assert_close(triton_initial.cpu(), expected.transpose(-1, -2), rtol=0, atol=0)
        final_state = triton_initial + step + 1
        scatter_ssm_states_(state, indices, final_state.transpose(-1, -2).contiguous())
        expected += step + 1
        for selected, block in enumerate((3, 0, 2)):
            start = offset + block * stride
            expected_storage[start : start + row_size] = expected[selected].reshape(-1)
        torch.testing.assert_close(storage.cpu(), expected_storage, rtol=0, atol=0)
        has_initial_state.fill_(True)


def test_rejects_empty_state_cache() -> None:
    state = torch.empty((0, 8), dtype=torch.float32, device="npu")
    indices = torch.empty((0,), dtype=torch.int32, device="npu")
    has_initial_state = torch.empty((0,), dtype=torch.bool, device="npu")

    with pytest.raises(ValueError, match="at least one row"):
        gather_ssm_states(state, indices, has_initial_state)


def test_rejects_overlapping_state_rows() -> None:
    row_size = 8
    storage = torch.zeros((2 * row_size,), dtype=torch.float32, device="npu")
    state = torch.as_strided(
        storage,
        size=(2, row_size),
        stride=(row_size - 1, 1),
    )
    indices = torch.tensor([0], dtype=torch.int32, device="npu")
    has_initial_state = torch.tensor([True], device="npu")

    with pytest.raises(ValueError, match=r"stride\(0\)"):
        gather_ssm_states(state, indices, has_initial_state)
