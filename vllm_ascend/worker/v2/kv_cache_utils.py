# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

import math

import torch


def reshape_paged_attention_kv_cache(
    raw_cache: torch.Tensor,
    kv_cache_shape: tuple[int, ...],
    dtype: torch.dtype,
    page_stride_bytes: int,
    num_blocks_per_kv_block: int = 1,
    head_size_v: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """View combined K/V pages without moving padding outside its block.

    Each scheduler page contains equally spaced kernel blocks, each laid out
    as [K, V, padding]. Only the scheduler block owning a page may write it,
    even when target, draft and Mamba groups share the backing allocation.
    """
    if len(kv_cache_shape) != 5 or kv_cache_shape[0] != 2:
        raise ValueError("Combined Attention cache must have shape [K/V, blocks, block_size, heads, dim].")
    typed_cache = raw_cache.view(dtype)
    dtype_size = typed_cache.element_size()
    if page_stride_bytes % dtype_size:
        raise ValueError("Physical Attention page is not aligned to its dtype.")
    if num_blocks_per_kv_block < 1:
        raise ValueError("The number of kernel blocks per KV block must be positive.")
    if page_stride_bytes % (num_blocks_per_kv_block * dtype_size):
        raise ValueError("Padded combined Attention pages must split into dtype-aligned kernel blocks.")

    k_shape = tuple(kv_cache_shape[1:])
    v_shape = (*k_shape[:-1], head_size_v if head_size_v is not None else k_shape[-1])
    k_elements = math.prod(k_shape[1:])
    v_elements = math.prod(v_shape[1:])
    block_stride = page_stride_bytes // num_blocks_per_kv_block // dtype_size
    if block_stride < k_elements + v_elements:
        raise ValueError("Physical Attention page is too small for its kernel blocks.")
    required_elements = (k_shape[0] - 1) * block_stride + k_elements + v_elements if k_shape[0] else 0
    if required_elements > typed_cache.numel():
        raise ValueError("Combined Attention cache view exceeds the backing allocation.")

    def make_view(shape: tuple[int, ...], offset: int) -> torch.Tensor:
        strides = (block_stride, *(math.prod(shape[dim + 1 :]) for dim in range(1, len(shape))))
        return torch.as_strided(
            typed_cache,
            size=shape,
            stride=strides,
            storage_offset=typed_cache.storage_offset() + offset,
        )

    return make_view(k_shape, 0), make_view(v_shape, k_elements)
