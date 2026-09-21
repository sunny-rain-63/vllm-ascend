#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
"""Small-ops reference implementation of FIA (npu_fused_infer_attention_score).

Decomposes FIA into gather / matmul / mask / softmax / matmul primitives.
Enabled via VLLM_ASCEND_FIA_SMALL_OPS as a correctness-oriented debug fallback
for the FIA high-address overflow (element offsets >= 2**32 read wrong blocks).
Semantics are calibrated against the real kernel: TND layout, paged KV cache,
right-aligned causal mask, sparse_mode 0/3/4, GQA and learnable sinks.

This path materializes fp32 score matrices and loops per request; it is NOT
capture-friendly and NOT suitable for production serving.
"""

from __future__ import annotations

import math
from typing import Optional

import torch


def _to_list(x) -> list[int]:
    if x is None:
        return []
    if isinstance(x, torch.Tensor):
        return [int(v) for v in x.flatten().tolist()]
    return [int(v) for v in x]


def gather_kv_from_paged_cache(
    cache: torch.Tensor,            # (num_blocks, block_size, N_kv, D)
    block_table_row: torch.Tensor,  # (max_num_blocks,) int
    kv_len: int,
    block_size: int,
) -> torch.Tensor:
    """Gather one request's contiguous KV from the paged cache: (kv_len, N_kv, D)."""
    num_blocks = (kv_len + block_size - 1) // block_size
    block_ids = block_table_row[:num_blocks].to(torch.long)
    kv = cache[block_ids]  # (num_blocks, block_size, N_kv, D)
    kv = kv.reshape(num_blocks * block_size, *kv.shape[2:])
    return kv[:kv_len]


@torch.no_grad()
def small_ops_fused_infer_attention(
    query: torch.Tensor,            # (T, N, D)
    key: torch.Tensor,              # paged: (num_blocks, block_size, N_kv, D) / no-cache: (T, N_kv, D)
    value: torch.Tensor,            # same as key
    *,
    num_heads: int,
    num_key_value_heads: int,
    block_table: Optional[torch.Tensor] = None,  # (B, max_num_blocks) or None (PrefillNoCache)
    block_size: Optional[int] = None,
    actual_seq_qlen=None,           # per-request q length cumsum, length B
    actual_seq_kvlen=None,          # per-request kv length (not cumsum); None means kv_len == q_len
    scale: Optional[float] = None,
    sparse_mode: int = 3,
    pre_tokens: Optional[int] = None,   # sliding window size for sparse_mode=4
    next_tokens: int = 0,
    atten_mask: Optional[torch.Tensor] = None,  # sparse_mode=0 only, bool: True = masked out
    learnable_sink: Optional[torch.Tensor] = None,  # (N,)
    return_lse: bool = False,
):
    """Small-ops FIA reference. Returns (attn_output, softmax_lse or None).

    attn_output: (T, N, D), dtype matches query
    softmax_lse: (T, N) fp32, natural log; None when return_lse=False
    """
    assert query.dim() == 3, f"query should be (T, N, D), got {tuple(query.shape)}"
    T, N, D = query.shape
    assert N == num_heads, f"query heads {N} != num_heads {num_heads}"
    assert num_heads % num_key_value_heads == 0, "GQA requires num_heads % num_kv_heads == 0"
    group_size = num_heads // num_key_value_heads
    if scale is None:
        scale = 1.0 / math.sqrt(D)

    q_cumsum = _to_list(actual_seq_qlen)
    if not q_cumsum:
        q_cumsum = [T]  # single request
    batch = len(q_cumsum)
    q_bounds = [0] + q_cumsum  # request i covers query [q_bounds[i], q_bounds[i+1])

    kv_lens = _to_list(actual_seq_kvlen)
    if not kv_lens:
        # PrefillNoCache: kv_len == q_len
        kv_lens = [q_bounds[i + 1] - q_bounds[i] for i in range(batch)]
    assert len(kv_lens) == batch

    if block_table is not None:
        assert block_size is not None, "block_size is required with a paged cache"
        assert key.dim() == 4, (
            f"paged cache should be (num_blocks, block_size, N_kv, D), got {tuple(key.shape)}")
    else:
        # no-cache KV is packed per request like query, but with its own
        # boundaries (they differ from q bounds for encoder-decoder)
        kv_bounds = [0]
        for length in kv_lens:
            kv_bounds.append(kv_bounds[-1] + length)

    dtype = query.dtype
    device = query.device
    out = torch.empty(T, N, D, dtype=dtype, device=device)
    lse = torch.empty(T, N, dtype=torch.float32, device=device) if return_lse else None

    sink = None
    if learnable_sink is not None:
        sink = learnable_sink.to(torch.float32).view(N)  # (N,)

    for i in range(batch):
        q0, q1 = q_bounds[i], q_bounds[i + 1]
        q_len = q1 - q0
        kv_len = kv_lens[i]
        if q_len == 0:
            continue
        assert kv_len >= q_len or block_table is None, (
            f"request {i}: kv_len({kv_len}) < q_len({q_len}), right-aligned causal does not hold")

        q_i = query[q0:q1].to(torch.float32)  # (q_len, N, D)

        if block_table is not None:
            k_i = gather_kv_from_paged_cache(key, block_table[i], kv_len, block_size)
            v_i = gather_kv_from_paged_cache(value, block_table[i], kv_len, block_size)
        else:
            k_i = key[kv_bounds[i]:kv_bounds[i + 1]]
            v_i = value[kv_bounds[i]:kv_bounds[i + 1]]
        k_i = k_i.to(torch.float32)  # (kv_len, N_kv, D)
        v_i = v_i.to(torch.float32)

        k_i = k_i.repeat_interleave(group_size, dim=1)  # (kv_len, N, D)
        v_i = v_i.repeat_interleave(group_size, dim=1)

        scores = torch.einsum("qnd,knd->nqk", q_i, k_i) * scale  # (N, q_len, kv_len) fp32

        # query token j's global position (right-aligned causal, FIA sparse_mode=3/4 convention)
        pos = torch.arange(kv_len - q_len, kv_len, device=device).view(1, q_len, 1)  # (1, q_len, 1)
        kidx = torch.arange(kv_len, device=device).view(1, 1, kv_len)                # (1, 1, kv_len)
        if sparse_mode == 0:
            keep = torch.ones(1, q_len, kv_len, dtype=torch.bool, device=device)
            if atten_mask is not None:
                m = atten_mask
                if m.dim() == 2:  # (q_len, kv_len), True = masked out
                    m = m.to(torch.bool)
                    keep = keep & ~m.view(1, q_len, kv_len)
        elif sparse_mode == 3:  # causal
            keep = kidx <= pos + next_tokens
        elif sparse_mode == 4:  # sliding-window band: pos - pre_tokens <= k <= pos
            # Matches the real FIA sparse_mode=4 band semantics: pre_tokens
            # preceding tokens + the current token are kept (pre_tokens + 1 in
            # total). To align with vLLM's sliding_window (W tokens including
            # the current one), callers pass pre_tokens = sliding_window - 1.
            assert pre_tokens is not None, "sparse_mode=4 requires pre_tokens"
            keep = (kidx <= pos + next_tokens) & (kidx >= pos - pre_tokens)
        else:
            raise NotImplementedError(f"unsupported sparse_mode={sparse_mode}")
        scores = scores.masked_fill(~keep, float("-inf"))

        m = scores.amax(dim=-1, keepdim=True)            # (N, q_len, 1)
        p = torch.exp(scores - m)                        # (N, q_len, kv_len)
        denom = p.sum(dim=-1, keepdim=True)              # (N, q_len, 1)
        if sink is not None:
            # attention sink: denominator gets an extra exp(sink_h - m); the
            # sink has no value vector of its own
            denom = denom + torch.exp(sink.view(N, 1, 1) - m)
        p = p / denom                                    # (N, q_len, kv_len)

        o_i = torch.einsum("nqk,knd->qnd", p, v_i)       # (q_len, N, D) fp32
        out[q0:q1] = o_i.to(dtype)

        # softmax_lse = log(sum(exp(s - m))) + m, sink term included (FIA-compatible)
        if return_lse:
            lse[q0:q1] = (torch.log(denom) + m).squeeze(-1).transpose(0, 1).contiguous()  # (q_len, N)

    return out, lse
