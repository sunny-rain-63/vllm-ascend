#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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

import math

import torch

from tests.ut.base import TestBase
from vllm_ascend.attention.fia_small_ops import (
    gather_kv_from_paged_cache,
    small_ops_fused_infer_attention,
)


def _naive_attention_per_request(q_i, k_i, v_i, scale, window=None, sink=None):
    """Naive per-head double-loop reference (all fp32) used as the golden
    baseline for the small-ops implementation."""
    q_len, N, D = q_i.shape
    kv_len = k_i.shape[0]
    o = torch.zeros(q_len, N, D, dtype=torch.float32)
    lse = torch.zeros(q_len, N, dtype=torch.float32)
    for h in range(N):
        for j in range(q_len):
            pos = kv_len - q_len + j
            logits = []
            for k in range(kv_len):
                if k > pos:
                    break
                if window is not None and k < pos - window:
                    continue
                logits.append(float((q_i[j, h] * k_i[k, h]).sum()) * scale)
            terms = [math.exp(x) for x in logits]
            if sink is not None:
                terms.append(math.exp(float(sink[h])))
            denom = sum(terms)
            lse[j, h] = math.log(denom)
            kks = [k for k in range(kv_len)
                   if k <= pos and (window is None or k >= pos - window)]
            for t, k in enumerate(kks):
                o[j, h] += terms[t] / denom * v_i[k, h]
    return o, lse


class TestFiaSmallOps(TestBase):
    """Cross-check the small-ops FIA reference against a naive per-token
    implementation on CPU (fp32)."""

    def setUp(self):
        torch.manual_seed(0)
        self.D = 64
        self.N, self.N_kv = 8, 2
        self.block_size = 4
        self.scale = 1.0 / math.sqrt(self.D)

    def _run_paged_case(self, q_lens, kv_lens, window=None, sink=False, atol=2e-5):
        B = len(q_lens)
        T = sum(q_lens)
        block_size = self.block_size
        max_blocks = max((l + block_size - 1) // block_size for l in kv_lens)
        num_blocks = B * max_blocks + 3
        kc = torch.randn(num_blocks, block_size, self.N_kv, self.D, dtype=torch.float32)
        vc = torch.randn(num_blocks, block_size, self.N_kv, self.D, dtype=torch.float32)
        perm = torch.randperm(num_blocks)
        block_table = torch.zeros(B, max_blocks, dtype=torch.int32)
        cursor = 0
        for i in range(B):
            nb = (kv_lens[i] + block_size - 1) // block_size
            block_table[i, :nb] = perm[cursor:cursor + nb].to(torch.int32)
            cursor += nb
        query = torch.randn(T, self.N, self.D, dtype=torch.float32)
        sink_t = torch.randn(self.N, dtype=torch.float32) * 0.1 if sink else None

        out, lse = small_ops_fused_infer_attention(
            query, kc, vc,
            num_heads=self.N, num_key_value_heads=self.N_kv,
            block_table=block_table, block_size=block_size,
            actual_seq_qlen=torch.tensor(q_lens).cumsum(0),
            actual_seq_kvlen=kv_lens,
            scale=self.scale,
            sparse_mode=4 if window else 3,
            pre_tokens=window,
            learnable_sink=sink_t,
            return_lse=True,
        )
        q_bounds = [0] + list(torch.tensor(q_lens).cumsum(0).tolist())
        for i in range(B):
            q0, q1 = q_bounds[i], q_bounds[i + 1]
            k_i = gather_kv_from_paged_cache(kc, block_table[i], kv_lens[i], block_size)
            v_i = gather_kv_from_paged_cache(vc, block_table[i], kv_lens[i], block_size)
            k_i = k_i.repeat_interleave(self.N // self.N_kv, dim=1)
            v_i = v_i.repeat_interleave(self.N // self.N_kv, dim=1)
            o_ref, lse_ref = _naive_attention_per_request(
                query[q0:q1], k_i, v_i, self.scale, window=window, sink=sink_t)
            self.assertTrue(
                torch.allclose(out[q0:q1], o_ref, atol=atol),
                msg=f"request {i} output mismatch: {(out[q0:q1] - o_ref).abs().max()}")
            self.assertTrue(
                torch.allclose(lse[q0:q1], lse_ref, atol=atol),
                msg=f"request {i} lse mismatch: {(lse[q0:q1] - lse_ref).abs().max()}")

    def test_decode_only_paged_gqa(self):
        self._run_paged_case([1, 1, 1], [7, 13, 4])

    def test_mtp_decode_multi_token(self):
        self._run_paged_case([3, 2], [11, 9])

    def test_chunked_prefill_mixed(self):
        self._run_paged_case([5, 1, 8], [12, 6, 8])

    def test_sliding_window(self):
        self._run_paged_case([6, 1], [20, 5], window=5)

    def test_learnable_sink(self):
        self._run_paged_case([2, 1], [9, 3], sink=True)

    def test_prefill_no_cache(self):
        T, q_lens = 11, [4, 7]
        query = torch.randn(T, self.N, self.D)
        key = torch.randn(T, self.N_kv, self.D)
        value = torch.randn(T, self.N_kv, self.D)
        out, _ = small_ops_fused_infer_attention(
            query, key, value,
            num_heads=self.N, num_key_value_heads=self.N_kv,
            block_table=None,
            actual_seq_qlen=torch.tensor(q_lens).cumsum(0),
            scale=self.scale,
            sparse_mode=3,
        )
        q_bounds = [0, 4, 11]
        for i in range(2):
            q0, q1 = q_bounds[i], q_bounds[i + 1]
            o_ref, _ = _naive_attention_per_request(
                query[q0:q1],
                key[q0:q1].repeat_interleave(self.N // self.N_kv, dim=1),
                value[q0:q1].repeat_interleave(self.N // self.N_kv, dim=1),
                self.scale)
            self.assertTrue(
                torch.allclose(out[q0:q1], o_ref, atol=2e-5),
                msg=f"PrefillNoCache request {i} mismatch: {(out[q0:q1] - o_ref).abs().max()}")
