# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Tests for flatten_runner_kv_caches.

Upstream vLLM's copy-on-write path (update_requests ->
copy_kv_cache_blocks_inplace) iterates the runner's kv_caches and calls
``cache.device`` on every entry, so each entry must be a single tensor even
though Ascend binds per-layer caches as (k, v) / indexer / Mamba tuples.
"""

import torch

from vllm_ascend.worker.v2.attn_utils import flatten_runner_kv_caches


class TestFlattenRunnerKvCaches:
    def test_flattens_tuple_entries(self):
        k_cache = torch.zeros(4, 2)
        v_cache = torch.zeros(4, 2)
        mamba_state = torch.zeros(3, 2)
        result = flatten_runner_kv_caches([(k_cache, v_cache), (mamba_state,)])
        assert result == [k_cache, v_cache, mamba_state]

    def test_flattens_list_entries(self):
        k_cache = torch.zeros(4, 2)
        v_cache = torch.zeros(4, 2)
        result = flatten_runner_kv_caches([[k_cache, v_cache]])
        assert result == [k_cache, v_cache]

    def test_plain_tensors_are_kept_in_order(self):
        caches = [torch.zeros(4, 2), torch.zeros(8, 2)]
        result = flatten_runner_kv_caches(caches)
        assert result == caches
        for original, flattened in zip(caches, result):
            assert flattened is original

    def test_mixed_entries(self):
        tensor = torch.zeros(4, 2)
        k_cache = torch.zeros(4, 2)
        v_cache = torch.zeros(4, 2)
        result = flatten_runner_kv_caches([tensor, (k_cache, v_cache)])
        assert result == [tensor, k_cache, v_cache]

    def test_empty(self):
        assert flatten_runner_kv_caches([]) == []
