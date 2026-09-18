# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""CPU planner contracts; these fixtures do not execute FIA or allocate KV."""

import ast
import copy
import unittest
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock


@dataclass
class _FullSpec:
    block_size: int = 1536
    num_kv_heads: int = 4
    head_size: int = 128
    head_size_v: int = 128


class _SlidingSpec(_FullSpec):
    pass


class _MambaSpec:
    pass


class TestDFlashAddressLimit(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).resolve().parents[3] / "vllm_ascend/patch/platform/patch_kv_cache_utils.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        tree.body = [
            node
            for node in tree.body
            if (isinstance(node, ast.FunctionDef) and node.name == "_limit_mixed_dflash_cache_blocks")
            or (
                isinstance(node, ast.Assign)
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id.startswith("_DFLASH_")
            )
        ]
        namespace = dict(
            copy=copy,
            wraps=wraps,
            logger=Mock(),
            FullAttentionSpec=_FullSpec,
            SlidingWindowSpec=_SlidingSpec,
            MambaSpec=_MambaSpec,
        )
        exec(compile(tree, str(path), "exec"), namespace)
        self.wrap = namespace["_limit_mixed_dflash_cache_blocks"]
        self.config = SimpleNamespace(
            use_v2_model_runner=True,
            speculative_config=SimpleNamespace(
                method="dflash",
                draft_model_config=SimpleNamespace(
                    hf_config=SimpleNamespace(layer_types=["sliding_attention"] * 4 + ["full_attention"])
                ),
            ),
            cache_config=SimpleNamespace(num_gpu_blocks_override=None),
        )
        self.specs = {"full": _FullSpec(), "swa": _SlidingSpec(), "mamba": _MambaSpec()}
        self.page = 3248128
        self.calls = []
        self.legacy = False

    def _original(self, config, specs, budgets):
        self.calls.append(config)
        override = config.cache_config.num_gpu_blocks_override
        pool_width = 2 if self.legacy else 1
        count = min(budget // (self.page * pool_width) for budget in budgets) if override is None else override
        return [
            SimpleNamespace(
                num_blocks=count,
                kv_cache_groups=["unchanged groups"],
                kv_cache_tensors=(
                    [SimpleNamespace(size=count * self.page, shared_by=[name]) for name in ("a", "b")]
                    if self.legacy
                    else [SimpleNamespace(size=count * self.page, layers=[name]) for name in ("a", "b")]
                ),
            )
            for _ in specs
        ]

    def _run(self, budgets=None):
        budgets = budgets or [9000 * self.page]
        return self.wrap(self._original)(self.config, [self.specs] * len(budgets), budgets)

    def test_automatic_cap_replans_descriptors_without_mutating_user_config(self):
        result = self._run()[0]
        self.assertEqual(result.num_blocks, 4096)
        self.assertEqual(result.kv_cache_tensors[0].size, 4096 * self.page)
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(self.calls[-1].cache_config.num_gpu_blocks_override, 4096)
        self.assertIsNone(self.config.cache_config.num_gpu_blocks_override)
        self.assertEqual(result.kv_cache_groups, ["unchanged groups"])

    def test_small_budget_is_not_increased_to_4096(self):
        self.assertEqual(self._run([2000 * self.page])[0].num_blocks, 2000)
        self.assertEqual(len(self.calls), 1)

    def test_small_explicit_override_is_preserved(self):
        self.config.cache_config.num_gpu_blocks_override = 1024
        self.assertEqual(self._run()[0].num_blocks, 1024)
        self.assertEqual(len(self.calls), 1)

    def test_large_override_is_bounded_by_real_budget(self):
        self.config.cache_config.num_gpu_blocks_override = 10000
        self.assertEqual(self._run([2000 * self.page])[0].num_blocks, 2000)
        self.assertEqual(self.config.cache_config.num_gpu_blocks_override, 10000)

    def test_rank_budgets_share_the_smaller_limit(self):
        plans = self._run([9000 * self.page, 3000 * self.page])
        self.assertEqual([plan.num_blocks for plan in plans], [3000, 3000])

    def test_larger_storage_block_respects_kernel_block_address_boundary(self):
        self.specs["swa"].block_size = 3072
        self.assertEqual(self._run()[0].num_blocks, 2730)

    def test_wider_heads_respect_plane_element_address_boundary(self):
        self.specs["full"].head_size_v = 1024
        self.assertEqual(self._run()[0].num_blocks, 682)

    def test_release_pools_are_summed_not_overlaid(self):
        self.legacy = True
        self.config.cache_config.num_gpu_blocks_override = 9000
        self.assertEqual(self._run([6000 * self.page])[0].num_blocks, 3000)

    def test_unmixed_target_only_and_v1_are_unchanged(self):
        for mode in ("full", "sliding", "target", "v1", "other_method", "no_mamba"):
            with self.subTest(mode=mode):
                config = copy.deepcopy(self.config)
                specs = dict(self.specs)
                if mode in ("full", "sliding"):
                    config.speculative_config.draft_model_config.hf_config.layer_types = [f"{mode}_attention"] * 5
                elif mode == "target":
                    config.speculative_config = None
                elif mode == "v1":
                    config.use_v2_model_runner = False
                elif mode == "other_method":
                    config.speculative_config.method = "eagle"
                else:
                    del specs["mamba"]
                result = self.wrap(self._original)(config, [specs], [9000 * self.page])
                self.assertEqual(result[0].num_blocks, 9000)

    def test_upstream_ignoring_limit_is_rejected(self):
        original = Mock(side_effect=lambda *_: self._original(self.config, [self.specs], [9000 * self.page]))
        with self.assertRaisesRegex(ValueError, "did not preserve"):
            self.wrap(original)(self.config, [self.specs], [9000 * self.page])


if __name__ == "__main__":
    unittest.main()
