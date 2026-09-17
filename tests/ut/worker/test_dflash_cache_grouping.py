# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""CPU metadata contracts for grouping; no vLLM/NPU runtime is simulated."""

import ast
import unittest
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace


@dataclass(frozen=True)
class _FullAttentionSpec:
    block_size: int = 1536
    page_size_bytes: int = 3248128
    max_blocks: int = 80
    num_kv_heads: int = 2
    head_size: int = 256

    def max_memory_usage_bytes(self, _config):
        return self.max_blocks * self.page_size_bytes


@dataclass(frozen=True)
class _SlidingWindowSpec(_FullAttentionSpec):
    max_blocks: int = 24
    sliding_window: int = 2048


@dataclass(frozen=True)
class _MambaSpec:
    block_size: int = 1536
    page_size_bytes: int = 3248128
    max_blocks: int = 9
    num_speculative_blocks: int = 7
    num_prefill_checkpoint_blocks: int = 0
    prefill_checkpoint_alignment: int | None = None

    def max_memory_usage_bytes(self, _config):
        return self.max_blocks * self.page_size_bytes


@dataclass
class _KVCacheGroupSpec:
    layer_names: list[str]
    kv_cache_spec: object
    is_eagle_group: bool = False
    enable_kv_transfer: bool = True


def _load_production():
    source = Path(__file__).resolve().parents[3] / "vllm_ascend/core/dflash_cache_grouping.py"
    namespace = dict(
        VllmConfig=SimpleNamespace,
        FullAttentionSpec=_FullAttentionSpec,
        SlidingWindowSpec=_SlidingWindowSpec,
        MambaSpec=_MambaSpec,
        KVCacheSpec=object,
        KVCacheGroupSpec=_KVCacheGroupSpec,
    )
    module = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    module.body = [
        node
        for node in module.body
        if not (isinstance(node, ast.ImportFrom) and (node.module or "").startswith("vllm"))
    ]
    exec(compile(module, str(source), "exec"), namespace)
    return namespace["choose_dflash_cache_groups"]


class TestDFlashCacheGrouping(unittest.TestCase):
    def setUp(self):
        self.choose = _load_production()
        self.config = SimpleNamespace(
            use_v2_model_runner=True,
            speculative_config=SimpleNamespace(method="dflash"),
            parallel_config=SimpleNamespace(pipeline_parallel_size=1),
            cache_config=SimpleNamespace(num_gpu_blocks_override=None),
        )
        self.specs = {
            **{f"target.full.{i}": _FullAttentionSpec() for i in range(16)},
            "draft.full": _FullAttentionSpec(num_kv_heads=4, head_size=128),
            **{f"draft.swa.{i}": _SlidingWindowSpec(num_kv_heads=4, head_size=128) for i in range(4)},
            **{f"target.mamba.{i}": _MambaSpec() for i in range(48)},
        }
        self.groups = [_KVCacheGroupSpec([name], spec) for name, spec in self.specs.items()]
        self.draft_names = {name for name in self.specs if name.startswith("draft.")}
        self.budget = 26556098560
        self.fia_limit = 5461
        self.page = _FullAttentionSpec().page_size_bytes

    def _choose(self, **kwargs):
        return self.choose(
            self.config,
            self.specs,
            self.groups,
            draft_layer_names=kwargs.pop("draft_layer_names", self.draft_names),
            available_memory=kwargs.pop("available_memory", self.budget),
            max_num_blocks=kwargs.pop("max_num_blocks", self.fia_limit),
            **kwargs,
        )

    def test_reported_budget_selects_width_two_without_raising_address_limit(self):
        groups = self._choose()
        self.assertIsNotNone(groups)
        self.assertEqual(len(groups), 35)
        width = max(len(group.layer_names) for group in groups)
        self.assertEqual(width, 2)
        count = min(self.budget // (width * self.page), self.fia_limit)
        self.assertEqual(count, 4087)
        self.assertEqual(width * self.page * count, 26550198272)
        self.assertLessEqual(count, self.fia_limit)
        self.assertLessEqual(width * self.page * count, self.budget)
        old_cost = sum(group.kv_cache_spec.max_blocks for group in self.groups)
        new_cost = sum(group.kv_cache_spec.max_blocks for group in groups)
        self.assertEqual((old_cost, new_cost), (1888, 984))
        self.assertGreater((count - 1) * old_cost, (self.fia_limit - 1) * new_cost)

    def test_every_layer_exact_spec_and_draft_role_are_preserved(self):
        groups = self._choose()
        names = [name for group in groups for name in group.layer_names]
        self.assertCountEqual(names, self.specs)
        for group in groups:
            roles = {name in self.draft_names for name in group.layer_names}
            self.assertEqual(roles, {group.is_eagle_group})
            for name in group.layer_names:
                self.assertEqual(group.kv_cache_spec, self.specs[name])
            if isinstance(group.kv_cache_spec, _SlidingWindowSpec):
                self.assertEqual(group.kv_cache_spec.sliding_window, 2048)
            if isinstance(group.kv_cache_spec, _MambaSpec):
                self.assertFalse(group.is_eagle_group)
                self.assertEqual(group.kv_cache_spec.num_speculative_blocks, 7)
        self.assertEqual(len([group for group in groups if group.is_eagle_group]), 3)
        self.assertTrue(all(len(group.layer_names) == 1 and not group.is_eagle_group for group in self.groups))
        self.assertIsNone(self.config.cache_config.num_gpu_blocks_override)

    def test_matching_target_and_draft_specs_still_have_separate_ownership(self):
        self.specs["draft.full"] = self.specs["target.full.0"]
        self.groups[16].kv_cache_spec = self.specs["draft.full"]
        groups = self._choose()
        draft_full_group = next(group for group in groups if "draft.full" in group.layer_names)
        self.assertEqual(draft_full_group.layer_names, ["draft.full"])
        self.assertTrue(draft_full_group.is_eagle_group)

    def test_transfer_and_checkpoint_metadata_are_preserved(self):
        for index in range(8):
            self.groups[index].enable_kv_transfer = False
        name = "target.mamba.0"
        self.specs[name] = replace(self.specs[name], num_prefill_checkpoint_blocks=1, prefill_checkpoint_alignment=128)
        self.groups[21].kv_cache_spec = self.specs[name]
        groups = self._choose()
        for group in groups:
            expected_transfer = {
                not (name.startswith("target.full.") and int(name.rsplit(".", 1)[1]) < 8) for name in group.layer_names
            }
            self.assertEqual(expected_transfer, {group.enable_kv_transfer})
        checkpoint_group = next(group for group in groups if name in group.layer_names)
        self.assertIs(checkpoint_group.kv_cache_spec, self.specs[name])

    def test_no_regroup_when_memory_instead_of_address_limit_is_binding(self):
        self.assertIsNone(self._choose(available_memory=5000 * self.page))

    def test_explicit_override_remains_a_block_limit_not_extra_memory(self):
        self.config.cache_config.num_gpu_blocks_override = 4096
        groups = self._choose()
        self.assertEqual(max(len(group.layer_names) for group in groups), 2)
        self.assertEqual(self.config.cache_config.num_gpu_blocks_override, 4096)

    def test_no_regroup_without_strict_admission_capacity_gain(self):
        # A one-layer draft Full group dominates; padding it cannot pay for
        # recovering the small amount of memory beyond this address limit.
        self.specs["draft.full"] = replace(self.specs["draft.full"], max_blocks=1000000)
        self.groups[16].kv_cache_spec = self.specs["draft.full"]
        self.assertIsNone(self._choose(available_memory=5500 * self.page))

    def test_no_regroup_for_unmixed_or_non_dflash_paths(self):
        self.config.speculative_config.method = "eagle"
        self.assertIsNone(self._choose())
        self.config.speculative_config.method = "dflash"
        self.assertIsNone(self._choose(draft_layer_names={"draft.full"}))
        self.config.use_v2_model_runner = False
        self.assertIsNone(self._choose())

    def test_pipeline_parallel_is_not_regrouped(self):
        self.config.parallel_config.pipeline_parallel_size = 2
        self.assertIsNone(self._choose())

    def test_missing_unknown_or_mamba_draft_roles_fail_closed(self):
        for names in (set(), {"not.a.layer"}, self.draft_names | {"target.mamba.0"}):
            with self.subTest(names=names), self.assertRaises(ValueError):
                self._choose(draft_layer_names=names)

    def test_nonuniform_pages_or_merged_specs_are_not_reinterpreted(self):
        self.specs["target.full.0"] = replace(self.specs["target.full.0"], page_size_bytes=self.page + 128)
        self.assertIsNone(self._choose())
        self.specs["target.full.0"] = _FullAttentionSpec()
        self.groups[0].kv_cache_spec = replace(self.groups[0].kv_cache_spec, head_size=128)
        self.assertIsNone(self._choose())

    def test_duplicate_or_missing_layer_ownership_fails_closed(self):
        self.groups[-1].layer_names = self.groups[0].layer_names
        with self.assertRaisesRegex(ValueError, "exactly once"):
            self._choose()

    def test_group_selection_is_deterministic(self):
        self.assertEqual(self._choose(), self._choose())

    def test_legacy_group_without_draft_field_keeps_existing_grouping(self):
        @dataclass
        class LegacyGroup:
            layer_names: list[str]
            kv_cache_spec: object

        self.groups = [LegacyGroup(group.layer_names, group.kv_cache_spec) for group in self.groups]
        self.assertIsNone(self._choose())


if __name__ == "__main__":
    unittest.main()
