# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Production grouping/wrapper contracts with CPU-only upstream fixtures.

These tests execute the actual Ascend helpers, but not the vLLM allocator or
NPU kernels. The fixture planner emulates upstream's metadata-stripping merge,
rank normalization and shared-backing descriptors without allocating memory.
"""

import ast
import copy
import math
import pickle
import runpy
import sys
import unittest
from dataclasses import dataclass, replace
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch


def _dtype_size(dtype):
    return {"torch.bfloat16": 2, "torch.float32": 4}[dtype]


@dataclass(frozen=True)
class _FullAttentionSpec:
    block_size: int = 1536
    num_kv_heads: int = 2
    head_size: int = 256
    head_size_v: int = 256
    dtype: str = "torch.bfloat16"
    page_size_padded: int = 3248128
    max_blocks: int = 80

    @property
    def page_size_bytes(self):
        return self.page_size_padded

    def max_memory_usage_bytes(self, _config):
        return self.max_blocks * self.page_size_bytes


@dataclass(frozen=True)
class _SlidingWindowSpec(_FullAttentionSpec):
    sliding_window: int = 2048
    max_blocks: int = 24


@dataclass(frozen=True)
class _MambaSpec:
    block_size: int = 1536
    shapes: tuple = ((5120, 10), (24, 128, 128))
    dtypes: tuple = ("torch.bfloat16", "torch.float32")
    page_size_padded: int = 3248128
    max_blocks: int = 9
    num_speculative_blocks: int = 7

    @property
    def page_size_bytes(self):
        return self.page_size_padded or sum(
            math.prod(shape) * _dtype_size(dtype) for shape, dtype in zip(self.shapes, self.dtypes)
        )

    def max_memory_usage_bytes(self, _config):
        return self.max_blocks * self.page_size_bytes


@dataclass
class _KVCacheGroupSpec:
    layer_names: list[str]
    kv_cache_spec: object
    is_eagle_group: bool = False
    enable_kv_transfer: bool = True


@dataclass
class _UniformTypeKVCacheSpecs:
    kv_cache_specs: dict


class _GPUModelRunner:
    def get_kv_cache_spec(self):
        return self.specs


class TestDFlashCachePlanning(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[3]
        self.impl = ModuleType("_dflash_cache_planning_test_runtime")
        modules_patch = patch.dict(sys.modules, {self.impl.__name__: self.impl})
        modules_patch.start()
        self.addCleanup(modules_patch.stop)
        self.impl.__dict__.update(
            VllmConfig=SimpleNamespace,
            KVCacheSpec=object,
            FullAttentionSpec=_FullAttentionSpec,
            SlidingWindowSpec=_SlidingWindowSpec,
            MambaSpec=_MambaSpec,
            KVCacheGroupSpec=_KVCacheGroupSpec,
            UniformTypeKVCacheSpecs=_UniformTypeKVCacheSpecs,
            get_dtype_size=_dtype_size,
            logger=Mock(),
        )
        layout = runpy.run_path(str(root / "vllm_ascend/core/dflash_cache_layout.py"))
        for name in ("get_dflash_aligned_block_size", "get_dflash_fia_safe_num_blocks"):
            self.impl.__dict__[name] = layout[name]
        for filename in ("dflash_cache_grouping.py", "dflash_cache.py"):
            source = root / "vllm_ascend/core" / filename
            tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
            tree.body = [
                node
                for node in tree.body
                if not (isinstance(node, ast.ImportFrom) and (node.module or "").startswith("vllm"))
            ]
            exec(compile(tree, str(source), "exec"), self.impl.__dict__)

        # Compile the actual override in a class so zero-argument super()
        # retains its normal __class__ cell, without importing the NPU runner.
        source = root / "vllm_ascend/worker/v2/model_runner.py"
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        runner_class = next(
            node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "NPUModelRunner"
        )
        runner_class.body = [
            node for node in runner_class.body if isinstance(node, ast.FunctionDef) and node.name == "get_kv_cache_spec"
        ]
        self.impl.GPUModelRunner = _GPUModelRunner
        exec(compile(ast.Module(body=[runner_class], type_ignores=[]), str(source), "exec"), self.impl.__dict__)

        self.config = SimpleNamespace(
            use_v2_model_runner=True,
            speculative_config=SimpleNamespace(
                method="dflash",
                draft_model_config=SimpleNamespace(
                    hf_config=SimpleNamespace(layer_types=["sliding_attention"] * 4 + ["full_attention"])
                ),
            ),
            parallel_config=SimpleNamespace(pipeline_parallel_size=1),
            cache_config=SimpleNamespace(num_gpu_blocks_override=None, prefix_cache_retention_interval=1536),
        )
        # Intentionally opaque names: ownership must come from the RPC payload.
        self.specs = {
            **{f"cache.{i}": _FullAttentionSpec() for i in range(16)},
            "cache.16": _FullAttentionSpec(num_kv_heads=4, head_size=128, head_size_v=128),
            **{f"cache.{i}": _SlidingWindowSpec(num_kv_heads=4, head_size=128, head_size_v=128) for i in range(17, 21)},
            **{f"cache.{i}": _MambaSpec() for i in range(21, 69)},
        }
        self.draft_names = {f"cache.{i}" for i in range(16, 21)}
        self.page = _FullAttentionSpec().page_size_bytes
        self.budget = 26556098560
        self.calls = []
        self.build = self.impl.wrap_dflash_cache_group_builder(
            lambda _, specs: [_KVCacheGroupSpec([name], spec) for name, spec in specs.items()]
        )
        self.plan = self.impl.wrap_dflash_cache_planner(self._original_planner)

    def _worker_specs(self, *, include_roles=True):
        if include_roles:
            return self.impl.DFlashKVCacheSpecs(self.specs, self.draft_names)
        return dict(self.specs)

    def _runner(self, speculator):
        runner = self.impl.NPUModelRunner()
        runner.vllm_config = self.config
        runner.speculator = speculator
        runner.specs = self.specs
        return runner

    def _original_planner(self, config, worker_specs, budgets):
        self.calls.append(config)
        # The real upstream merge discards the dict subclass and its attributes.
        merged_specs = dict(worker_specs[0])
        groups = self.build(config, merged_specs)
        width = max(len(group.layer_names) for group in groups)
        override = config.cache_config.num_gpu_blocks_override
        count = min(override if override is not None else budget // (width * self.page) for budget in budgets)
        return [
            SimpleNamespace(
                num_blocks=count,
                kv_cache_groups=[replace(group, layer_names=list(group.layer_names)) for group in groups],
                kv_cache_tensors=[
                    SimpleNamespace(
                        size=count * width * self.page,
                        layers=list(group.layer_names),
                        offset=0,
                        layer_stride=count * self.page,
                        block_stride=self.page,
                    )
                    for group in groups
                ],
                prefix_cache_retention_interval=config.cache_config.prefix_cache_retention_interval,
            )
            for _ in budgets
        ]

    def test_rpc_roles_survive_pickle_and_plain_dict_merge(self):
        specs = self._worker_specs()
        restored = pickle.loads(pickle.dumps(specs))
        self.assertIs(type(restored), self.impl.DFlashKVCacheSpecs)
        self.assertEqual(restored, self.specs)
        self.assertEqual(restored.draft_layer_names, frozenset(self.draft_names))
        (plan,) = self.plan(self.config, [restored], [self.budget])
        self.assertEqual(plan.num_blocks, 4087)
        self.assertEqual(len(plan.kv_cache_groups), 35)

    def test_runner_override_transports_loaded_names_and_ordinary_spec_objects(self):
        runner = self._runner(SimpleNamespace(draft_attn_layer_names=self.draft_names))
        specs = runner.get_kv_cache_spec()
        self.assertIs(type(specs), self.impl.DFlashKVCacheSpecs)
        self.assertEqual(specs.draft_layer_names, frozenset(self.draft_names))
        self.assertEqual(specs, self.specs)
        self.assertTrue(all(specs[name] is spec for name, spec in self.specs.items()))

    def test_runner_override_leaves_nonmixed_specs_unchanged(self):
        runner = self._runner(None)
        draft_config = self.config.speculative_config.draft_model_config.hf_config
        for layer_types in (["full_attention"] * 5, ["sliding_attention"] * 5):
            with self.subTest(layer_types=layer_types):
                draft_config.layer_types = layer_types
                self.assertIs(runner.get_kv_cache_spec(), self.specs)
        self.config.speculative_config = None
        self.assertIs(runner.get_kv_cache_spec(), self.specs)

    def test_runner_override_rejects_missing_loaded_draft_names(self):
        for speculator in (None, SimpleNamespace(), SimpleNamespace(draft_attn_layer_names=set())):
            with self.subTest(speculator=speculator), self.assertRaisesRegex(ValueError, "loaded draft attention"):
                self._runner(speculator).get_kv_cache_spec()

    def test_reported_plan_rebuilds_safe_descriptors_and_preserves_config(self):
        original_config = copy.deepcopy(self.config)
        (plan,) = self.plan(self.config, [self._worker_specs()], [self.budget])
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(self.config, original_config)
        self.assertNotIn("_ascend_dflash_cache_groups", vars(self.config))
        self.assertNotIn("_ascend_dflash_draft_layer_names", vars(self.config))
        self.assertEqual(plan.num_blocks, 4087)
        self.assertEqual(len(plan.kv_cache_groups), 35)
        self.assertEqual(plan.prefix_cache_retention_interval, 1536)
        self.assertEqual(self.impl._layer_specs(plan), self.specs)
        for tensor in plan.kv_cache_tensors:
            self.assertEqual(tensor.size, 26550198272)
            self.assertLessEqual(tensor.size, self.budget)
            self.assertEqual(tensor.layer_stride, plan.num_blocks * self.page)
        draft_groups = [group for group in plan.kv_cache_groups if group.is_eagle_group]
        self.assertEqual(len(draft_groups), 3)
        self.assertEqual({name for group in draft_groups for name in group.layer_names}, self.draft_names)
        self.assertTrue(
            all(
                not group.is_eagle_group
                for group in plan.kv_cache_groups
                if isinstance(group.kv_cache_spec, _MambaSpec)
            )
        )
        messages = [call.args[0] % call.args[1:] for call in self.impl.logger.info.call_args_list]
        self.assertTrue(any("groups=69 -> 35" in message for message in messages))
        self.assertTrue(any("fia_limit_blocks=5461" in message for message in messages))
        self.assertTrue(any("effective_budget_limit_blocks=4087" in message for message in messages))

    def test_explicit_override_is_preserved_on_original_config_and_budgeted_on_replan(self):
        self.config.cache_config.num_gpu_blocks_override = 4096
        (plan,) = self.plan(self.config, [self._worker_specs()], [self.budget])
        self.assertEqual(self.config.cache_config.num_gpu_blocks_override, 4096)
        self.assertEqual(self.calls[-1].cache_config.num_gpu_blocks_override, 4087)
        self.assertEqual((plan.num_blocks, len(plan.kv_cache_groups)), (4087, 35))

    def test_asymmetric_worker_budgets_share_one_safe_grouping(self):
        budgets = [9000 * self.page, 7500 * self.page]
        plans = self.plan(self.config, [self._worker_specs(), self._worker_specs()], budgets)
        self.assertEqual([plan.num_blocks for plan in plans], [3750, 3750])
        self.assertEqual(plans[0].kv_cache_groups, plans[1].kv_cache_groups)
        self.assertEqual(len(plans[0].kv_cache_groups), 35)
        for plan, budget in zip(plans, budgets):
            self.assertLessEqual(plan.num_blocks * self.impl._pool_bytes_per_block(plan), budget)

    def test_missing_role_metadata_retains_original_grouping_and_address_guard(self):
        (plan,) = self.plan(self.config, [self._worker_specs(include_roles=False)], [self.budget])
        self.assertEqual((plan.num_blocks, len(plan.kv_cache_groups)), (5461, 69))
        self.assertFalse(any(group.is_eagle_group for group in plan.kv_cache_groups))

    def test_workers_with_missing_or_different_roles_are_rejected_before_planning(self):
        alternatives = (
            self._worker_specs(include_roles=False),
            self.impl.DFlashKVCacheSpecs(self.specs, {"cache.16"}),
        )
        for other in alternatives:
            with self.subTest(other=type(other).__name__), self.assertRaisesRegex(ValueError, "disagree"):
                self.plan(self.config, [self._worker_specs(), other], [self.budget] * 2)
        self.assertEqual(self.calls, [])

    def test_unknown_or_mamba_rpc_draft_roles_are_rejected_before_planning(self):
        for invalid in ("cache.21", "missing.layer"):
            specs = self.impl.DFlashKVCacheSpecs(self.specs, self.draft_names | {invalid})
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "valid loaded draft"):
                self.plan(self.config, [specs], [self.budget])
        self.assertEqual(self.calls, [])

    def test_selected_group_builder_rejects_changed_specs_or_coverage(self):
        (plan,) = self.plan(self.config, [self._worker_specs()], [self.budget])
        config = copy.copy(self.config)
        config._ascend_dflash_cache_groups = plan.kv_cache_groups
        changed = dict(self.specs)
        changed["cache.17"] = replace(changed["cache.17"], block_size=128)
        with self.assertRaisesRegex(ValueError, "changed cache layer coverage or specs"):
            self.build(config, changed)
        with self.assertRaisesRegex(ValueError, "changed cache layer coverage or specs"):
            self.build(config, {name: spec for name, spec in self.specs.items() if name != "cache.0"})

    def test_selected_group_builder_does_not_alias_mutable_layer_lists(self):
        (plan,) = self.plan(self.config, [self._worker_specs()], [self.budget])
        config = copy.copy(self.config)
        config._ascend_dflash_cache_groups = plan.kv_cache_groups
        rebuilt = self.build(config, dict(self.specs))
        self.assertEqual(rebuilt, plan.kv_cache_groups)
        self.assertIsNot(rebuilt[0], plan.kv_cache_groups[0])
        self.assertIsNot(rebuilt[0].layer_names, plan.kv_cache_groups[0].layer_names)

    def test_replanner_losing_selected_groups_or_draft_roles_fails_closed(self):
        for failure in ("discard_selection", "drop_draft_role", "empty_worker_groups"):

            def original(config, specs, budgets, failure=failure):
                regrouping = hasattr(config, "_ascend_dflash_cache_groups")
                if regrouping and failure == "discard_selection":
                    config = copy.copy(config)
                    del config._ascend_dflash_cache_groups
                plans = self._original_planner(config, specs, budgets)
                if regrouping and failure == "drop_draft_role":
                    draft_group = next(group for group in plans[0].kv_cache_groups if group.is_eagle_group)
                    draft_group.is_eagle_group = False
                if regrouping and failure == "empty_worker_groups":
                    plans[0].kv_cache_groups = []
                return plans

            planner = self.impl.wrap_dflash_cache_planner(original)
            with (
                self.subTest(failure=failure),
                self.assertRaisesRegex(ValueError, "selected groups or draft ownership"),
            ):
                planner(self.config, [self._worker_specs(), self._worker_specs()], [self.budget] * 2)

    def test_annotation_corrects_target_and_mamba_flags_before_warning(self):
        config = self.impl._with_dflash_draft_layers(self.config, self._worker_specs())
        groups = [_KVCacheGroupSpec([name], spec, is_eagle_group=True) for name, spec in self.specs.items()]
        original = Mock()
        annotation = self.impl.wrap_dflash_cache_group_annotation(original)
        annotation(config, dict(self.specs), groups, use_deepseek_v4_fallback=False)
        original.assert_called_once_with(config, dict(self.specs), groups, use_deepseek_v4_fallback=False)
        self.assertEqual({group.layer_names[0] for group in groups if group.is_eagle_group}, self.draft_names)


if __name__ == "__main__":
    unittest.main()
