# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""CPU contracts for the production hybrid-spec finalization block.

Only dependency fixtures and layer discovery are substituted. These tests do
not run the real allocator, attention operators or NPU graph execution.
"""

import ast
import unittest
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace


@dataclass(frozen=True)
class _AttentionSpec:
    block_size: int
    page_size_padded: int | None = None
    indexes_kv_by_block_stride: bool = False

    @property
    def page_size_bytes(self):
        return self.page_size_padded or self.block_size * 2048


@dataclass(frozen=True)
class _SlidingWindowSpec(_AttentionSpec):
    sliding_window: int = 2048


class TestDFlashBlockSize(unittest.TestCase):
    def setUp(self):
        source = Path(__file__).resolve().parents[3] / "vllm_ascend/worker/v2/attn_utils.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "get_kv_cache_spec")
        finalization = next(
            n
            for n in function.body
            if isinstance(n, ast.If) and isinstance(n.test, ast.Name) and n.test.id == "mamba_specs"
        )
        function.body = [finalization, function.body[-1]]
        function.returns = None
        function.args.args[0].annotation = None
        self.specs = {"full": _AttentionSpec(1536), "swa": _SlidingWindowSpec(128)}
        self.mamba = {"mamba": SimpleNamespace(page_size_bytes=3248128)}
        self.namespace = dict(
            replace=replace,
            SlidingWindowSpec=_SlidingWindowSpec,
            kv_cache_spec=self.specs,
            mamba_specs=self.mamba,
            attention_layer_names=list(self.specs),
            vllm_version_is=lambda _: False,
        )
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), self.namespace)
        self.finalize = self.namespace["get_kv_cache_spec"]
        self.config = SimpleNamespace(
            speculative_config=SimpleNamespace(
                method="dflash",
                draft_model_config=SimpleNamespace(
                    hf_config=SimpleNamespace(layer_types=["sliding_attention"] * 4 + ["full_attention"])
                ),
            ),
            cache_config=SimpleNamespace(block_size=1536),
        )

    def test_mixed_storage_blocks_match_without_changing_window(self):
        original = self.specs["swa"]
        result = self.finalize(self.config)
        self.assertEqual(result["full"].block_size, 1536)
        self.assertEqual(result["swa"].block_size, 1536)
        self.assertEqual(result["swa"].sliding_window, 2048)
        self.assertEqual(result["swa"].page_size_bytes, 3248128)
        self.assertEqual(original.block_size, 128)
        self.assertIs(result["mamba"], self.mamba["mamba"])

    def test_uses_resolved_full_block_instead_of_hardcoding_tp2(self):
        self.config.cache_config.block_size = 768
        self.specs["full"] = _AttentionSpec(768)
        self.assertEqual(self.finalize(self.config)["swa"].block_size, 768)

    def test_all_sliding_draft_is_unchanged(self):
        self.config.speculative_config.draft_model_config.hf_config.layer_types = ["sliding_attention"] * 5
        self.assertEqual(self.finalize(self.config)["swa"].block_size, 128)

    def test_all_full_draft_is_unchanged(self):
        self.config.speculative_config.draft_model_config.hf_config.layer_types = ["full_attention"] * 5
        self.assertEqual(self.finalize(self.config)["swa"].block_size, 128)

    def test_target_only_is_unchanged(self):
        self.config.speculative_config = None
        self.assertEqual(self.finalize(self.config)["swa"].block_size, 128)

    def test_other_speculative_method_is_unchanged(self):
        self.config.speculative_config.method = "eagle"
        self.assertEqual(self.finalize(self.config)["swa"].block_size, 128)

    def test_without_mamba_is_unchanged(self):
        self.mamba.clear()
        self.assertEqual(self.finalize(self.config)["swa"].block_size, 128)

    def test_release_padding_flag_is_preserved(self):
        self.namespace["vllm_version_is"] = lambda _: True
        spec = self.finalize(self.config)["swa"]
        self.assertEqual(spec.block_size, 1536)
        self.assertTrue(spec.indexes_kv_by_block_stride)


if __name__ == "__main__":
    unittest.main()
