# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""CPU metadata-contract tests for DFlash draft SWA spec widening.

Execute the production functions with lightweight spec fixtures so these
regressions can also run with the standard library on a machine without torch.
Only external imports are substituted; the widening logic is real.
"""

import ast
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock


@dataclass(frozen=True)
class _FullAttentionSpec:
    block_size: int = 128
    num_kv_heads: int = 4
    head_size: int = 128
    head_size_v: int = 128
    dtype: str = "torch.bfloat16"
    sliding_window: int | None = None


@dataclass(frozen=True)
class _SlidingWindowSpec:
    block_size: int = 128
    num_kv_heads: int = 4
    head_size: int = 128
    head_size_v: int = 128
    dtype: str = "torch.bfloat16"
    sliding_window: int = 2048


def _load_production():
    root = Path(__file__).resolve().parents[3]
    source = root / "vllm_ascend/core/dflash_swa_spec.py"
    namespace = dict(
        logger=Mock(),
        FullAttentionSpec=_FullAttentionSpec,
        SlidingWindowSpec=_SlidingWindowSpec,
    )
    module = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    module.body = [
        node
        for node in module.body
        if not (isinstance(node, ast.ImportFrom) and (node.module or "").startswith("vllm"))
    ]
    exec(compile(module, str(source), "exec"), namespace)
    return SimpleNamespace(**namespace)


def _config(layer_types, *, method="dflash", use_v2=True):
    return SimpleNamespace(
        use_v2_model_runner=use_v2,
        speculative_config=(
            SimpleNamespace(
                method=method,
                draft_model_config=SimpleNamespace(hf_config=SimpleNamespace(layer_types=layer_types)),
            )
            if method
            else None
        ),
    )


class TestDFlashSwaSpecWidening(unittest.TestCase):
    def setUp(self):
        self.impl = _load_production()
        self.mixed = _config(["sliding_attention"] * 4 + ["full_attention"])
        self.draft_names = {f"draft.layers.{i}.self_attn.attn" for i in range(5)}

    def _specs(self):
        specs = {"model.layers.0.self_attn.attn": _FullAttentionSpec()}
        for i in range(4):
            specs[f"draft.layers.{i}.self_attn.attn"] = _SlidingWindowSpec()
        specs["draft.layers.4.self_attn.attn"] = _FullAttentionSpec()
        return specs

    def test_gate_matches_upstream_mixed_dflash_detection(self):
        self.assertTrue(self.impl.dflash_draft_has_mixed_windows(self.mixed))
        self.assertFalse(self.impl.dflash_draft_has_mixed_windows(_config(["full_attention"] * 5)))
        self.assertFalse(self.impl.dflash_draft_has_mixed_windows(_config(["sliding_attention"] * 5)))
        self.assertFalse(self.impl.dflash_draft_has_mixed_windows(_config(["sliding_attention"] * 4 + ["full_attention"], method="eagle")))
        self.assertFalse(self.impl.dflash_draft_has_mixed_windows(_config(["sliding_attention"] * 4 + ["full_attention"], use_v2=False)))
        self.assertFalse(self.impl.dflash_draft_has_mixed_windows(_config([], method=None)))

    def test_mixed_draft_swa_specs_are_widened_with_window_carried(self):
        specs = self._specs()
        result = self.impl.widen_dflash_draft_swa_specs(self.mixed, specs, self.draft_names)
        for i in range(4):
            name = f"draft.layers.{i}.self_attn.attn"
            widened = result[name]
            self.assertIs(type(widened), _FullAttentionSpec)
            self.assertEqual(widened.sliding_window, 2048)
            self.assertEqual(widened.block_size, specs[name].block_size)
            self.assertEqual(widened.num_kv_heads, specs[name].num_kv_heads)
            self.assertEqual(widened.head_size, specs[name].head_size)
            self.assertEqual(widened.head_size_v, specs[name].head_size_v)
            self.assertEqual(widened.dtype, specs[name].dtype)
            # The input mapping is not mutated.
            self.assertIs(type(specs[name]), _SlidingWindowSpec)
        # Draft full layers, target layers and non-draft specs are untouched.
        self.assertIs(result["draft.layers.4.self_attn.attn"], specs["draft.layers.4.self_attn.attn"])
        self.assertIs(result["model.layers.0.self_attn.attn"], specs["model.layers.0.self_attn.attn"])

    def test_target_swa_layers_keep_their_spec(self):
        specs = self._specs()
        specs["model.layers.1.self_attn.attn"] = _SlidingWindowSpec(sliding_window=4096)
        result = self.impl.widen_dflash_draft_swa_specs(self.mixed, specs, self.draft_names)
        self.assertIs(result["model.layers.1.self_attn.attn"], specs["model.layers.1.self_attn.attn"])

    def test_passthrough_outside_scope(self):
        specs = self._specs()
        for config in (_config(["full_attention"] * 5), _config(["sliding_attention"] * 5), _config([], method=None)):
            with self.subTest(config=config):
                self.assertIs(self.impl.widen_dflash_draft_swa_specs(config, specs, self.draft_names), specs)

    def test_missing_draft_names_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "loaded draft attention layer names"):
            self.impl.widen_dflash_draft_swa_specs(self.mixed, self._specs(), None)
        with self.assertRaisesRegex(ValueError, "missing from KV cache specs"):
            self.impl.widen_dflash_draft_swa_specs(self.mixed, self._specs(), {"draft.layers.99.self_attn.attn"})


if __name__ == "__main__":
    unittest.main()
