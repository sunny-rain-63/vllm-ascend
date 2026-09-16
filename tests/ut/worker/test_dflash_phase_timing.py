# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""CPU checks for the env-gated DFlash phase timer.

The module is loaded by file path behind stub ``torch`` / ``vllm.logger`` /
``vllm_ascend.envs`` modules so these tests run without torch, torch_npu,
pytest or an NPU. They cover the enable gate, phase accumulation, capture
passthrough and the interval log/reset cycle — not NPU timing accuracy.
"""

import importlib.util
import logging
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

PHASE_TIMING_PATH = (
    Path(__file__).resolve().parents[3]
    / "vllm_ascend"
    / "worker"
    / "v2"
    / "spec_decode"
    / "dflash"
    / "phase_timing.py"
)


def _stub_modules(*, enabled: str, interval: str = "2"):
    fake_npu = types.SimpleNamespace(
        synchronize=lambda: None,
        is_current_stream_capturing=lambda: False,
    )
    fake_torch = types.ModuleType("torch")
    fake_torch.npu = fake_npu

    fake_vllm = types.ModuleType("vllm")
    fake_vllm_logger = types.ModuleType("vllm.logger")
    fake_vllm_logger.init_logger = logging.getLogger
    fake_vllm.logger = fake_vllm_logger

    fake_pkg = types.ModuleType("vllm_ascend")
    fake_envs = types.ModuleType("vllm_ascend.envs")
    fake_envs.VLLM_ASCEND_DFLASH_PHASE_TIMING = enabled == "1"
    fake_envs.VLLM_ASCEND_DFLASH_PHASE_TIMING_INTERVAL = int(interval)
    fake_pkg.envs = fake_envs

    stubs = {
        "torch": fake_torch,
        "vllm": fake_vllm,
        "vllm.logger": fake_vllm_logger,
        "vllm_ascend": fake_pkg,
        "vllm_ascend.envs": fake_envs,
    }
    return mock.patch.dict(sys.modules, stubs), fake_npu


def _exec_phase_timing():
    spec = importlib.util.spec_from_file_location("_dflash_phase_timing_under_test", PHASE_TIMING_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestDFlashPhaseTimer(unittest.TestCase):
    def test_disabled_by_default(self):
        stubs, _ = _stub_modules(enabled="0")
        with stubs:
            module = _exec_phase_timing()
            self.assertIsNone(module.get_phase_timer())

    def test_enabled_reads_interval(self):
        stubs, _ = _stub_modules(enabled="1", interval="7")
        with stubs:
            module = _exec_phase_timing()
            timer = module.get_phase_timer()
            self.assertIsNotNone(timer)
            self.assertEqual(timer.interval, 7)

    def test_phase_accumulates_and_logs_at_interval(self):
        stubs, _ = _stub_modules(enabled="1", interval="2")
        with stubs:
            module = _exec_phase_timing()
            timer = module.get_phase_timer()
            with timer.phase("metadata_build"):
                pass
            self.assertIn("metadata_build", timer.phase_seconds)

            timer.begin_step()
            timer.end_step(num_groups=2)
            self.assertEqual(timer.steps, 1)
            timer.begin_step()
            timer.full_graph_steps += 1
            with self.assertLogs("_dflash_phase_timing_under_test", level="INFO") as logs:
                timer.end_step(num_groups=2)
            self.assertIn("kv_groups=2", logs.output[0])
            self.assertIn("full_graph=1", logs.output[0])
            # The cycle resets after logging.
            self.assertEqual(timer.steps, 0)
            self.assertEqual(timer.phase_seconds, {})

    def test_phase_passes_through_during_capture(self):
        stubs, fake_npu = _stub_modules(enabled="1")
        with stubs:
            module = _exec_phase_timing()
            timer = module.get_phase_timer()
            fake_npu.is_current_stream_capturing = lambda: True
            with timer.phase("graph_replay"):
                pass
            self.assertEqual(timer.phase_seconds, {})


if __name__ == "__main__":
    unittest.main()
