# SPDX-License-Identifier: Apache-2.0
"""Run without NPU setup: pytest --confcutdir=tests/ut/tools <this file>."""

import copy
import importlib.util
import json
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]


def load_tool(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def validator():
    return load_tool("validate_qwen4_exp_mtp")


@pytest.fixture
def worker_tool(monkeypatch):
    boundary = ModuleType("vllm_ascend.worker.worker")
    boundary.NPUWorker = type("NPUWorker", (), {})
    monkeypatch.setitem(sys.modules, boundary.__name__, boundary)
    return load_tool("qwen4_exp_mtp_diagnostics_worker")


def configuration(tmp_path):
    return {
        "model": "/model",
        "seed": 42,
        "batch_sizes": [1, 8],
        "num_speculative_tokens": 3,
        "output_dir": str(tmp_path),
        "engine_args": {
            "tensor_parallel_size": 8,
            "additional_config": {"enable_cpu_binding": True, "ascend_compilation_config": {"fuse_norm_quant": False}},
            "hf_overrides": {"existing_field": True, "text_config": {"existing_nested_field": 7}},
        },
    }


@pytest.mark.parametrize("mode", ["base", "draft_eager", "draft_graph", "prefill_graph", "decode_graph"])
def test_mode_kwargs_and_preserved_additional_config(validator, tmp_path, mode):
    config = configuration(tmp_path)
    config.update(graph_trace_limit=4, disable_index_sharing=True)
    original = copy.deepcopy(config)
    kwargs = validator.build_engine_args(config, mode)
    assert config == original
    assert kwargs["worker_cls"] == "tools.qwen4_exp_mtp_diagnostics_worker.DiagnosticNPUWorker"
    additional = kwargs["additional_config"]
    assert additional["enable_cpu_binding"] is True
    assert additional["ascend_compilation_config"] == {"fuse_norm_quant": False}
    assert additional == original["engine_args"]["additional_config"]
    assert kwargs["hf_overrides"]["existing_field"] is True
    assert kwargs["hf_overrides"]["text_config"] == {"existing_nested_field": 7}
    options = kwargs["hf_overrides"]["qwen4_exp_mtp_diagnostics"]
    assert options["disable_prefill_graph"] == (mode == "decode_graph")
    assert options["disable_decode_graph"] == (mode == "prefill_graph")
    if mode == "base":
        assert "speculative_config" not in kwargs
    else:
        assert kwargs["speculative_config"]["enforce_eager"] == (mode == "draft_eager")
        assert kwargs["speculative_config"]["index_share_for_mtp_iteration"] is False


def test_no_hooks_when_diagnostics_not_requested(validator, tmp_path):
    kwargs = validator.build_engine_args(configuration(tmp_path), "draft_graph")
    assert "worker_cls" not in kwargs
    assert "qwen4_exp_mtp_diagnostics" not in kwargs["hf_overrides"]


def test_pool_option_requires_full_decode_only(validator, tmp_path):
    config = configuration(tmp_path)
    config["separate_graph_pools"] = True
    config["engine_args"]["compilation_config"] = {"cudagraph_mode": "FULL_AND_PIECEWISE"}
    with pytest.raises(ValueError, match="FULL_DECODE_ONLY"):
        validator.build_engine_args(config, "draft_graph")


def report(tokens, accepted):
    return {
        "runs": [
            {
                "run_index": 0,
                "batch_size": 1,
                "requests": [{"max_tokens": 3, "output_token_ids": tokens}],
                "speculative_counters": {"drafts": 2, "accepted_tokens": accepted},
            }
        ]
    }


def test_comparison_supports_selected_modes_without_base(validator):
    result = validator.compare_results({"prefill_graph": report([1, 2], 4), "decode_graph": report([1, 3], 2)})
    assert result["token_mismatches_against_base"] == []
    assert result["graph_to_eager_acceptance_ratio"] is None
    assert result["acceptance"]["prefill_graph"]["mean_acceptance_length"] == 3


def test_comparison_keeps_eager_graph_acceptance_and_base_tokens(validator):
    result = validator.compare_results(
        {
            "base": report([1, 2], 0),
            "draft_eager": report([1, 2], 4),
            "draft_graph": report([1, 3], 2),
        }
    )
    assert result["graph_to_eager_acceptance_ratio"] == 2 / 3
    assert result["token_mismatches_against_base"] == [
        {"mode": "draft_graph", "run_index": 0, "request_index": 0, "first_different_token": 1}
    ]


class Mode(Enum):
    FULL = 1
    NONE = 0


@dataclass(frozen=True)
class Descriptor:
    cg_mode: Mode
    num_tokens: int = 8
    num_reqs: int = 2


class FakeManager:
    def __init__(self, is_prefill=False):
        self.is_draft_model_prefill = is_prefill
        self._capture_descs = {Mode.FULL: [Descriptor(Mode.FULL)]}
        self._candidates = {8: [Descriptor(Mode.FULL)]}
        self.capture_sizes = [8]
        self.graphs = {}
        self.pool = "global"
        self.use_breakable_cg = False

    def needs_capture(self):
        return bool(self._capture_descs)

    def capture(self):
        for descriptors in self._capture_descs.values():
            self.graphs.update(dict.fromkeys(descriptors, "captured"))

    def dispatch(self):
        return Descriptor(Mode.FULL if self._candidates else Mode.NONE)

    def run_fullgraph(self, desc):
        return self.graphs[desc]


@pytest.mark.parametrize("disable_phase", ["prefill", "decode"])
def test_phase_hooks_dispatch_and_one_pool_per_manager(worker_tool, tmp_path, disable_phase):
    class Target(FakeManager):
        pass

    class Draft(FakeManager):
        pass

    pools = []

    def pool_factory():
        pools.append(f"pool-{len(pools)}")
        return pools[-1]

    options = {
        f"disable_{disable_phase}_graph": True,
        "separate_graph_pools": True,
        "trace_limit": 2,
        "trace_dir": str(tmp_path),
    }
    trace = worker_tool.GraphTrace(options, rank=3)
    restorers = worker_tool.install_manager_hooks(Target, Draft, options, trace, pool_factory)
    try:
        managers = [Target(), Draft(True), Draft(False)]
        expected_disabled = 1 if disable_phase == "prefill" else 2
        for index, manager in enumerate(managers):
            manager.capture()
            mode = manager.dispatch().cg_mode
            assert mode == (Mode.NONE if index == expected_disabled else Mode.FULL)
            if mode == Mode.FULL:
                for _ in range(4):
                    assert manager.run_fullgraph(manager.dispatch()) == "captured"
        assert pools == ["pool-0", "pool-1"]
        assert len({manager.pool for index, manager in enumerate(managers) if index != expected_disabled}) == 2
        events = [json.loads(line) for line in trace.path.read_text(encoding="utf-8").splitlines()]
        assert sum(row["event"] == "replay" for row in events) == 4  # Limit 2 per enabled phase.
        assert all(row["phase"] in {"target", "draft_prefill", "draft_decode"} for row in events)
    finally:
        for restore in reversed(restorers):
            restore()
    assert "__init__" not in Target.__dict__
    assert "dispatch" not in Draft.__dict__


def test_descriptor_probe_does_not_read_device_values(worker_tool):
    class DeviceValue:
        def item(self):
            raise AssertionError("Device synchronization is forbidden")

    description = worker_tool.describe_descriptor(SimpleNamespace(cg_mode=Mode.FULL, num_tokens=DeviceValue()))
    assert description["cg_mode"] == "FULL"
    assert description["num_tokens"] == "DeviceValue"


def test_trace_quota_does_not_hide_a_recreated_manager(worker_tool, tmp_path):
    trace = worker_tool.GraphTrace({"trace_limit": 1, "trace_dir": str(tmp_path)}, rank=0)
    for manager_id in (1, 1, 2, 2):
        trace.record("target", "captured", manager_id=manager_id)
    rows = [json.loads(line) for line in trace.path.read_text(encoding="utf-8").splitlines()]
    assert [row["manager_id"] for row in rows] == [1, 2]


def test_gsm8k_questions_offset_and_selected_modes_dry_run(validator, monkeypatch, tmp_path):
    dataset = tmp_path / "test.jsonl"
    dataset.write_text(
        "\n".join(json.dumps({"question": f"question {index}", "answer": "unused"}) for index in range(3))
    )
    output = tmp_path / "report"
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate_qwen4_exp_mtp.py",
            "--model",
            "/model",
            "--gsm8k-jsonl",
            str(dataset),
            "--offset",
            "1",
            "--modes",
            "prefill_graph,decode_graph",
            "--output-dir",
            str(output),
            "--dry-run",
        ],
    )
    assert validator.main() == 0
    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    assert config["prompts"] == ["question 1", "question 2"]
    assert config["modes"] == ["prefill_graph", "decode_graph"]
    kwargs = json.loads((output / "engine_args_by_mode.json").read_text(encoding="utf-8"))
    assert set(kwargs) == {"prefill_graph", "decode_graph"}


def test_prompt_sources_are_mutually_exclusive(validator, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["validator", "--prompts", "a.json", "--gsm8k-jsonl", "b.jsonl"])
    with pytest.raises(SystemExit):
        validator.parse_args()


def test_worker_installs_hooks_after_device_setup(worker_tool, monkeypatch, tmp_path):
    def init_device(worker):
        worker.device_ready = True
        worker.model_runner = object()
        return "device initialized"

    monkeypatch.setattr(worker_tool.DiagnosticNPUWorker.__bases__[0], "init_device", init_device, raising=False)
    torch_boundary = ModuleType("torch")
    torch_boundary.npu = SimpleNamespace(graph_pool_handle=lambda: "pool")
    monkeypatch.setitem(sys.modules, "torch", torch_boundary)
    target_module = ModuleType("vllm_ascend.worker.v2.aclgraph_utils")
    target_module.ModelAclGraphManager = object()
    draft_module = ModuleType("vllm_ascend.worker.v2.spec_decode.autoregressive.aclgraph")
    draft_module.AutoRegressiveAclGraphManager = object()
    monkeypatch.setitem(sys.modules, target_module.__name__, target_module)
    monkeypatch.setitem(sys.modules, draft_module.__name__, draft_module)
    worker = worker_tool.DiagnosticNPUWorker()
    options = {"trace_dir": str(tmp_path), "trace_limit": 0}
    worker.vllm_config = SimpleNamespace(
        additional_config={"enable_cpu_binding": True},
        model_config=SimpleNamespace(hf_config=SimpleNamespace(qwen4_exp_mtp_diagnostics=options)),
    )
    worker.rank = 0
    installed = []

    def install(target, draft, received, trace, pool_factory):
        assert worker.device_ready and worker.model_runner is not None
        assert received is options
        assert target is target_module.ModelAclGraphManager
        assert draft is draft_module.AutoRegressiveAclGraphManager
        installed.append(True)
        return []

    monkeypatch.setattr(worker_tool, "install_manager_hooks", install)
    assert worker.init_device() == "device initialized"
    assert installed == [True]
