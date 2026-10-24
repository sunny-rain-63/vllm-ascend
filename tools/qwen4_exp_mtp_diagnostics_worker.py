# SPDX-License-Identifier: Apache-2.0
"""Worker-local graph diagnostics used only by validate_qwen4_exp_mtp.py."""

import functools
import json
from collections import Counter
from pathlib import Path

from vllm_ascend.worker.worker import NPUWorker


def describe_descriptor(desc):
    result = {}
    for name in (
        "cg_mode",
        "num_tokens",
        "num_reqs",
        "uniform_token_count",
        "max_query_len",
        "num_active_loras",
        "num_ubatches",
        "num_speculative_tokens",
    ):
        value = getattr(desc, name, None)
        if name == "cg_mode":
            value = getattr(value, "name", str(value))
        # Do not inspect or transfer any device tensor in these probes.
        result[name] = value if value is None or isinstance(value, (bool, int, float, str)) else type(value).__name__
    return result


class GraphTrace:
    def __init__(self, options, rank):
        self.limit = options.get("trace_limit", 0)
        self.counts = Counter()
        self.path = Path(options["trace_dir"]) / f"rank-{rank}.jsonl"
        if self.limit:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text("", encoding="utf-8")

    def record(self, phase, event, **fields):
        key = (phase, event, fields.get("manager_id"))
        if self.counts[key] >= self.limit:
            return
        self.counts[key] += 1
        with self.path.open("a", encoding="utf-8") as output:
            output.write(json.dumps({"phase": phase, "event": event, **fields}) + "\n")


def configure_manager(manager, phase, options, pool_factory):
    disabled = (phase == "draft_prefill" and options.get("disable_prefill_graph", False)) or (
        phase == "draft_decode" and options.get("disable_decode_graph", False)
    )
    if disabled:
        # Keep upstream dispatch: an empty candidate table naturally selects
        # NONE. Clearing only capture descriptors would leave FULL candidates.
        manager._capture_descs.clear()
        manager._candidates.clear()
        manager.capture_sizes = []
        manager.use_breakable_cg = False
    if options.get("separate_graph_pools") and manager.needs_capture():
        # One handle per manager, reused across all its capture sizes.
        manager._qwen_diagnostic_pool = pool_factory()
        manager.pool = manager._qwen_diagnostic_pool
    manager._qwen_diagnostic_phase = phase
    return disabled


def install_manager_hooks(target_cls, draft_cls, options, trace, pool_factory):
    """Install hooks in one worker process; return exact restoration callbacks."""
    restorers = []

    def patch_method(cls, name, wrapper):
        original = getattr(cls, name)
        owned = name in cls.__dict__
        setattr(cls, name, functools.wraps(original)(wrapper(original)))

        def restore():
            if owned:
                setattr(cls, name, original)
            else:
                delattr(cls, name)

        restorers.append(restore)

    for cls, is_draft in ((target_cls, False), (draft_cls, True)):

        def init_wrapper(original, is_draft=is_draft):
            def initialize(manager, *args, **kwargs):
                original(manager, *args, **kwargs)
                phase = "target"
                if is_draft:
                    phase = "draft_prefill" if manager.is_draft_model_prefill else "draft_decode"
                disabled = configure_manager(manager, phase, options, pool_factory)
                trace.record(phase, "manager_init", disabled=disabled, pool=repr(manager.pool), manager_id=id(manager))

            return initialize

        def capture_wrapper(original):
            def capture(manager, *args, **kwargs):
                phase = manager._qwen_diagnostic_phase
                if hasattr(manager, "_qwen_diagnostic_pool"):
                    # Memory profiling temporarily replaces manager.pool.
                    # Restore this diagnostic manager's own handle at capture.
                    manager.pool = manager._qwen_diagnostic_pool
                trace.record(phase, "capture_begin", pool=repr(manager.pool), manager_id=id(manager))
                result = original(manager, *args, **kwargs)
                for desc in manager.graphs:
                    trace.record(
                        phase,
                        "captured",
                        descriptor=describe_descriptor(desc),
                        pool=repr(manager.pool),
                        manager_id=id(manager),
                    )
                return result

            return capture

        def dispatch_wrapper(original):
            def dispatch(manager, *args, **kwargs):
                desc = original(manager, *args, **kwargs)
                trace.record(
                    manager._qwen_diagnostic_phase,
                    "dispatch",
                    descriptor=describe_descriptor(desc),
                    manager_id=id(manager),
                )
                return desc

            return dispatch

        def replay_wrapper(original):
            def replay(manager, desc, *args, **kwargs):
                trace.record(
                    manager._qwen_diagnostic_phase,
                    "replay",
                    descriptor=describe_descriptor(desc),
                    pool=repr(manager.pool),
                    manager_id=id(manager),
                )
                return original(manager, desc, *args, **kwargs)

            return replay

        patch_method(cls, "__init__", init_wrapper)
        patch_method(cls, "capture", capture_wrapper)
        patch_method(cls, "dispatch", dispatch_wrapper)
        patch_method(cls, "run_fullgraph", replay_wrapper)
    return restorers


class DiagnosticNPUWorker(NPUWorker):
    """Install probes after device/runner setup, before load/cache/capture."""

    def init_device(self):
        result = super().init_device()
        options = getattr(self.vllm_config.model_config.hf_config, "qwen4_exp_mtp_diagnostics", None)
        if not options:
            return result
        import torch

        from vllm_ascend.worker.v2.aclgraph_utils import ModelAclGraphManager
        from vllm_ascend.worker.v2.spec_decode.autoregressive.aclgraph import AutoRegressiveAclGraphManager

        self._qwen_diagnostic_restorers = install_manager_hooks(
            ModelAclGraphManager,
            AutoRegressiveAclGraphManager,
            options,
            GraphTrace(options, self.rank),
            torch.npu.graph_pool_handle,
        )
        return result

    def shutdown(self):
        try:
            return super().shutdown()
        finally:
            for restore in reversed(getattr(self, "_qwen_diagnostic_restorers", [])):
                restore()
            self._qwen_diagnostic_restorers = []
