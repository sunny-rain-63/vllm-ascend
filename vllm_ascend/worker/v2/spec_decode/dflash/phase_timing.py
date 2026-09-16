# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Env-gated per-phase wall-clock timing for the DFlash propose path.

The mixed Full/SWA drafter carries per-step costs that do not exist for an
all-full drafter (a second KV cache group, windowed FIA calls, sliding-window
bookkeeping). This timer attributes propose-step wall time to individual
phases so the dominant term can be identified from a single serving run.

Enable with::

    VLLM_ASCEND_DFLASH_PHASE_TIMING=1
    VLLM_ASCEND_DFLASH_PHASE_TIMING_INTERVAL=50   # optional, default 50

When disabled (default), :func:`get_phase_timer` returns ``None`` and every
instrumented call site collapses to a single ``is None`` check.

Timing uses ``torch.npu.synchronize`` around each phase, so it serializes the
pipeline it measures. Use it for diagnosis, not for production benchmarks.
"""

import time
from collections.abc import Iterator
from contextlib import contextmanager

import torch
from vllm.logger import init_logger

import vllm_ascend.envs as envs

logger = init_logger(__name__)


class DFlashPhaseTimer:
    """Accumulates per-phase wall time across propose steps."""

    def __init__(self, interval: int) -> None:
        self.interval = max(1, interval)
        self.steps = 0
        self.full_graph_steps = 0
        self.eager_steps = 0
        self.phase_seconds: dict[str, float] = {}
        self._start_total: float | None = None

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        if torch.npu.is_current_stream_capturing():
            # Synchronizing inside graph capture is illegal; never time there.
            yield
            return
        torch.npu.synchronize()
        start = time.perf_counter()
        try:
            yield
        finally:
            torch.npu.synchronize()
            self.phase_seconds[name] = self.phase_seconds.get(name, 0.0) + (time.perf_counter() - start)

    def begin_step(self) -> None:
        torch.npu.synchronize()
        self._start_total = time.perf_counter()

    def end_step(self, num_groups: int) -> None:
        torch.npu.synchronize()
        total = time.perf_counter() - (self._start_total or time.perf_counter())
        self._start_total = None
        self.phase_seconds["propose_total"] = self.phase_seconds.get("propose_total", 0.0) + total
        self.steps += 1
        if self.steps % self.interval:
            return
        per_step = {name: value / self.steps * 1e3 for name, value in sorted(self.phase_seconds.items())}
        breakdown = ", ".join(f"{name}={ms:.3f}ms" for name, ms in per_step.items())
        logger.info(
            "DFlash phase timing over %d steps (kv_groups=%d, full_graph=%d, eager=%d): %s",
            self.steps,
            num_groups,
            self.full_graph_steps,
            self.eager_steps,
            breakdown,
        )
        self.steps = 0
        self.full_graph_steps = 0
        self.eager_steps = 0
        self.phase_seconds.clear()


def get_phase_timer() -> DFlashPhaseTimer | None:
    """Return a timer when enabled, else ``None`` (checked once per speculator)."""
    if not envs.VLLM_ASCEND_DFLASH_PHASE_TIMING:
        return None
    return DFlashPhaseTimer(envs.VLLM_ASCEND_DFLASH_PHASE_TIMING_INTERVAL)
