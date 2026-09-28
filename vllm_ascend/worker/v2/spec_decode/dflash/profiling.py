# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Lightweight per-stage step profiler for the DFlash speculator.

Enabled by ``VLLM_ASCEND_DFLASH_PROFILE=1``. Every
``VLLM_ASCEND_DFLASH_PROFILE_WINDOW`` propose calls (default 50) it logs the
average per-stage device time in milliseconds. Each stage boundary forces a
device synchronize, so the numbers are diagnostic (they serialize pipelined
execution) — never enable this in production benchmarks you quote.

Stages:
  prepare        prepare_dflash_inputs triton kernel + host launch
  kv_precompute  precompute_and_store_context_kv (GEMM + grouped norm + RoPE
                 + per-layer cache insert)
  draft_forward  FULL aclgraph replay (or eager _generate_draft fallback)
  propose_total  the whole DFlashSpeculator.propose call
"""

import os
import time
from collections.abc import Iterator
from contextlib import contextmanager

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)

ENABLED = os.environ.get("VLLM_ASCEND_DFLASH_PROFILE", "0") == "1"
_WINDOW = int(os.environ.get("VLLM_ASCEND_DFLASH_PROFILE_WINDOW", "50"))


class DFlashStepProfiler:
    def __init__(self) -> None:
        self.n = 0
        self.totals: dict[str, float] = {}

    def record(self, stage: str, dt: float) -> None:
        self.totals[stage] = self.totals.get(stage, 0.0) + dt

    def step_done(self) -> None:
        self.n += 1
        if self.n >= _WINDOW:
            parts = [f"{stage}={dt / self.n * 1000:.3f}" for stage, dt in sorted(self.totals.items())]
            logger.info("DFlash profile: avg ms/step over %d proposes -> %s", self.n, ", ".join(parts))
            self.n = 0
            self.totals.clear()


profiler = DFlashStepProfiler()


@contextmanager
def timed(stage: str) -> Iterator[None]:
    if not ENABLED:
        yield
        return
    torch.npu.synchronize()
    t0 = time.perf_counter()
    try:
        yield
    finally:
        torch.npu.synchronize()
        profiler.record(stage, time.perf_counter() - t0)
