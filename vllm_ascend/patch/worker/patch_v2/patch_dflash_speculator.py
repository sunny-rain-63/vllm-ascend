# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
import vllm.v1.worker.gpu.spec_decode.dflash.cudagraph as cudagraph_module
import vllm.v1.worker.gpu.spec_decode.dflash.speculator as speculator_module
import vllm.v1.worker.gpu.spec_decode.dflash2.speculator as speculator2_module
from vllm.model_executor.models.qwen3_dflash import DFlashQwen3Model

from vllm_ascend.worker.v2.attn_utils import build_attn_metadata
from vllm_ascend.worker.v2.spec_decode.dflash.aclgraph import DFlashAclGraphManager
from vllm_ascend.worker.v2.spec_decode.dflash2.speculator import (
    _selector_walk_kernel_ascend,
)

cudagraph_module.build_attn_metadata = build_attn_metadata
speculator_module.DFlashCudaGraphManager = DFlashAclGraphManager

# triton-ascend cannot lower tldevice.log1p in the upstream selector walk;
# swap in the algebraically equivalent log(1 - u) variant.
speculator2_module._selector_walk_kernel = _selector_walk_kernel_ascend

# Optional per-stage step timing, enabled with VLLM_ASCEND_DFLASH_PROFILE=1.
# Imported late so patch_qwen3_dflash's precompute override is already in
# place (patch.worker.__init__ imports it before this module).
from vllm_ascend.worker.v2.spec_decode.dflash import profiling as _dflash_profiling

if _dflash_profiling.ENABLED:
    _timed = _dflash_profiling.timed

    _orig_prepare = speculator_module.prepare_dflash_inputs

    def _timed_prepare(*args, **kwargs):
        with _timed("prepare"):
            return _orig_prepare(*args, **kwargs)

    speculator_module.prepare_dflash_inputs = _timed_prepare

    _orig_precompute = DFlashQwen3Model.precompute_and_store_context_kv

    def _timed_precompute(self, *args, **kwargs):
        with _timed("kv_precompute"):
            return _orig_precompute(self, *args, **kwargs)

    DFlashQwen3Model.precompute_and_store_context_kv = _timed_precompute

    _orig_run_fullgraph = DFlashAclGraphManager.run_fullgraph

    def _timed_run_fullgraph(self, *args, **kwargs):
        with _timed("draft_forward"):
            return _orig_run_fullgraph(self, *args, **kwargs)

    DFlashAclGraphManager.run_fullgraph = _timed_run_fullgraph

    _orig_generate_draft = speculator_module.DFlashSpeculator._generate_draft

    def _timed_generate_draft(self, *args, **kwargs):
        with _timed("draft_forward"):
            return _orig_generate_draft(self, *args, **kwargs)

    speculator_module.DFlashSpeculator._generate_draft = _timed_generate_draft

    _orig_propose = speculator_module.DFlashSpeculator.propose

    def _timed_propose(self, *args, **kwargs):
        with _timed("propose_total"):
            out = _orig_propose(self, *args, **kwargs)
        _dflash_profiling.profiler.step_done()
        return out

    speculator_module.DFlashSpeculator.propose = _timed_propose
