# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

"""Local-weight regression for Qwen3.5 + DFlash on one Ascend NPU.

Mount the target and draft at the paths below. Run with VLLM_USE_V2_MODEL_RUNNER=1.
Compare greedy tokens AND termination with the target-only reference; allowing
EOS is intentional. This does not replace the stochastic GPQA accuracy run.
"""

import os
from unittest.mock import patch

import pytest
from vllm import SamplingParams

from tests.e2e.conftest import VllmRunner

TARGET = "/data/weights/Qwen3.5-9B"
DRAFT = "/data/weights/Qwen3.5-9B-DFlash"


@pytest.mark.parametrize("enforce_eager", [True, False])
@patch.dict(os.environ, {"VLLM_USE_V2_MODEL_RUNNER": "1"})
def test_dflash_matches_target_tokens_and_termination(enforce_eager):
    # More requests than max_num_seqs forces request-slot reuse. Different
    # context/output lengths exercise chunked prefill and changing batches.
    prompts = [
        "The sequence is 1, 2, 3, 4. " * (1, 64, 1024, 2048)[i % 4]
        + f"\nExplain how to calculate the sum of integers from 1 to {100 + i}."
        for i in range(32)
    ]
    params = [
        SamplingParams(temperature=0, max_tokens=(64, 256, 1024)[i % 3], ignore_eos=False) for i in range(len(prompts))
    ]
    common = dict(
        max_model_len=80000,
        block_size=None,
        max_num_batched_tokens=16384,
        max_num_seqs=16,
        gpu_memory_utilization=0.9,
        async_scheduling=True,
        enable_prefix_caching=True,
        enforce_eager=enforce_eager,
        compilation_config={"cudagraph_capture_sizes": [8, 16, 32], "cudagraph_mode": "FULL_DECODE_ONLY"},
    )

    def run(speculative_config=None):
        rounds = []
        with VllmRunner(TARGET, speculative_config=speculative_config, **common) as runner:
            for _ in range(2):
                # Repeat to cover prefix hits and resumed Mamba state columns.
                outputs = runner.model.generate(prompts, params)
                assert len(outputs) == len(prompts)
                rounds.append(
                    [
                        (
                            list(result.outputs[0].token_ids),
                            result.outputs[0].finish_reason,
                            result.outputs[0].stop_reason,
                        )
                        for result in outputs
                    ]
                )
        return rounds

    reference = run()
    actual = run({"method": "dflash", "model": DRAFT, "num_speculative_tokens": 7})
    for round_idx, (expected_round, actual_round) in enumerate(zip(reference, actual)):
        for request_idx, (expected, result) in enumerate(zip(expected_round, actual_round)):
            assert result == expected, (
                f"DFlash diverged at round={round_idx}, request={request_idx}: "
                f"target length/termination={len(expected[0]), expected[1:]}, "
                f"draft length/termination={len(result[0]), result[1:]}"
            )
