# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compiled NPU parity test and opt-in DFlash input-preparation microbenchmark.

Run parity with pytest. To compare identical shapes on fix_swa2 and fix_swa3,
run this file with PYTHONPATH=. and --benchmark on each branch. Copy both new
test files to the baseline checkout if necessary; do not copy production code.
The benchmark includes warmup and synchronization, but not model execution.
"""

import argparse
import importlib
import json
import statistics
import time

import pytest

from tests.ut.worker.test_dflash_prepare_inputs import make_case, reference_outputs

torch = pytest.importorskip("torch")
pytest.importorskip("torch_npu")
pytestmark = pytest.mark.skipif(not torch.npu.is_available(), reason="Requires an Ascend NPU")


def _prepare_launch(**case_kwargs):
    module = importlib.import_module("vllm_ascend.worker.v2.spec_decode.dflash.speculator")
    kernel = module._prepare_dflash_inputs_kernel_ascend
    arrays, scalars, grid = make_case(**case_kwargs)
    expected = reference_outputs(arrays, scalars, grid)
    tensors = {name: torch.from_numpy(array.copy()).to("npu") for name, array in arrays.items()}
    arguments = tensors | scalars
    arguments = {name: arguments[name] for name in kernel.arg_names}

    def launch():
        kernel[grid](**arguments)

    return launch, tensors, expected


@pytest.mark.parametrize("lengths", [(8,), (8,) * 32, (1, 255, 256, 257), (2047, 1025)])
@pytest.mark.parametrize("sample_from_anchor", [False, True])
def test_prepare_dflash_inputs_npu(lengths, sample_from_anchor):
    launch, tensors, expected = _prepare_launch(
        lengths=lengths, max_num_tokens=4099, max_num_reqs=41, sample_from_anchor=sample_from_anchor
    )
    launch()
    torch.npu.synchronize()
    for name, array in expected.items():
        torch.testing.assert_close(tensors[name].cpu(), torch.from_numpy(array), rtol=0, atol=0, msg=name)


@pytest.mark.parametrize("max_num_tokens", [4096, 16384])
def test_prepare_dflash_inputs_npu_decode_padding(max_num_tokens):
    launch, tensors, expected = _prepare_launch(lengths=(8,), max_num_tokens=max_num_tokens)
    launch()
    torch.npu.synchronize()
    for name, array in expected.items():
        torch.testing.assert_close(tensors[name].cpu(), torch.from_numpy(array), rtol=0, atol=0, msg=name)


def benchmark(iterations=100):
    """Print median end-to-end preparation time; compare same device/settings."""
    if not torch.npu.is_available():
        raise RuntimeError("The microbenchmark requires an Ascend NPU")
    for tokens in (4096, 16384):
        for batch in (1, 32):
            for phase in ("decode", "prefill"):
                lengths = (8,) * batch if phase == "decode" else (tokens // batch,) * batch
                launch, tensors, expected = _prepare_launch(lengths=lengths, max_num_tokens=tokens)
                launch()
                torch.npu.synchronize()
                for name, array in expected.items():
                    torch.testing.assert_close(tensors[name].cpu(), torch.from_numpy(array), rtol=0, atol=0, msg=name)
                for _ in range(10):
                    launch()
                torch.npu.synchronize()
                elapsed_ms = []
                for _ in range(5):
                    start = time.perf_counter()
                    for _ in range(iterations):
                        launch()
                    torch.npu.synchronize()
                    elapsed_ms.append((time.perf_counter() - start) * 1000 / iterations)
                print(
                    json.dumps(
                        dict(
                            batch=batch,
                            max_num_tokens=tokens,
                            phase=phase,
                            median_ms=statistics.median(elapsed_ms),
                            iterations=iterations,
                        )
                    )
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", action="store_true", help="Run synchronized, warmed-up NPU microbenchmarks")
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()
    if not args.benchmark or args.iterations < 1:
        parser.error("Pass --benchmark and a positive --iterations value")
    benchmark(args.iterations)
