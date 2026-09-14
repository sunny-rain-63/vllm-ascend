# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU emulation of the production slot-guard kernel, without NPU imports.

This tests integer/masking semantics, not Triton compilation or device runtime.
The accompanying single-card test covers real eager execution and graph replay.
"""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


class _Pointer:
    def __init__(self, values, offsets=0):
        self.values = values
        self.offsets = offsets

    def __add__(self, offsets):
        return _Pointer(self.values, self.offsets + offsets)


class _Language:
    constexpr = int
    where = staticmethod(np.where)
    arange = staticmethod(np.arange)

    def __init__(self):
        self.pid = 0

    def program_id(self, _axis):
        return self.pid

    @staticmethod
    def load(pointer, mask, other):
        values = np.full(mask.shape, other, dtype=pointer.values.dtype)
        values[mask] = pointer.values[pointer.offsets[mask]]
        return values

    @staticmethod
    def store(pointer, values, mask):
        pointer.values[pointer.offsets[mask]] = values[mask]


def _load_functions():
    source = Path(__file__).resolve().parents[3] / "vllm_ascend/ops/triton/v2/block_table/mask_dflash_slots.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    for function in functions:
        function.decorator_list = []
    language = _Language()
    namespace = {
        "tl": language,
        "torch": SimpleNamespace(Tensor=object, int32=np.int32, int64=np.int64, where=np.where),
    }
    body: list[ast.stmt] = list(functions)
    exec(compile(ast.Module(body=body, type_ignores=[]), str(source), "exec"), namespace)
    return namespace, language


class TestDFlashSlotMapping(unittest.TestCase):
    def test_kernel_exact_integer_boundaries_and_tails(self):
        namespace, language = _load_functions()
        kernel = namespace["_mask_dflash_slots_kernel"]
        null_block_size = 1536
        slot_limit = 5461 * null_block_size
        for dtype in (np.int32, np.int64):
            boundary = np.array(
                [-1, 0, 127, 128, 1535, 1536, slot_limit - 1, slot_limit, np.iinfo(dtype).min, np.iinfo(dtype).max],
                dtype=dtype,
            )
            for length in (1, 255, 256, 257, 4096, 16384):
                with self.subTest(dtype=dtype, length=length):
                    slots: np.ndarray = np.resize(boundary, length)
                    before = slots.copy()
                    output = np.full_like(slots, 99)
                    for language.pid in range((length + 255) // 256):
                        kernel(_Pointer(slots), _Pointer(output), length, null_block_size, slot_limit, -1, 256)
                    expected = np.where((slots >= null_block_size) & (slots < slot_limit), slots, -1)
                    np.testing.assert_array_equal(output, expected)
                    np.testing.assert_array_equal(slots, before)

    def test_wrapper_preserves_cpu_and_noncontiguous_fallback(self):
        namespace, _ = _load_functions()

        class Slots(np.ndarray):
            device_type: str

            @property
            def device(self):
                return SimpleNamespace(type=self.device_type)

            def is_contiguous(self):
                return self.flags.c_contiguous

        values = np.array([-1, 0, 1535, 1536, 2000, 3071, 3072, 4000], dtype=np.int64)
        for device, slots in (("cpu", values), ("npu", values[::2]), ("npu", values.reshape(2, 4))):
            with self.subTest(device=device, shape=slots.shape):
                slots = slots.view(Slots)
                slots.device_type = device
                before = slots.copy()
                output = namespace["mask_dflash_slots"](slots, 1536, 3072, -1)
                np.testing.assert_array_equal(output, np.where((slots >= 1536) & (slots < 3072), slots, -1))
                np.testing.assert_array_equal(slots, before)


if __name__ == "__main__":
    unittest.main()
