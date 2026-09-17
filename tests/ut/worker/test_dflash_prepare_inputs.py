# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU instruction-level checks for both DFlash kernel compatibility variants.

The production bodies execute with a small NumPy implementation of their Triton
operations. This checks indexing and masked tails, not Triton compilation or NPU
performance; the companion nightly test runs the real compiled kernel.
"""

import ast
import unittest
from pathlib import Path

import numpy as np


class _Value(np.ndarray):
    def to(self, dtype):
        return self.astype(dtype)


class _Pointer:
    def __init__(self, array, offset=0):
        self.array = array
        self.offset = offset

    def __add__(self, offset):
        return _Pointer(self.array, self.offset + offset)

    def __sub__(self, offset):
        return self + (-offset)


class _Language:
    int32 = np.int32
    int64 = np.int64
    minimum = staticmethod(np.minimum)

    def __init__(self, num_reqs):
        self.num_reqs = num_reqs
        self.program = (0, 0)
        self.store_calls = 0

    def program_id(self, axis):
        return self.program[axis]

    def num_programs(self, axis):
        assert axis == 0
        return self.num_reqs

    @staticmethod
    def arange(start, end):
        return np.arange(start, end, dtype=np.int32).view(_Value)

    @staticmethod
    def _indices(pointer, mask):
        indices, active = np.broadcast_arrays(pointer.offset, mask)
        selected = indices[active.astype(bool)]
        assert np.all(selected >= 0) and np.all(selected < pointer.array.size), "Unmasked out-of-bounds access"
        return indices, active.astype(bool)

    def load(self, pointer, mask=True, other=0):
        indices, active = self._indices(pointer, mask)
        result = np.full(indices.shape, other, dtype=pointer.array.dtype)
        result[active] = pointer.array[indices[active]]
        return result.view(_Value)

    def store(self, pointer, value, mask=True):
        indices, active = self._indices(pointer, mask)
        values = np.broadcast_to(value, indices.shape)
        pointer.array[indices[active]] = values[active]
        self.store_calls += 1


def _load_kernels(language):
    source = Path(__file__).resolve().parents[3] / "vllm_ascend/worker/v2/spec_decode/dflash/speculator.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    kernels = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name != "_prepare_dflash_inputs_kernel_ascend":
            continue
        node.decorator_list = []
        for argument in node.args.args:
            argument.annotation = None
        namespace = {"tl": language}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
        kernels.append((namespace[node.name], [argument.arg for argument in node.args.args]))
    assert len(kernels) == 2, "Test both v0.28 and main compatibility variants"
    return kernels


def make_case(lengths, max_num_tokens=4096, max_num_reqs=32, sample_from_anchor=False, position_base=0):
    """Shared deterministic fixture for CPU parity, NPU parity, and benchmarks."""
    num_reqs = len(lengths)
    num_query = 8
    num_steps = num_query if sample_from_anchor else num_query - 1
    assert num_reqs <= max_num_reqs
    assert max(sum(lengths), num_reqs * num_query) <= max_num_tokens
    max_model_len = 122880
    block_size = 128
    stride = max_model_len // block_size
    query_starts = np.concatenate(([0], np.cumsum(lengths))).astype(np.int32)
    positions = np.concatenate(
        [np.arange(length, dtype=np.int64) + position_base + i * 127 for i, length in enumerate(lengths)]
    )
    arrays = {
        "out_input_ids_ptr": np.full(max_num_tokens, 777, dtype=np.int32),
        "out_query_positions_ptr": np.full(max_num_tokens, 777, dtype=np.int64),
        "out_query_start_loc_ptr": np.full(max_num_reqs + 1, 777, dtype=np.int32),
        "out_seq_lens_ptr": np.full(max_num_reqs, 777, dtype=np.int32),
        "out_query_slot_mapping_ptr": np.full(max_num_tokens, 777, dtype=np.int64),
        "out_context_positions_ptr": np.full(max_num_tokens, 777, dtype=np.int64),
        "out_context_slot_mapping_ptr": np.full(max_num_tokens, 777, dtype=np.int32),
        "out_sample_indices_ptr": np.full(max_num_reqs * num_steps, 777, dtype=np.int32),
        "out_sample_pos_ptr": np.full(max_num_reqs * num_steps, 777, dtype=np.int64),
        "out_sample_idx_mapping_ptr": np.full(max_num_reqs * num_steps, 777, dtype=np.int32),
        "out_temperature_ptr": np.full(max_num_reqs, -99, dtype=np.float32),
        "out_seeds_ptr": np.full(max_num_reqs, 777, dtype=np.int64),
        "target_positions_ptr": positions,
        "target_query_start_loc_ptr": query_starts,
        "idx_mapping_ptr": np.arange(max_num_reqs - 1, max_num_reqs - num_reqs - 1, -1, dtype=np.int32),
        "last_sampled_ptr": np.arange(max_num_reqs, dtype=np.int32) + 100,
        "next_prefill_tokens_ptr": np.arange(max_num_reqs, dtype=np.int32) + 200,
        "num_sampled_ptr": np.arange(num_reqs, dtype=np.int32) % 2,
        "num_rejected_ptr": np.array([min(i % 3, length - 1) for i, length in enumerate(lengths)], dtype=np.int32),
        "temperature_ptr": np.linspace(0, 1, max_num_reqs, dtype=np.float32),
        "seeds_ptr": np.arange(max_num_reqs, dtype=np.int64) + (1 << 40),
        "block_table_ptr": (np.arange(num_reqs * stride, dtype=np.int32) * 13 % 5400 + 1),
    }
    max_tokens_per_req = max(lengths) + num_query
    tile = min(256, 1 << (max_tokens_per_req - 1).bit_length())
    scalars = dict(
        block_table_stride=stride,
        parallel_drafting_token_id=248070,
        block_size=block_size,
        num_query_per_req=num_query,
        num_speculative_steps=num_steps,
        max_num_reqs=max_num_reqs,
        max_num_tokens=max_num_tokens,
        max_model_len=max_model_len,
        cp_rank=0,
        SAMPLE_FROM_ANCHOR=sample_from_anchor,
        PAD_SLOT_ID=-1,
        CP_SIZE=1,
        CP_INTERLEAVE=1,
        BLOCK_SIZE=tile,
    )
    return arrays, scalars, (num_reqs, (max_tokens_per_req + tile - 1) // tile)


def reference_outputs(arrays, scalars, grid):
    """Independent scalar oracle, including untouched storage and graph tails."""
    output = {name: array.copy() for name, array in arrays.items() if name.startswith("out_")}
    num_reqs = grid[0]
    query_count = scalars["num_query_per_req"]
    steps = scalars["num_speculative_steps"]
    block_size = scalars["block_size"]
    stride = scalars["block_table_stride"]
    sample_offset = 0 if scalars["SAMPLE_FROM_ANCHOR"] else 1
    table = arrays["block_table_ptr"].reshape(num_reqs, stride)

    def slot(req, position):
        return int(table[req, min(position // block_size, stride - 1)]) * block_size + position % block_size

    for req in range(num_reqs):
        start, end = arrays["target_query_start_loc_ptr"][req : req + 2]
        state = arrays["idx_mapping_ptr"][req]
        valid_end = end - arrays["num_rejected_ptr"][req]
        last_position = int(arrays["target_positions_ptr"][valid_end - 1])
        for index in range(start, end):
            position = int(arrays["target_positions_ptr"][index])
            output["out_context_positions_ptr"][index] = position
            output["out_context_slot_mapping_ptr"][index] = slot(req, position)
        bonus_source = "last_sampled_ptr" if arrays["num_sampled_ptr"][req] > 0 else "next_prefill_tokens_ptr"
        for offset in range(query_count):
            index = req * query_count + offset
            position = last_position + 1 + offset
            output["out_input_ids_ptr"][index] = arrays[bonus_source][state] if offset == 0 else 248070
            output["out_query_positions_ptr"][index] = min(position, scalars["max_model_len"] - 1)
            output["out_query_slot_mapping_ptr"][index] = slot(req, position)
            if offset >= sample_offset:
                sample_index = req * steps + offset - sample_offset
                output["out_sample_indices_ptr"][sample_index] = index
                output["out_sample_pos_ptr"][sample_index] = position + (sample_offset == 0)
                output["out_sample_idx_mapping_ptr"][sample_index] = state
        output["out_query_start_loc_ptr"][req] = req * query_count
        output["out_seq_lens_ptr"][req] = last_position + 1 + query_count
        output["out_temperature_ptr"][state] = arrays["temperature_ptr"][state]
        output["out_seeds_ptr"][state] = arrays["seeds_ptr"][state]
    output["out_query_start_loc_ptr"][num_reqs:] = num_reqs * query_count
    output["out_seq_lens_ptr"][num_reqs:] = 0
    output["out_sample_indices_ptr"][num_reqs * steps :] = 0
    output["out_sample_pos_ptr"][num_reqs * steps :] = 0
    output["out_sample_idx_mapping_ptr"][num_reqs * steps :] = -1
    output["out_query_slot_mapping_ptr"][num_reqs * query_count :] = scalars["PAD_SLOT_ID"]
    return output


class TestDFlashPrepareInputs(unittest.TestCase):
    def _check_case(self, **case_kwargs):
        arrays, scalars, grid = make_case(**case_kwargs)
        expected = reference_outputs(arrays, scalars, grid)
        for variant in range(2):
            with self.subTest(variant=variant, **case_kwargs):
                actual = {name: array.copy() for name, array in arrays.items()}
                language = _Language(grid[0])
                kernel, names = _load_kernels(language)[variant]
                arguments = {name: _Pointer(array) for name, array in actual.items()} | scalars
                for req in reversed(range(grid[0])):
                    for block in reversed(range(grid[1])):
                        language.program = (req, block)
                        kernel(**{name: arguments[name] for name in names})
                for name, array in actual.items():
                    np.testing.assert_array_equal(array, expected.get(name, arrays[name]), err_msg=name)
                # Decode with 16K graph slots used to issue >16K scalar stores.
                if case_kwargs.get("lengths") == (8,) and case_kwargs.get("max_num_tokens") == 16384:
                    self.assertLess(language.store_calls, 1200)

    def test_decode_and_graph_padding(self):
        for lengths in ((8,), (8,) * 32):
            for tokens in (4096, 16384):
                self._check_case(lengths=lengths, max_num_tokens=tokens)

    def test_ragged_context_and_partial_tiles(self):
        self._check_case(lengths=(1, 15, 16, 17, 255, 256, 257), max_num_tokens=4099, max_num_reqs=41)

    def test_chunked_prefill_and_anchor_sampling(self):
        for anchor in (False, True):
            self._check_case(lengths=(4095, 9, 7), max_num_tokens=16384, sample_from_anchor=anchor)

    def test_position_and_block_table_clamping(self):
        self._check_case(lengths=(8,), max_num_tokens=8, max_num_reqs=1, position_base=122878)


if __name__ == "__main__":
    unittest.main()
