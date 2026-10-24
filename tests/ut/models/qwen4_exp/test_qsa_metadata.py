# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

"""CPU checks of captured QSA metadata with changing device boundaries."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def metadata_helpers():
    # Load just the helpers so these CPU tests need neither Triton nor an NPU.
    path = Path(__file__).resolve().parents[4] / "vllm_ascend/models/qwen4_exp/qsa_metadata.py"
    module = ast.parse(path.read_text(encoding="utf-8"))
    functions = [
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name != "_sanitize_qsa_metadata_kernel"
    ]
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def common_metadata(query_start_loc, seq_lens, actual, padded):
    return SimpleNamespace(
        query_start_loc=torch.tensor(query_start_loc, dtype=torch.int32),
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32),
        num_actual_tokens=actual,
        num_input_tokens=padded,
    )


def sanitize(helpers, common, mappings, positions, slots):
    token_to_req = torch.tensor(mappings, dtype=torch.int32)
    logical_positions = torch.tensor(positions, dtype=torch.int64)
    visible_blocks = torch.full_like(token_to_req, 20)
    slot_mapping = torch.tensor(slots, dtype=torch.int64)
    helpers["_sanitize_qsa_metadata_torch"](
        common, token_to_req, token_to_req, logical_positions, visible_blocks, slot_mapping, slot_mapping
    )
    return token_to_req, logical_positions, visible_blocks, slot_mapping


def test_fixed_capture_count_does_not_allow_dummy_raw_ring_writes(metadata_helpers):
    # The graph captured four mapped rows. Runtime GPU boundaries now own two.
    # The old mapper clamps dummy rows to req 3, whose stale block is still 22.
    common = common_metadata([0, 1, 2, 2, 2], [101, 201, 0, 0], actual=4, padded=4)
    rows = torch.arange(4)
    requests = torch.searchsorted(common.query_start_loc[1:], rows, right=True).clamp(max=3)
    positions = common.seq_lens[requests] - torch.diff(common.query_start_loc)[requests]
    positions += rows - common.query_start_loc[requests]
    blocks = torch.tensor([11, 12, 21, 22])
    old_slots = blocks[requests] * 8 + positions.remainder(8)
    assert old_slots.tolist() == [92, 96, 176, 177]
    mapping, logical, visible, slots = sanitize(
        metadata_helpers, common, requests.tolist(), positions.tolist(), old_slots.tolist()
    )
    assert mapping.tolist() == [0, 1, 0, 0]
    assert logical.tolist() == [100, 200, -1, -1]
    assert visible.tolist() == [20, 20, 0, 0]
    assert slots.tolist() == [92, 96, -1, -1]


def test_dummy_full_width_queries_with_zero_lengths_are_invalid(metadata_helpers):
    common = common_metadata([0, 1, 2, 3, 4], [101, 201, 0, 0], actual=4, padded=4)
    _, logical, visible, slots = sanitize(metadata_helpers, common, [0, 1, 2, 3], [100, 200, 0, 0], [92, 96, 168, 176])
    assert logical.tolist() == [100, 200, -1, -1]
    assert visible.tolist() == [20, 20, 0, 0]
    assert slots.tolist() == [92, 96, -1, -1]


def test_clears_persistent_tail_held_by_a_larger_graph(metadata_helpers):
    common = common_metadata([0, 1, 2], [101, 201], actual=2, padded=4)
    _, logical, visible, slots = sanitize(
        metadata_helpers, common, [0, 1, 1, 1], [100, 200, 201, 202], [92, 96, 97, 98]
    )
    assert logical.tolist() == [100, 200, -1, -1]
    assert visible.tolist() == [20, 20, 0, 0]
    assert slots.tolist() == [92, 96, -1, -1]


def test_preserves_legitimate_ring_window_and_does_not_use_common_slot_mask(metadata_helpers):
    common = common_metadata([0, 12], [12], actual=12, padded=12)
    common.slot_mapping = torch.full((12,), -1, dtype=torch.int64)
    ring_slots = [-1] * 4 + [11 * 8 + position % 8 for position in range(4, 12)]
    _, logical, _, slots = sanitize(metadata_helpers, common, [0] * 12, list(range(12)), ring_slots)
    assert logical.tolist() == list(range(12))
    assert slots.tolist() == ring_slots


def test_cached_token_map_is_copied_to_each_builders_stable_owned_buffer(metadata_helpers):
    cached_view = torch.tensor([0, 1, 3, 3], dtype=torch.int32)
    common = common_metadata([0, 1, 2, 2, 2], [101, 201, 0, 0], actual=4, padded=6)
    output_buffers = []
    for _ in range(2):
        owned = torch.full((6,), 55, dtype=torch.int32)
        logical = torch.tensor([100, 200, 0, 1, 99, 99], dtype=torch.int64)
        visible = torch.full((6,), 20, dtype=torch.int32)
        slots = torch.tensor([92, 96, 176, 177, 99, 99], dtype=torch.int64)
        result = (cached_view, logical[:4], visible[:4], slots[:4])
        metadata_helpers["qsa_cache"] = SimpleNamespace(
            _build_qsa_metadata_torch=lambda *args, result=result, **kwargs: result
        )
        actual = metadata_helpers["build_qsa_metadata"](
            common, owned, logical, visible, slots, storage_block_size=8, compress_ratio=1, circular_buffer_size=8
        )
        assert actual[0].data_ptr() == owned.data_ptr()
        assert actual[0].data_ptr() != cached_view.data_ptr()
        assert owned.tolist() == [0, 1, 0, 0, 0, 0]
        assert slots.tolist() == [92, 96, -1, -1, -1, -1]
        output_buffers.append(owned)
    assert output_buffers[0].data_ptr() != output_buffers[1].data_ptr()
    assert cached_view.tolist() == [0, 1, 3, 3]


def test_short_cached_and_common_slot_views_are_read_only_actual_sized_sources(metadata_helpers):
    cached = torch.tensor([0, 1], dtype=torch.int32)
    common_slots = torch.tensor([92, 96], dtype=torch.int64)
    owned = torch.full((4,), 55, dtype=torch.int32)
    logical = torch.tensor([100, 200, 201, 202], dtype=torch.int64)
    visible = torch.full((4,), 20, dtype=torch.int32)
    slots = torch.full((4,), 99, dtype=torch.int64)
    common = common_metadata([0, 1, 2], [101, 201], actual=2, padded=4)
    result = (cached, logical[:2], visible[:2], common_slots)
    metadata_helpers["qsa_cache"] = SimpleNamespace(_build_qsa_metadata_torch=lambda *args, **kwargs: result)
    actual = metadata_helpers["build_qsa_metadata"](
        common, owned, logical, visible, slots, storage_block_size=8, compress_ratio=1
    )
    assert actual[0].data_ptr() == owned.data_ptr()
    assert actual[3].data_ptr() == slots.data_ptr()
    assert owned.tolist() == [0, 1, 0, 0]
    assert slots.tolist() == [92, 96, -1, -1]
    assert cached.tolist() == [0, 1]
    assert common_slots.tolist() == [92, 96]
    with pytest.raises(ValueError, match="padded model input"):
        metadata_helpers["build_qsa_metadata"](
            common, torch.empty(1), logical, visible, slots, storage_block_size=8, compress_ratio=1
        )


def test_invalid_request_indices_are_clamped_before_metadata_gathers(metadata_helpers):
    common = common_metadata([0, 1, 2], [101, 201], actual=2, padded=4)
    mapping, logical, visible, slots = sanitize(
        metadata_helpers, common, [-1, 9999, 9999, -1], [100, 200, 201, 202], [92, 96, 97, 98]
    )
    assert mapping.tolist() == [0, 0, 0, 0]
    assert logical.tolist() == [-1, -1, -1, -1]
    assert visible.tolist() == [0, 0, 0, 0]
    assert slots.tolist() == [-1, -1, -1, -1]


def test_builder_hook_clears_buffers_beyond_returned_actual_views(metadata_helpers):
    common = common_metadata([0, 1, 2], [101, 201], actual=2, padded=4)
    token_buffer = torch.tensor([0, 1, 9, 9], dtype=torch.int32)
    logical_buffer = torch.tensor([100, 200, 201, 202], dtype=torch.int64)
    visible_buffer = torch.full((4,), 20, dtype=torch.int32)
    slot_buffer = torch.tensor([92, 96, 97, 98], dtype=torch.int64)
    result = (token_buffer[:2], logical_buffer[:2], visible_buffer[:2], slot_buffer[:2])
    metadata_helpers["qsa_cache"] = SimpleNamespace(_build_qsa_metadata_torch=lambda *args, **kwargs: result)
    actual = metadata_helpers["build_qsa_metadata"](
        common,
        token_buffer,
        logical_buffer,
        visible_buffer,
        slot_buffer,
        storage_block_size=8,
        compress_ratio=1,
        circular_buffer_size=8,
    )
    assert all(tensor.numel() == 2 for tensor in actual)
    assert token_buffer.tolist() == [0, 1, 0, 0]
    assert logical_buffer.tolist() == [100, 200, -1, -1]
    assert visible_buffer.tolist() == [20, 20, 0, 0]
    assert slot_buffer.tolist() == [92, 96, -1, -1]


def test_empty_actual_batch_clears_nonempty_captured_input_tail(metadata_helpers):
    common = common_metadata([0, 0], [0], actual=0, padded=4)
    token_buffer = torch.full((4,), 9, dtype=torch.int32)
    logical_buffer = torch.full((4,), 99, dtype=torch.int64)
    visible_buffer = torch.full((4,), 20, dtype=torch.int32)
    slot_buffer = torch.full((4,), 99, dtype=torch.int64)
    result = (token_buffer[:0], logical_buffer[:0], visible_buffer[:0], slot_buffer[:0])
    metadata_helpers["qsa_cache"] = SimpleNamespace(_build_qsa_metadata_torch=lambda *args, **kwargs: result)
    actual = metadata_helpers["build_qsa_metadata"](
        common, token_buffer, logical_buffer, visible_buffer, slot_buffer, storage_block_size=8, compress_ratio=1
    )
    assert all(tensor.numel() == 0 for tensor in actual)
    assert token_buffer.tolist() == [0] * 4
    assert logical_buffer.tolist() == [-1] * 4
    assert visible_buffer.tolist() == [0] * 4
    assert slot_buffer.tolist() == [-1] * 4
