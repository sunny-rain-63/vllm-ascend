# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks of production DFlash metadata lifetime and FULL replay dispatch.

Only the Ascend class definitions are loaded, with a small upstream stand-in,
so these checks run without Torch, vLLM, or an NPU. They exercise production
methods, not attention numerics or graph execution on device.
"""

import ast
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace


def _load_class(relative_path, class_name, namespace):
    source = Path(__file__).resolve().parents[3] / relative_path
    tree = ast.parse(source.read_text(encoding="utf-8"))
    node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, node], type_ignores=[]))
    exec(compile(module, str(source), "exec"), namespace)
    return namespace[class_name]


class _UpstreamSpeculator:
    def __init__(self, *args):
        self.build_calls = []
        self.fail_build = False
        self.return_no_metadata = False

    def _build_draft_attn_metadata(self, num_reqs, num_reqs_padded, num_tokens_padded, *args, **kwargs):
        if self.fail_build:
            raise RuntimeError("metadata build failed")
        self.build_calls.append((num_reqs, num_reqs_padded, num_tokens_padded, args, kwargs))
        if self.return_no_metadata:
            return None
        self.last_built = {
            "swa": SimpleNamespace(causal=True, sliding_window=2048, block_tables=object()),
            "full": SimpleNamespace(causal=False, sliding_window=None, block_tables=object()),
        }
        for metadata in self.last_built.values():
            metadata.actual_seq_lengths_q = [min(index + 1, num_reqs) * 8 for index in range(num_reqs_padded)]
        return self.last_built

    def propose(self, *args, **kwargs):
        return self.scenario(self)


def _make_speculator():
    cls = _load_class(
        "vllm_ascend/worker/v2/spec_decode/dflash/speculator.py",
        "AscendDFlashSpeculator",
        {
            "DFlashSpeculator": _UpstreamSpeculator,
            "build_attn_metadata_wrapper": nullcontext,
            "vllm_version_is": lambda version: False,
        },
    )
    speculator = cls(None, None)
    speculator.input_batch = SimpleNamespace(num_reqs=1, seq_lens_cpu_upper_bound=object())
    speculator.num_query_per_req = 8
    speculator._group_causal = {2: True, 3: False}
    return speculator


def _propose(speculator, **kwargs):
    return speculator.propose(
        speculator.input_batch,
        {},
        {},
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        **kwargs,
    )


def _build(speculator, num_reqs=1, num_reqs_padded=2, num_tokens_padded=16, **kwargs):
    return speculator._build_draft_attn_metadata(
        num_reqs,
        num_reqs_padded,
        num_tokens_padded,
        seq_lens_cpu_upper_bound=speculator.input_batch.seq_lens_cpu_upper_bound,
        step=8,
        causal=speculator._group_causal,
        **kwargs,
    )


def _replay(speculator):
    return speculator.get_draft_attn_metadatas_for_replay(2, 16, speculator.input_batch.seq_lens_cpu_upper_bound)[0]


class TestDFlashMetadataReuse(unittest.TestCase):
    def assert_cleared(self, speculator):
        self.assertFalse(speculator._reuse_draft_metadata_within_propose)
        self.assertIsNone(speculator._current_propose_draft_metadata)

    def test_same_propose_reuses_all_groups_and_corrects_padded_queries(self):
        speculator = _make_speculator()

        def scenario(model):
            first = _build(model)
            tables = [metadata.block_tables for metadata in first.values()]
            self.assertEqual(first["swa"].actual_seq_lengths_q, [8, 8])
            replay = _replay(model)
            self.assertIs(replay, first)
            self.assertEqual(len(model.build_calls), 1)
            self.assertEqual([metadata.block_tables for metadata in replay.values()], tables)
            self.assertEqual([metadata.causal for metadata in replay.values()], [True, False])
            self.assertEqual([metadata.sliding_window for metadata in replay.values()], [2048, None])
            for metadata in replay.values():
                self.assertEqual(metadata.actual_seq_lengths_q, [8, 16])
            return replay

        speculator.scenario = scenario
        _propose(speculator)
        self.assert_cleared(speculator)

    def test_eager_result_cannot_leak_into_next_propose(self):
        speculator = _make_speculator()
        speculator.scenario = _build
        first = _propose(speculator)
        self.assertEqual(first["swa"].actual_seq_lengths_q, [8, 8])
        self.assert_cleared(speculator)
        speculator.scenario = lambda model: (_build(model), _replay(model))[1]
        second = _propose(speculator)
        self.assertIsNot(second, first)
        self.assertEqual(len(speculator.build_calls), 2)
        self.assert_cleared(speculator)

    def test_exception_clears_metadata(self):
        speculator = _make_speculator()

        def scenario(model):
            _build(model)
            raise RuntimeError("draft failed")

        speculator.scenario = scenario
        with self.assertRaisesRegex(RuntimeError, "draft failed"):
            _propose(speculator)
        self.assert_cleared(speculator)

    def test_failed_or_empty_build_discards_previous_result(self):
        for failure in (True, False):
            with self.subTest(failure=failure):
                speculator = _make_speculator()
                speculator._reuse_draft_metadata_within_propose = True
                _build(speculator)
                speculator.fail_build = failure
                speculator.return_no_metadata = not failure
                if failure:
                    with self.assertRaisesRegex(RuntimeError, "metadata build failed"):
                        _build(speculator)
                else:
                    self.assertIsNone(_build(speculator))
                self.assertIsNone(speculator._current_propose_draft_metadata)

    def test_dummy_and_profile_keep_rebuild_path(self):
        for options in ({"dummy_run": True}, {"is_profile": True}):
            with self.subTest(options=options):
                speculator = _make_speculator()

                def scenario(model):
                    first = _build(model)
                    self.assertIsNone(model._current_propose_draft_metadata)
                    replay = _replay(model)
                    self.assertIsNot(replay, first)
                    return replay

                speculator.scenario = scenario
                _propose(speculator, **options)
                self.assertEqual(len(speculator.build_calls), 2)
                self.assert_cleared(speculator)

    def test_capture_outside_propose_does_not_reuse(self):
        speculator = _make_speculator()
        first = _build(speculator)
        self.assertIsNot(_replay(speculator), first)
        self.assertEqual(len(speculator.build_calls), 2)
        self.assert_cleared(speculator)

    def test_descriptor_or_request_count_mismatch_rebuilds(self):
        for counts in ((2, 2, 16), (1, 1, 8), (1, 2, 32)):
            with self.subTest(counts=counts):
                speculator = _make_speculator()

                def scenario(model, counts=counts):
                    first = _build(model, *counts)
                    replay = _replay(model)
                    self.assertIsNot(replay, first)
                    self.assertEqual(model.build_calls[-1][:3], (1, 2, 16))

                speculator.scenario = scenario
                _propose(speculator)
                self.assertEqual(len(speculator.build_calls), 2)
                self.assert_cleared(speculator)

    def test_reuse_is_consumed_once(self):
        speculator = _make_speculator()

        def scenario(model):
            first = _build(model)
            self.assertIs(_replay(model), first)
            self.assertIsNot(_replay(model), first)
            self.assertIsNone(model._current_propose_draft_metadata)

        speculator.scenario = scenario
        _propose(speculator)
        self.assertEqual(len(speculator.build_calls), 2)

    def test_new_upstream_metadata_kwargs_are_forwarded(self):
        speculator = _make_speculator()
        local_lengths = object()
        _build(speculator, dcp_local_seq_lens=local_lengths)
        self.assertIs(speculator.build_calls[0][4]["dcp_local_seq_lens"], local_lengths)
        self.assertIs(speculator.build_calls[0][4]["causal"], speculator._group_causal)

    def test_full_graph_replay_uses_upstream_build_once(self):
        speculator = _make_speculator()
        updates = []
        replayed = []
        result = object()

        class UpstreamGraph:
            def run_fullgraph(self, desc):
                replayed.append(desc)
                return result

        graph_cls = _load_class(
            "vllm_ascend/worker/v2/spec_decode/dflash/aclgraph.py",
            "DFlashAclGraphManager",
            {
                "DFlashCudaGraphManager": UpstreamGraph,
                "torch": SimpleNamespace(npu=SimpleNamespace(current_stream=lambda: None), full=lambda *args: None),
                "set_forward_context": lambda *args, **kwargs: nullcontext(),
                "get_forward_context": lambda: None,
                "_EXTRA_CTX": SimpleNamespace(),
                "update_full_graph_params": lambda *args, **kwargs: updates.append(kwargs),
            },
        )
        manager = object.__new__(graph_cls)
        manager.speculator = speculator
        manager.update_stream = SimpleNamespace(wait_stream=lambda stream: None)
        manager.vllm_config = None
        speculator.dp_size = 1
        speculator.model_state = SimpleNamespace(attn_metadata={})
        speculator.attn_backends = {"swa": object(), "full": object()}
        speculator.speculative_config = None
        descriptor = SimpleNamespace(num_reqs=2, num_tokens=16, cg_mode="FULL")

        def scenario(model):
            metadata = _build(model)
            self.assertIs(manager.run_fullgraph(descriptor), result)
            self.assertIs(updates[0]["draft_attn_metadatas"][0], metadata)

        speculator.scenario = scenario
        _propose(speculator)
        self.assertEqual(replayed, [descriptor])
        self.assertEqual(len(speculator.build_calls), 1)
        self.assert_cleared(speculator)


if __name__ == "__main__":
    unittest.main()
