# Investigating Qwen4Exp MTP graph acceptance

This investigation targets vLLM `a98247a` and vLLM Ascend PR #17249 at
`4e6f56c`. Keep the checkpoint, quantization, prompts, sampling parameters,
parallelism and scheduling settings identical when comparing modes.

## Correctness before performance

With greedy sampling, a sustained acceptance gap between eager drafting and
graph drafting needs a correctness investigation. A low acceptance length
increases target verification work per generated token. Optimizing collectives
or individual kernels cannot recover proposals that the target rejects.

The following source checks distinguish confirmed defects from hypotheses:

| PR commit | Finding |
| --- | --- |
| `ef64b9f` (six-region V2 cache allocation) | Official TP8 geometry at block size 768 gives both QSA K and GDN SSM pages 393216 bytes. Sharing the r2 slab is consistent with distinct physical block ownership at this geometry. It does not establish correctness for a modified checkpoint or block size. |
| `b710680` (QSA graph update hook) | The hook is a no-op. QSA metadata must therefore be regenerated into the addresses read by the captured kernels. Testing only the original capture inputs cannot validate this contract. |
| `4380bab` (V2 MTP integration) | The hidden-state path distinguishes the sampling stream from HC feedback. Step 0 selects sparse indices, compaction follows it, and later steps reuse them. Source inspection did not prove an ordering or feedback-stream error. |
| `4e6f56c` (padded QSA boundaries) | Clipping the CPU boundaries prevents the reported shape failure, but does not by itself invalidate device-side dummy rows during replay or clear the persistent metadata tail. This is a reproducible cache-write defect. |

A separate allocation edge case remains outside this replay fix: r2 uses the
maximum of the QSA K and SSM page widths, while `make_contiguous_slab_view`
uses each owner's dense block stride. With the official TP8 SSM geometry but
block size 1536, QSA K occupies 786432 bytes per block and SSM still strides
393216 bytes. Distinct physical IDs can then overlap. Do not infer that this
explains a run using block size 768; verify the resolved block size and local
checkpoint configuration before attributing symptoms to allocation.

For example, a graph captured for four draft-decode rows may replay with GPU
boundaries `[0, 1, 2, 2, 2]` and lengths `[101, 201, 0, 0]`. The captured CPU
mapped-token count remains four. The original metadata path can assign the
last two rows to request 3, produce logical positions `0, 1`, and write raw
ring slots `176, 177` if that request's block-table row still contains block
22 and the ring size is 8. Those rows must have slot `-1`.

The fix validates row ownership and logical positions using device boundaries
and lengths, writes invalid rows as request 0 / position -1 / visibility 0 /
slot -1, and clears the padded persistent tail. Outputs remain in each
builder's own buffers. Raw circular groups intentionally have disabled common
slot mappings on Ascend, so using their common slot `-1` as a blanket mask
would incorrectly disable valid raw-cache writes.

This defect covers padded buckets and shrinking batches. It is **not proof of
the cause of every acceptance drop at a steady, exactly captured batch size**.
Qwen3.6 MTP working well does not validate Qwen4Exp's additional QSA side-cache
and sparse-index replay contracts.

## Isolate the failing draft phase

Run from this checkout on the NPU server. The example engine arguments retain
TP8/EP8, asynchronous scheduling and the original `[4,8,...,32]` capture sizes.
The tool starts a separate offline engine for each selected mode; it does not
stop an existing serving process. Use available devices for these engines.

```bash
export VLLM_USE_V2_MODEL_RUNNER=1
python tools/validate_qwen4_exp_mtp.py \
  --model /mnt/share/weight/Qwen3.8-Flash-Next-w8a8 \
  --engine-args tools/qwen4_exp_mtp_engine_args.example.json \
  --gsm8k-jsonl /mnt/share/datasets/gsm8k/test.jsonl \
  --modes base,draft_eager,draft_graph,prefill_graph,decode_graph \
  --batch-sizes 8,8,3,8,2 --max-tokens 128 --repeat 2 \
  --output-dir mtp-phase-check --check
```

| Mode | Target graph | Draft step 0 graph | Later draft steps graph |
| --- | --- | --- | --- |
| `base` | enabled | no MTP | no MTP |
| `draft_eager` | enabled | disabled | disabled |
| `draft_graph` | enabled | enabled | enabled |
| `prefill_graph` | enabled | enabled | disabled |
| `decode_graph` | enabled | disabled | enabled |

Here "prefill" refers to MTP step 0, which also runs during ordinary generation.
It does not mean only the initial prompt prefill. Graph-enabled phases still
fall back to eager when their input shape is ineligible. Inspect
`graph_traces/<mode>/rank-*.jsonl` for actual capture, dispatch and replay events.
The trace is bounded and does not transfer device tensors to the CPU.

Read counter deltas in each mode's JSON report and the aggregate MAL in
`summary.json`. Inspect `accepted_per_position` as well: degradation at the
first position directs attention to step 0 or persistent state from previous
iterations; degradation only at later positions directs attention to decode,
feedback and sparse-index reuse. End-to-end output token agreement alone
cannot establish that draft tokens are correct, because the target rejects
incorrect proposals.

If the phase comparison implicates graph memory reuse, rerun the same modes
with `--separate-graph-pools` and a new output directory. This gives target,
draft step 0 and draft decode separate pools, while retaining pool sharing
between capture sizes within each manager. A recovery is evidence to inspect
temporary tensor lifetimes; the option is a diagnostic, not a validated
performance recommendation. `--disable-index-sharing` independently tests
the existing sparse-index sharing option. Change one diagnostic at a time.

The tool uses zero-shot GSM8K questions, varying output limits to exercise
batch shrinkage. Its elapsed throughput includes prefill, and graph tracing
adds host overhead. It is **not an AISBench TPOT measurement**. After acceptance
and replay equivalence recover, rerun the original GSM8K performance script
with diagnostics disabled to measure performance.

## Operator and metadata regressions

```bash
pytest -q tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_qsa_metadata.py
pytest -q tests/e2e/singlecard/test_qwen4_exp_qsa_graph_equivalence.py
```

The metadata tests change GPU query boundaries after capture and retain stale
positive block-table rows. The LI/GBSA tests compare eager and graph execution
from identical cache contents, update queries, positions, lengths, block
tables and sparse indices after capture, and interleave graph buckets in one
pool. They use 768-token storage pages and contexts longer than 2048 tokens so
the indexer must select a subset of the visible groups.

CPU metadata and diagnostic-tool tests are available separately. Passing them
does not validate NPU compilation, LI/GBSA replay, distributed execution,
model acceptance or TPOT. Those checks remain required on the affected server.

## Validation record

On the development host, 25 isolated CPU tests passed for metadata sanitation
and diagnostic configuration/dispatch. Ruff, syntax, Markdown, JSON/YAML and
the applicable repository Python checks passed. The five-mode command also
completed its configuration-only `--dry-run`.

The required `bash format.sh ci` was attempted but stopped while installing
actionlint's Go dependencies because `proxy.golang.org` timed out. The full
format check is incomplete. No NPU was available: the eight NPU regression
cases, TP8 model acceptance and GSM8K performance have not been executed.
