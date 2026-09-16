# Mixed-window DFlash: correctness root cause and performance analysis

Audience: maintainers of the Ascend V2 DFlash path for Qwen3.6 (GDN/Mamba
target + DFlash drafter with mixed `full_attention` / `sliding_attention`
layers). This document explains

1. why the mixed configuration produced garbled output and collapsing draft
   acceptance on current vLLM, while the same configuration was healthy on
   vLLM 0.23.0;
2. what PR #16490 changes and why that repair is structurally required;
3. why, after the repair, the mixed-window drafter can be *slower* end to end
   than an all-full drafter even though its draft acceptance length is higher;
4. how to confirm the dominant term on hardware (this branch ships an
   env-gated phase timer), and the recommended fixes in priority order.

## 1. Why the mixed configuration corrupted on main (and not on 0.23.0)

### 1.1 The aliasing mechanism

On Ascend, the V2 hybrid cache materializes each layer's state as contiguous
planes inside one shared backing, not as strided pages:

- Mamba/GDN layers: `[all conv | all SSM | padding]` per physical block.
- Attention layers: `[padding | all K | all V]` per physical block.

With conv page bytes `C`, SSM plane bytes `S`, the common padded page is
`P = C + 2S`. Equal `page_size_padded` alone does **not** keep layers
disjoint: the K plane of an attention layer only fills
`block_size * kv_row_bytes` per physical block. When the Full and SWA groups
advertise different row geometries (the drafter's SWA layers have far fewer KV
bytes per token than the target's Full layers), a 128-token SWA block uses
`128 * 1024 = 131072` bytes of its K/V planes while the Full layers treat the
same plane as `S = 1572864` bytes per block. Distinct physical block IDs then
alias: in the reported TP2 geometry, SWA block 1's K/V writes land inside
full-attention V blocks 10/11. Verification reads corrupted K/V, the target
rejects nearly every draft token, acceptance length degrades towards 1, and
the generated text degenerates into garbage.

Current vLLM makes this possible because the hybrid KV cache manager places
several cache groups into **one shared backing with one common `num_blocks`**
(`UniformTypeKVCacheSpecs`, vLLM #51718: descriptors are views of a single
hybrid pool). Plane offsets are derived per spec from the same base address,
so mismatched per-block plane sizes overlap across groups.

### 1.2 Why vLLM 0.23.0 was healthy

On 0.23.0 the KV cache allocator did not share one backing across the Full
and SWA groups. Each group's cache was carved out of its own allocation with
its own geometry, so a SWA block and a Full block with different plane sizes
could never occupy the same bytes. No realignment was necessary, block size
stayed 128 everywhere, and the FIA addressing boundary (below) was never
approached because kernel block counts stayed far under `2**16`.

In other words: 0.23.0 was correct *by accident of layout* — the upgrade to
the shared hybrid backing introduced the hazard, and it only fires when the
drafter mixes Full and SWA layers (identical specs never alias, which is why
all-full and all-sliding drafters were unaffected).

## 2. What PR #16490 changes, and why the shape is forced

The hybrid manager requires **one physical block count shared by every
group**, so the groups must agree on physical block granularity. The repair
therefore derives, per attention layer,

```
aligned_block_size = SSM_plane_bytes / kv_row_bytes        (128 → 1536 for the reported SWA geometry)
```

so that every layer's K and V plane occupies exactly `S` bytes per physical
block. The kernel still sees 128-token blocks
(`kernel_num_blocks = num_blocks * aligned/128`); only scheduler-visible
granularity changes. Additionally the PR

- validates materialized pointers/dtypes/contiguity/arena offsets at startup;
- replans capacity under two conservative FIA addressing bounds
  (`< 2**16` kernel blocks, `< 2**32` plane elements; 5461 physical blocks for
  the reported geometry) — an operator workaround, not a CANN fix;
- masks the entire reserved physical null block (block 0, which now spans
  several 128-token kernel blocks) and out-of-range slots on both write paths
  (`reshape_and_cache`, `do_kv_cache_update`), because the hybrid manager
  inserts null-block IDs into sliding-window block tables after eviction and
  DFlash prewrites context K/V for *all* target tokens, unlike the target
  forward.

This is the minimal layout that keeps disjointness under a shared pool; the
alternative (keep 128-token SWA blocks) would require per-group block counts,
which the shared-pool manager does not support.

## 3. Why mixed can be slower than all-full despite higher acceptance

End-to-end time ≈ (decode steps) × (per-step time). The mixed drafter wins on
the first factor (higher acc-len → fewer target verifications), so the
regression must live in per-step time or in the measurement setup. The
mixed-only differences, ranked by expected impact:

### S0. Measurement protocol (rule this out first)

The repair's validation guide prescribes `--enforce-eager` and
`--num-gpu-blocks-override 4096` for the first mixed-window runs. If the
all-full comparison ran with graphs enabled and/or automatic capacity, the
comparison is not apples-to-apples and can easily dominate the result. Also
confirm: same concurrency, same vLLM/vllm-ascend commits for the shared code,
similar output-length distributions, and zero preemptions
(`preemption` counts in the server logs). A preemption-driven recompute loop
would also present as "higher acceptance but slower".

### S1. Per-step work that scales with the number of draft KV groups

Mixed runs **two** drafter KV groups (SWA + Full) where all-full runs one:

- Draft attention metadata is built per group; the builder performs
  GPU→CPU transfers (`seq_lens.to("cpu")`, `.tolist()`) per group per step
  (`attention_v1.py`), and before the metadata-reuse patch it was built twice
  per step (once in `propose`, once in `run_fullgraph`). Each build is a
  pipeline-sync point.
- `prepare_dflash_inputs` launches once per group; the pre-vectorization
  kernel issued O(max_num_tokens) scalar padding stores per launch.

The `fix_swa3` commits (included in this branch) already remove most of S1:
tiled context/padding writes, fused slot guards, per-propose metadata reuse,
and exact-seq-len sharing across builders. If a gap remains after S0 is
excluded, measure before optimizing further.

### S2. FIA `sparse_mode=4` on the SWA drafter layers (primary kernel suspect)

In the mixed drafter, the four SWA layers are *causal* and run FIA with
`sparse_mode=4, pre_tokens=2048, next_tokens=0`; the single Full layer (and
every layer of the all-full drafter) runs non-causal `sparse_mode=0`. On
Ascend, the windowed/sparse FIA schedule is not guaranteed to cost the same
as the dense no-mask schedule: depending on the CANN version it may disable
block skipping (reading the whole KV anyway) and add mask overhead, or take a
less optimized tiling. The drafter runs every decode step with
`seq_lens = full context + query`, so even a moderate per-op regression on
4 of 5 drafter layers is multiplied by every step. This is the only remaining
*kernel-level* difference between the two configurations and cannot be
settled by reading host code — it needs a micro-benchmark or profiling
(Section 4).

### S3. Small constant terms (unlikely to dominate, listed for completeness)

- `SlidingWindowManager.remove_skipped_blocks` per `allocate_slots` (Python,
  per running request per step).
- Null-block masking (`torch.where` per attention layer per forward; fused
  into a Triton guard by the `fix_swa3` perf commit).
- Kernel block-table expansion `storage(1536) → kernel(128)` per step
  (id arithmetic, same table width as all-full).
- Prefix-cache hashing granularity 1536 vs 128 tokens; only matters with long
  shared prefixes, and the GDN target typically disables prefix caching in
  both configurations anyway.
- FIA guard capacity (5461 physical blocks ≈ 8.4M tokens for the reported
  geometry) — a capacity bound, not a per-step cost; only relevant if the
  workload would otherwise preempt (see S0).

## 4. Confirming the dominant term on hardware

This branch adds an env-gated phase timer (zero overhead when off):

```bash
VLLM_ASCEND_DFLASH_PHASE_TIMING=1 \
VLLM_ASCEND_DFLASH_PHASE_TIMING_INTERVAL=50 \
vllm serve ...   # same flags for both configurations
```

Every `interval` propose steps it logs per-step averages for:

- `prepare_inputs` — the per-group Triton input/slot preparation;
- `context_kv` — `precompute_and_store_context_kv` (eager, outside the graph);
- `metadata_build` — draft attention metadata construction (CPU + syncs);
- `graph_replay` / `eager_forward` — draft forward, split by whether the
  FULL ACL graph actually replayed (an unexpected `eager_forward` share in
  the mixed configuration would pinpoint a graph-capture regression);
- `propose_total` and the running full-graph/eager step ratio.

Run both configurations with identical settings and compare:

- If `graph_replay`/`eager_forward` dominates and grows with the SWA group,
  S2 is confirmed → pursue the tail-window block-table experiment (Section 5)
  and/or raise `sparse_mode=4` performance with CANN.
- If `metadata_build`/`prepare_inputs` dominate, S1 is not fully closed →
  deduplicate per-group work further.
- If mixed unexpectedly logs `eager_forward` steps, the draft is falling out
  of FULL graph mode → investigate capture for hybrid drafter backends (the
  `FIXME: support hybrid attn backend` in `dflash/aclgraph.py`).

A targeted FIA micro-benchmark at the drafter geometry (TND, paged,
`block_size=128`, `seq_len` sweeping 1k–8k, `sparse_mode=0` vs
`sparse_mode=4, pre_tokens=2048`) settles S2 independently of serving.

## 5. Recommended fixes, in order

1. **Normalize the benchmark protocol** (S0): identical graph mode, capacity
   settings, concurrency; log preemption counts and output lengths.
2. **Keep the `fix_swa3` per-step reductions** (already on this branch).
3. **Measure with the phase timer** and act on the largest term.
4. If S2 confirms: evaluate a *tail-window block table* for SWA drafter
   layers — pass FIA only the trailing `ceil((window + q_len) / 128)` kernel
   blocks per request with `actual_seq_kvlen` clamped to the window and run
   the dense causal/no-mask path instead of `sparse_mode=4`. This preserves
   window semantics up to block granularity (≤127 tokens of extra context),
   removes the masked schedule, and caps KV reads at the window. It must be
   validated for acceptance parity before adoption.
5. If the draft falls back to eager in the mixed configuration: extend
   full-graph capture/replay to hybrid drafter backends instead of keying on
   the first backend (`dflash/aclgraph.py`, `update_full_graph_params`).

## 6. Validation status of this document

Sections 1–2 are derived from the cache-layout code, the PR's CPU reproduction
and its geometry tests. Section 3 ranks host-visible differences; the split
between S1/S2 requires the on-device measurements in Section 4. Nothing here
changes sampling, masks, GDN rollback, or model weights.
