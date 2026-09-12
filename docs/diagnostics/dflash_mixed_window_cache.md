# Qwen3.6-27B mixed-window DFlash cache repair and validation

## Scope

This change addresses Ascend V2 with an ordinary FP16/BF16 Full/SWA DFlash
draft and a Mamba/GDN target, including the reported Qwen3.6-27B TP2 geometry.
It preserves the official four `sliding_attention` layers, one
`full_attention` layer and `sliding_window=2048`. It does not rewrite the model
configuration or turn sliding attention into full attention.

Target-only, all-full DFlash and all-sliding DFlash paths are unchanged.
Unsupported mixed layouts fail at initialization instead of silently using an
unverified layout. Pipeline parallelism and quantized KV storage are not covered
by this repair.

## Two independent failure mechanisms

### Shared-plane aliasing at small block IDs

The Ascend V2 materializer places contiguous Mamba conv/SSM planes at the start
of a raw pool, and contiguous attention K/V planes at its tail. Merely setting
the same `page_size_padded` on all specs does not make those planes agree on
the byte ranges owned by each physical block ID.

For the reported BF16 KV / FP32 recurrent-state TP2 geometry:

| Quantity | Bytes |
| --- | ---: |
| Conv page `C` | 102400 |
| SSM page `S` | 1572864 |
| Common page `P = C + 2*S` | 3248128 |
| One full K or V block, 1536 tokens | 1572864 |
| One unaligned SWA K or V block, 128 tokens | 131072 |

With just **12 physical blocks**, the old SWA block 1 K and V addresses lie
inside full-attention V blocks 10 and 11, respectively. These IDs are different
and non-null, so correct scheduler ownership cannot prevent the overwrite.
The regression test reproduces this with a roughly 39 MB byte array; it does
not require high addresses or an NPU.

The repair derives an attention storage block size from `S / KV_row_bytes`:
1536 in this example. Both Full and SWA then use the same physical plane sizes.
The existing block-table expansion still presents 128-token blocks to the
attention kernel. The sliding window remains 2048 tokens; storage granularity
and attention visibility are different concepts.

The resulting shared pool is `[all conv, all K/SSM, all V]`. Different physical
block IDs have disjoint byte ranges across groups. Startup also checks actual
tensor pointers, dtype, contiguity and shared-arena offsets, rather than only
checking spec metadata.

### Independent large-address FIA failure

The supplied `check_kv_cache_addressing.log` previously showed that KV scatter
writes passed while FIA reads failed at kernel block IDs 65536 and 65543,
including with independently filled reference KV. For the probe geometry,
kernel block 65536 starts at element `2**32` in a K/V plane.

This identifies an operator addressing boundary, but does not establish which
internal CANN index type is responsible. The repair conservatively limits both
the kernel-block count and the per-plane element count before allocating device
memory. It also respects the real profiled per-worker memory budget, including
when an explicit block override is supplied. The original planner is rerun so
admission checks and descriptor offsets use the reduced capacity.

With 1536-token storage blocks, 128-token kernel blocks and 512 BF16 elements
per token, the address-derived maximum is 5461 physical blocks. The actual plan
may be smaller because of memory or an explicit override. This is an operator
workaround, not a replacement for a CANN kernel fix; the 5461 boundary still
needs hardware validation on each affected operator/device version.

The write paths also mask the **entire** reserved physical null block, not just
kernel block zero. With a 1536-token physical block this covers kernel blocks
0 through 11. Negative and out-of-range slots are masked as padding too.

## Local regression tests

In the normal vLLM-Ascend test environment, run from the repository root:

```bash
pytest -q tests/ut/worker/test_dflash_cache_layout.py tests/ut/worker/test_dflash_cache.py tests/ut/worker/test_dflash_cache_views.py
```

The layout suite covers byte-level alias reproduction and disjoint aligned
ranges. The metadata suite executes the production alignment/planner functions
with lightweight dependency fixtures; it does not validate vLLM runtime
integration. The view suite uses real Torch tensors, real vLLM cache specs and
the production materializer, and checks that SWA writes preserve other groups'
physical blocks. It needs the normal Torch/vLLM/Ascend test dependencies.

Passing these tests is not proof that NPU attention outputs are accurate.

## Device validation order

1. Deploy all changed Python modules together and restart every worker. Check
   `vllm_ascend.__file__` in the serving environment to ensure the edited checkout
   is the package actually imported. No C++ kernel rebuild is required by this
   change itself. Keep the official draft configuration unchanged.
2. First retain `--num-gpu-blocks-override 4096` and `--enforce-eager`. This tests
   the new layout at the previously usable capacity without changing two
   variables at once. Keep the same target/draft weights, prompt template,
   reasoning setting, `temperature=0`, `top_p=1` and `num_speculative_tokens=7`.
3. Confirm startup emits all applicable markers below. For the reported TP2
   geometry, SWA storage should change from 128 to 1536 and retain window 2048.
   The final capacity must not exceed the address bound or the profiled memory
   budget. A startup exception means the layout was rejected, not that a model
   accuracy test passed.
4. Replay the **same complete GPQA subset and order that reproduced the issue**,
   including warmup. First use client concurrency 1, then 32. Include long
   generations crossing 1536, 2048, 3072 and later boundaries, different prompts,
   repeated prompts, and multiple rounds without restarting to exercise block
   reuse. One short curl is insufficient.
5. Save raw responses with `finish_reason`, completion-token counts and request
   identifiers, plus the full server log. Preserve text before the AISBench
   reasoning postprocessor. Count repetitive outputs and distinguish
   `finish_reason=length` from EOS/stop; high acceptance alone is not a pass.
6. Once eager + 4096 passes, remove the override while keeping eager. Check the
   automatic plan and rerun the identical workload. Finally enable graph mode
   and repeat. Do not apply an arbitrary 4096 override to the target-only
   baseline: its memory-per-block plan can differ substantially.

Expected startup markers:

```text
DFlash mixed cache layout aligned: ... block_size=128 -> 1536, sliding_window=2048 ...
DFlash mixed cache guard: physical_blocks=... -> ...  # only when clamping is needed
DFlash mixed cache plan ready: physical_blocks=... (memory/address safe)
DFlash mixed cache views verified: ... physical_blocks=... conv_bytes=102400, plane_bytes=1572864
```

## If repetitions remain after the layout checks pass

Do not treat a long accepted draft as proof of correctness: if target cache
state was corrupted, the draft can agree with incorrect target logits.
Conversely, repetition by itself does not prove corruption, and the alias
reproduction does not prove it caused every previously observed repetition.

The next discriminating experiment is target verification on an identical
token prefix. Retain the request/prompt and emitted token IDs leading into the
first repeated passage; replay that exact prefix on a fresh target-only service
with identical rendering and greedy settings. Compare the next-token top
logits/token IDs at the same positions, allowing for numerical near-ties.
Comparing two free-running texts after their first divergence is not equivalent.

If target logits differ substantially at an identical prefix, inspect the
target KV/state and rollback path. If target logits match but an inconsistent
draft token is accepted, inspect verification/rejection and accepted-length
bookkeeping. If target-only also repeats from that identical prefix, inspect the
generation trajectory/template and stopping behavior. This second-stage test
is not installed as a runtime verifier by this patch.
