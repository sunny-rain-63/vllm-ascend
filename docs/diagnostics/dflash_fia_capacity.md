# Experimental DFlash FIA cache capacity

This experiment extends the standalone FIA guard, not the SWA layout patch.
It preserves the copy-on-write container compatibility fix. It does not change
attention windows, block sizes, cache grouping, or the FIA kernel.

## Policies

Set `VLLM_ASCEND_DFLASH_FIA_MAX_BLOCKS` before starting the service:

| Value | Policy | CPU fixture effective blocks | Capacity vs default |
| --- | --- | --- | --- |
| 4096 (default) | Existing conservative cap | 4096 | 1.00x |
| 4608 | Intermediate experimental cap | 4608 | 1.125x |
| 5120 | Intermediate experimental cap | 5120 | 1.25x |
| 0 | Calculated address and memory limits only | 5461 | 1.333x |
| 8192 | Larger request, still address-limited | 5461 | 1.333x |

The fixture uses 1536-token full-attention storage blocks, 128-token kernel
blocks, four KV heads, and 128-element heads, with sufficient memory. These
are capacity calculations, not measured concurrency or throughput gains.
The existing kernel-block and plane-element bounds remain active in all modes.
A smaller explicit `--num-gpu-blocks-override` remains effective, so keeping
4096 on the command line prevents this experiment from increasing capacity.

The calculated bounds describe the existing workaround's assumptions. They
are not proof of CANN kernel safety across versions or device types. Only the
default cap has the user's reported workload validation. Review the new
environment variable and NPU results before production adoption.

## CPU validation

Run from the repository root, without Torch or an NPU:

```bash
python tests/ut/worker/test_dflash_address_limit.py -v
```

The tests execute the production planning wrapper with simulated vLLM specs
and allocation descriptors. Sixteen tests cover five capacity policies, 36
block/head geometries, memory budgets, rank limits, explicit overrides, legacy
separate allocations, unaffected model paths, and a planner ignoring the cap.
They do not allocate real KV tensors or exercise scheduler admission or FIA.

## NPU acceptance

1. Keep the same dataset/order, sampling, concurrency, graph mode and versions.
   Record the default 4096 baseline first.
2. Remove a fixed 4096 CLI override. Start with:

   ```bash
   export VLLM_ASCEND_DFLASH_FIA_MAX_BLOCKS=4608
   ```

3. Check `DFlash FIA capacity:` for the effective allocation. Run the address
   probe with that allocation, including the highest valid blocks, then the
   original long-context concurrent workload. Stop if accuracy regresses.
4. If successful, separately test 5120, then 0. Restart between policies.
5. Compare total output tokens, output tokens/s, Running/Waiting, KV usage,
   preemptions, finish reasons and output correctness. Acceptance length or
   elapsed dataset time alone is insufficient.
6. Return to the default with:

   ```bash
   unset VLLM_ASCEND_DFLASH_FIA_MAX_BLOCKS
   ```

Do not remove address bounds to recover additional capacity. Beyond this
limited experiment, improving grouping or fixing the operator requires
separate implementation and correctness validation.
