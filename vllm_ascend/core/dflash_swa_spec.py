# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Widen DFlash draft sliding-window specs to full attention on Ascend V2.

WHY
===
A DFlash drafter whose ``layer_types`` mix ``full_attention`` and
``sliding_attention`` is supported upstream since vLLM #47914 (per-KV-group
causal metadata), but on Ascend the resulting mixed Full/SWA cache groups
share one contiguous backing with the Mamba planes. Distinct logical block
ids then alias the same physical memory and DFlash context prewrites corrupt
each other (garbled output, acceptance length decaying to 1).

Upstream's own follow-up direction (unmerged #40898, and #50457 for the
all-sliding case) avoids the mixed pool entirely: DFlash prewrites context
K/V for every target token and cannot evict out-of-window blocks, so draft
sliding-window layers are booked as ``FullAttentionSpec`` and the window is
enforced at compute time. This module applies that design on Ascend.

On Ascend the window does NOT depend on the spec type:
``AscendAttentionBackendImpl`` receives ``sliding_window`` from the Attention
layer constructor and enforces it in the FIA kernel via ``sparse_mode=4`` /
``pre_tokens``. Widening the spec therefore changes only cache *accounting*:
draft SWA layers allocate full-length KV (the drafter is small), block tables
come from the full-attention manager (no null blocks), and every attention
group shares one uniform geometry — the same shape as the already-working
all-full DFlash + Mamba target configuration.

REMOVAL GUIDE
=============
This module is self-contained; its only call site is
``NPUModelRunner.get_kv_cache_spec`` in
``vllm_ascend/worker/v2/model_runner.py``, marked with
``# DFLASH-SWA-SPEC-WIDENING``. Re-evaluate on every upstream vLLM bump:

- If upstream lands #40898-style spec widening for mixed DFlash drafters,
  drop this module and use the upstream path.
- If the Ascend allocator stops materializing contiguous per-component
  planes, or vllm-ascend adopts the fix_swa_kimi5 cache-geometry alignment
  instead, this widening becomes unnecessary.
"""

from vllm.logger import logger
from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec


def dflash_draft_has_mixed_windows(vllm_config) -> bool:
    """Whether a V2 DFlash drafter mixes sliding-window and full attention.

    Mirrors upstream ``VllmConfig._dflash_needs_multi_kv_group`` (vLLM
    #47914), restricted to the V2 runner where the Ascend hybrid cache
    materializes the mixed groups.
    """
    if not getattr(vllm_config, "use_v2_model_runner", False):
        return False
    speculative = getattr(vllm_config, "speculative_config", None)
    if speculative is None or getattr(speculative, "method", None) != "dflash":
        return False
    draft_model_config = getattr(speculative, "draft_model_config", None)
    hf_config = getattr(draft_model_config, "hf_config", None)
    layer_types = getattr(hf_config, "layer_types", None) or []
    num_sliding = sum(layer_type == "sliding_attention" for layer_type in layer_types)
    return 0 < num_sliding < len(layer_types)


def widen_dflash_draft_swa_specs(vllm_config, specs, draft_layer_names):
    """Book mixed DFlash draft SWA layers as full-attention cache specs.

    Only draft layers named by the loaded speculator are widened; genuine
    target sliding-window layers keep their ``SlidingWindowSpec``. The
    sliding window value is carried onto the widened spec so metadata
    builders and diagnostics retain it. Identity for every other
    configuration.
    """
    if not dflash_draft_has_mixed_windows(vllm_config):
        return specs
    if not draft_layer_names:
        raise ValueError("Mixed DFlash spec widening requires loaded draft attention layer names.")
    missing = set(draft_layer_names) - specs.keys()
    if missing:
        raise ValueError(f"Mixed DFlash draft layers missing from KV cache specs: {sorted(missing)}.")
    result = dict(specs)
    widened: list[tuple[str, int]] = []
    for name in draft_layer_names:
        spec = result[name]
        if not isinstance(spec, SlidingWindowSpec):
            continue
        kwargs = dict(
            block_size=spec.block_size,
            num_kv_heads=spec.num_kv_heads,
            head_size=spec.head_size,
            head_size_v=spec.head_size_v,
            dtype=spec.dtype,
            sliding_window=spec.sliding_window,
        )
        for optional in ("kv_quant_mode", "page_size_padded", "num_head_slots", "state_content_bytes"):
            value = getattr(spec, optional, None)
            if value is not None:
                kwargs[optional] = value
        result[name] = FullAttentionSpec(**kwargs)
        widened.append((name, spec.sliding_window))
    if widened:
        logger.info(
            "DFlash draft SWA specs widened to full-attention cache specs: %s "
            "(window still enforced by the attention layer at compute time)",
            ", ".join(f"{name}(window={window})" for name, window in widened),
        )
    return result
