# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Budget-aware grouping without changing mixed DFlash's cache page layout."""

from dataclasses import fields, replace

from vllm.config import VllmConfig
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheGroupSpec,
    KVCacheSpec,
    MambaSpec,
    SlidingWindowSpec,
)


def _same_group_metadata(left: KVCacheGroupSpec, right: KVCacheGroupSpec) -> bool:
    return all(
        getattr(left, field.name) == getattr(right, field.name)
        for field in fields(left)
        if field.name not in ("layer_names", "is_eagle_group")
    )


def choose_dflash_cache_groups(
    vllm_config: VllmConfig,
    kv_cache_spec: dict[str, KVCacheSpec],
    original_groups: list[KVCacheGroupSpec],
    *,
    draft_layer_names: set[str],
    available_memory: int,
    max_num_blocks: int,
) -> list[KVCacheGroupSpec] | None:
    """Reduce cap-induced waste, keeping exact specs and target/draft ownership.

    The caller must first align and validate mixed DFlash's physical planes.
    ``available_memory`` and ``max_num_blocks`` are the minimum real worker
    budget and FIA block limit, respectively; this routine never raises them.
    Only widths up to the first one that can use the budget below the cap are
    considered. A candidate must strictly improve startup admission capacity;
    this estimate is not a prediction of runtime concurrency or throughput.
    """
    speculative = getattr(vllm_config, "speculative_config", None)
    if (
        not getattr(vllm_config, "use_v2_model_runner", False)
        or speculative is None
        or speculative.method != "dflash"
        or getattr(vllm_config.parallel_config, "pipeline_parallel_size", 1) != 1
        or not original_groups
    ):
        return None
    if any(type(spec) not in (FullAttentionSpec, SlidingWindowSpec, MambaSpec) for spec in kv_cache_spec.values()):
        return None
    if not draft_layer_names or not draft_layer_names <= kv_cache_spec.keys():
        raise ValueError("DFlash grouping requires an explicit, valid draft-layer set.")
    if any(not any(field.name == "is_eagle_group" for field in fields(group)) for group in original_groups):
        # Older upstream groups cannot preserve explicit draft ownership when
        # projected to workers. Keep their existing safe grouping.
        return None
    draft_types = {type(kv_cache_spec[name]) for name in draft_layer_names}
    if MambaSpec in draft_types:
        raise ValueError("DFlash grouping cannot mark target Mamba states as draft layers.")
    if draft_types != {FullAttentionSpec, SlidingWindowSpec} or not any(
        isinstance(spec, MambaSpec) for spec in kv_cache_spec.values()
    ):
        return None
    page_sizes = {spec.page_size_bytes for spec in kv_cache_spec.values()}
    if len(page_sizes) != 1 or min(available_memory, max_num_blocks) <= 0:
        return None
    page_size = next(iter(page_sizes))
    if page_size <= 0:
        return None
    original_names = [name for group in original_groups for name in group.layer_names]
    if len(original_names) != len(kv_cache_spec) or set(original_names) != kv_cache_spec.keys():
        raise ValueError("DFlash groups must contain every cache layer exactly once.")
    if any(kv_cache_spec[name] != group.kv_cache_spec for group in original_groups for name in group.layer_names):
        # Do not discard merged or UniformType per-layer metadata.
        return None

    block_limit = max_num_blocks
    override = vllm_config.cache_config.num_gpu_blocks_override
    if override is not None:
        block_limit = min(block_limit, override)
    if block_limit < 2:
        return None
    original_width = max(len(group.layer_names) for group in original_groups)
    original_blocks = min(available_memory // (page_size * original_width), block_limit)
    if original_blocks < 2 or available_memory // (page_size * original_width) <= block_limit:
        return None

    # Group metadata, including enable_kv_transfer, is part of ownership.
    # Never merge a target/draft pair, even when its attention specs match.
    buckets: list[tuple[KVCacheGroupSpec, list[str], bool]] = []
    for group in original_groups:
        for name in group.layer_names:
            is_draft = name in draft_layer_names
            for template, names, bucket_is_draft in buckets:
                if is_draft == bucket_is_draft and _same_group_metadata(group, template):
                    names.append(name)
                    break
            else:
                buckets.append((group, [name], is_draft))

    per_request_blocks = {
        name: (spec.max_memory_usage_bytes(vllm_config) + page_size - 1) // page_size
        for name, spec in kv_cache_spec.items()
    }
    if any(count <= 0 for count in per_request_blocks.values()):
        return None
    best_request_blocks = sum(per_request_blocks[group.layer_names[0]] for group in original_groups)
    best_blocks = original_blocks
    best_groups: list[KVCacheGroupSpec] | None = None
    # Wider pools beyond this point cannot recover any more cap-limited bytes.
    max_width = min(
        (available_memory + page_size * block_limit - 1) // (page_size * block_limit),
        max(len(names) for _, names, _ in buckets),
    )
    for width in range(original_width + 1, max_width + 1):
        groups: list[KVCacheGroupSpec] = []
        for template, names, is_draft in buckets:
            num_groups = (len(names) + width - 1) // width
            groups.extend(
                replace(template, layer_names=names[index::num_groups], is_eagle_group=is_draft)
                for index in range(num_groups)
            )
        actual_width = max(len(group.layer_names) for group in groups)
        blocks = min(available_memory // (page_size * actual_width), block_limit)
        request_blocks = sum(per_request_blocks[group.layer_names[0]] for group in groups)
        # Account for the permanent null block, and compare exact ratios.
        if blocks > 1 and (blocks - 1) * best_request_blocks > (best_blocks - 1) * request_blocks:
            best_groups = groups
            best_blocks = blocks
            best_request_blocks = request_blocks
    return best_groups
