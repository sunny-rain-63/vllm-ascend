import torch
import torch.nn.functional as F
from vllm.model_executor.models.qwen3_dflash import (
    DFlashQwen3ForCausalLM,
    DFlashQwen3Model,
)
from vllm.triton_utils import HAS_TRITON, triton

if HAS_TRITON:
    from vllm_ascend.ops.triton.rope import rope_forward_triton_siso


def _ensure_ascend_dflash_buffers(self) -> None:
    """Build the fused KV/RoPE buffers plus Ascend-specific derived tensors.

    ``_build_fused_kv_buffers`` is idempotent and may already have run during
    weight loading, so only the missing pieces are (re)built here.
    """
    if not hasattr(self, "_num_attn_layers"):
        self._build_fused_kv_buffers()
    if hasattr(self, "_rope_use_siso"):
        return
    L = self._num_attn_layers
    hd = self._head_dim
    # fp32 stacked per-layer K-norm weights [L, 1, 1, hd] so the grouped
    # RMSNorm below is one broadcast multiply instead of L small launches.
    self._k_norm_weights_f32 = self._k_norm_weights.float().view(L, 1, 1, hd).contiguous()
    # cos_sin_cache is [max_position, rotary_dim].
    self._rope_rotary_dim = self._rope_cos_sin_cache.shape[-1]
    # rope_forward_triton_siso reads the sine half at pad_rope_dim // 2, so the
    # cos_sin_cache path is only exact when rotary_dim is a power of two.
    self._rope_use_siso = HAS_TRITON and self._rope_rotary_dim == triton.next_power_of_2(self._rope_rotary_dim)


def precompute_and_store_context_kv(
    self,
    context_states: torch.Tensor,
    context_positions: torch.Tensor,
    context_slot_mapping: torch.Tensor | list[torch.Tensor | None] | None = None,
) -> None:
    _ensure_ascend_dflash_buffers(self)

    num_ctx = context_states.shape[0]
    L = self._num_attn_layers
    kv = self._kv_size
    hd = self._head_dim
    nkv = self._num_kv_heads

    # --- Fused KV projection (one GEMM for all layers) ---
    normed_context_states = self.hidden_norm(context_states)
    all_kv_flat = F.linear(normed_context_states, self._fused_kv_weight, self._fused_kv_bias)
    # Single contiguous copy that separates K/V and transposes to
    # layer-major layout.  Result: [2, L, num_ctx, nkv, hd] contiguous.
    # Indexing dim-0 gives contiguous [L, num_ctx, nkv, hd] for K and V.
    all_kv = all_kv_flat.view(num_ctx, L, 2, nkv, hd).permute(2, 1, 0, 3, 4).contiguous()
    all_k = all_kv[0]  # [L, num_ctx, nkv, hd], contiguous
    all_v = all_kv[1]  # [L, num_ctx, nkv, hd], contiguous

    # --- Grouped RMSNorm K across all layers ---
    # One vectorized pass over [L, num_ctx, nkv, hd] with the per-layer weights
    # broadcast on the layer axis, instead of L separate small npu_rms_norm
    # launches.  Numerics match RMSNorm: fp32 variance, scale, then weight.
    k32 = all_k.float()
    var = k32.pow(2).mean(-1, keepdim=True)
    all_k_normed = (k32 * torch.rsqrt(var + self._rms_norm_eps) * self._k_norm_weights_f32).to(all_k.dtype)

    # --- Fused RoPE across all layers ---
    positions_repeated = context_positions.repeat(L)
    if self._rope_use_siso:
        # Single-tensor in-place RoPE: no dummy key and no full-tensor clone.
        all_k_flat = all_k_normed.view(L * num_ctx, nkv, hd)
        cos_sin_cache = self._rope_cos_sin_cache
        if cos_sin_cache.dtype != all_k_flat.dtype:
            cos_sin_cache = cos_sin_cache.to(dtype=all_k_flat.dtype)
        rope_forward_triton_siso(
            all_k_flat,
            cos_sin_cache=cos_sin_cache,
            positions=positions_repeated,
            rope_dim=self._rope_rotary_dim,
            is_neox_style=self._rope_is_neox,
        )
    else:
        # Non-triton fallback: the module's custom op rotates in place but
        # requires a writable key tensor, so keep the clone on this path.
        all_k_flat = all_k_normed.view(L * num_ctx, kv)
        tmpv = all_k_flat.clone()
        self.layers[0].self_attn.rotary_emb(positions_repeated, all_k_flat, tmpv)
        all_k_flat = all_k_flat.view(L * num_ctx, nkv, hd)

    if context_slot_mapping is None:
        return

    # --- Per-layer cache insert ---
    all_k_final = all_k_flat.view(L, num_ctx, nkv, hd)
    per_layer = isinstance(context_slot_mapping, (list, tuple))
    for i in range(L):
        slot_mapping = context_slot_mapping[i] if per_layer else context_slot_mapping
        if slot_mapping is None:
            continue
        attn = self._attn_layers[i]
        kv_cache = attn.kv_cache
        attn.impl.do_kv_cache_update(
            attn,
            all_k_final[i],
            all_v[i],
            kv_cache,
            slot_mapping,
        )


DFlashQwen3Model.precompute_and_store_context_kv = precompute_and_store_context_kv

_orig_read_mask_embedding = DFlashQwen3ForCausalLM._read_mask_embedding


def _patched_read_mask_embedding(self):
    try:
        return _orig_read_mask_embedding(self)
    except Exception:
        return None


DFlashQwen3ForCausalLM._read_mask_embedding = _patched_read_mask_embedding
