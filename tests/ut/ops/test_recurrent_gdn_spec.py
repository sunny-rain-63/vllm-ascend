# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from unittest.mock import MagicMock

import pytest
import torch

from vllm_ascend.ops.triton.fla import recurrent_gdn


def make_inputs(padding=13, index_stride=2):
    tokens, heads, value_heads, key_dim, value_dim = 10, 1, 2, 4, 3
    backing = torch.full((19, value_heads * value_dim * key_dim + padding), -7.0)
    state = backing[:, : value_heads * value_dim * key_dim].view(19, value_heads, value_dim, key_dim)
    indices = torch.full((2, 8 * index_stride), -1, dtype=torch.int32)[:, ::index_stride]
    indices.copy_(torch.arange(1, 17, dtype=torch.int32).reshape(2, 8))
    return dict(
        query=torch.zeros(tokens, heads, key_dim),
        key=torch.zeros(tokens, heads, key_dim),
        value=torch.zeros(tokens, value_heads, value_dim),
        state=state,
        g=torch.zeros(tokens, value_heads),
        beta=torch.zeros(tokens, value_heads),
        scale=0.5,
        query_start_loc=torch.tensor([0, 2, 10], dtype=torch.int32),
        ssm_state_indices=indices,
        num_accepted_tokens=torch.tensor([8, 5], dtype=torch.int32),
    )


@pytest.mark.parametrize("padding", [0, 13])
@pytest.mark.parametrize("index_stride", [1, 2])
def test_recurrent_launch_preserves_cache_and_request_strides(monkeypatch, padding, index_stride):
    inputs = make_inputs(padding, index_stride)
    kernel = MagicMock()
    monkeypatch.setattr(recurrent_gdn, "fused_recurrent_gated_delta_rule_fwd_kernel", kernel)
    output = recurrent_gdn.recurrent_gated_delta_rule_spec(**inputs)
    kwargs = kernel.__getitem__.return_value.call_args.kwargs
    assert kwargs["h0"] is kwargs["ht"] is inputs["state"]
    assert kwargs["ssm_state_indices"] is inputs["ssm_state_indices"]
    assert kwargs["stride_init_state_token"] == inputs["state"].stride(0)
    assert kwargs["stride_final_state_token"] == inputs["state"].stride(0)
    assert kwargs["stride_indices_seq"] == 8 * index_stride
    assert kwargs["stride_indices_tok"] == index_stride
    assert kwargs["SPEC_STATE_WIDTH"] == 8
    assert kwargs["IS_KDA"] is False
    assert kwargs["INPLACE_FINAL_STATE"] is True
    assert kwargs["cu_seqlens"] is inputs["query_start_loc"]
    assert kwargs["num_accepted_tokens"].tolist() == [8, 5]
    assert torch.count_nonzero(output) == 0  # skipped graph rows are initialized


@pytest.mark.parametrize("padding", [0, 13])
@pytest.mark.parametrize("index_stride", [1, 2])
def test_fla_stages_only_batch_states_and_scatter_keeps_original_cache(monkeypatch, padding, index_stride):
    inputs = make_inputs(padding, index_stride)
    for name in ("query", "key", "value", "beta"):
        inputs[name] = inputs[name].to(torch.bfloat16)
    prepare, scatter, fallback = MagicMock(), MagicMock(), MagicMock()
    fused = MagicMock(return_value=torch.empty_like(inputs["value"]))
    monkeypatch.setattr(recurrent_gdn, "prepare_spec_states_kernel", prepare)
    monkeypatch.setattr(recurrent_gdn, "scatter_spec_states_kernel", scatter)
    monkeypatch.setattr(recurrent_gdn, "recurrent_gated_delta_rule", fused)
    monkeypatch.setattr(recurrent_gdn, "fused_recurrent_gated_delta_rule_fwd_kernel", fallback)
    output = recurrent_gdn.recurrent_gated_delta_rule_spec(**inputs)
    fused.assert_called_once()
    fallback.__getitem__.assert_not_called()
    workspace = fused.call_args.args[3]
    assert workspace.is_contiguous()
    assert workspace.shape == (11, 2, 3, 4)  # 10 scheduled tokens, not 19 cache rows
    assert workspace.untyped_storage().data_ptr() != inputs["state"].untyped_storage().data_ptr()
    assert fused.call_args.kwargs["num_accepted_tokens"] is None
    assert fused.call_args.kwargs["ssm_state_indices"].tolist() == list(range(1, 11))
    assert prepare.__getitem__.return_value.call_args.args[0] is inputs["state"]
    assert scatter.__getitem__.return_value.call_args.args[1] is inputs["state"]
    assert scatter.__getitem__.return_value.call_args.args[6] == inputs["state"].stride(0)
    assert output is fused.return_value


def test_more_than_eight_states_keeps_verified_triton_path(monkeypatch):
    inputs = make_inputs()
    for name in ("query", "key", "value", "beta"):
        inputs[name] = inputs[name].to(torch.bfloat16)
    inputs["ssm_state_indices"] = torch.ones(2, 9, dtype=torch.int32)
    fused, fallback = MagicMock(), MagicMock()
    monkeypatch.setattr(recurrent_gdn, "recurrent_gated_delta_rule", fused)
    monkeypatch.setattr(recurrent_gdn, "fused_recurrent_gated_delta_rule_fwd_kernel", fallback)
    recurrent_gdn.recurrent_gated_delta_rule_spec(**inputs)
    fused.assert_not_called()
    fallback.__getitem__.assert_called_once()


@pytest.mark.parametrize("field", ["query_start_loc", "ssm_state_indices", "num_accepted_tokens"])
def test_recurrent_rejects_noninteger_metadata(field):
    inputs = make_inputs()
    inputs[field] = inputs[field].float()
    with pytest.raises(TypeError, match="must be integers"):
        recurrent_gdn.recurrent_gated_delta_rule_spec(**inputs)


def test_recurrent_rejects_flattened_state_table():
    inputs = make_inputs()
    inputs["ssm_state_indices"] = inputs["ssm_state_indices"].flatten()
    with pytest.raises(ValueError, match="fixed-width"):
        recurrent_gdn.recurrent_gated_delta_rule_spec(**inputs)


def test_recurrent_rejects_nondense_inner_state():
    inputs = make_inputs()
    inputs["state"] = torch.zeros(19, 2, 3, 8)[..., ::2]
    with pytest.raises(ValueError, match="State rows"):
        recurrent_gdn.recurrent_gated_delta_rule_spec(**inputs)


def test_empty_recurrent_does_not_launch(monkeypatch):
    inputs = make_inputs()
    for name in ("query", "key", "value", "g", "beta"):
        inputs[name] = inputs[name][:0]
    inputs["query_start_loc"] = torch.tensor([0], dtype=torch.int32)
    inputs["ssm_state_indices"] = inputs["ssm_state_indices"][:0]
    inputs["num_accepted_tokens"] = inputs["num_accepted_tokens"][:0]
    kernel = MagicMock()
    monkeypatch.setattr(recurrent_gdn, "fused_recurrent_gated_delta_rule_fwd_kernel", kernel)
    assert recurrent_gdn.recurrent_gated_delta_rule_spec(**inputs).shape == (0, 2, 3)
    kernel.__getitem__.assert_not_called()
