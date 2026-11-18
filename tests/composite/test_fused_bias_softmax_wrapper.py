# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import pytest
import torch

from oil_gas_ops_prospect import fused_bias_softmax as wrapper


class _FakeExtension:
    """Model the native binding's explicit in-place and out-of-place variants."""

    @staticmethod
    def fused_bias_softmax_forward(x, bias1=None, bias2=None):
        x.add_(17)
        return x

    @staticmethod
    def fused_bias_softmax_forward_out(x, y, bias1=None, bias2=None):
        y.copy_(x + 17)
        return y

    @staticmethod
    def fused_softmax_grad_backward(probs, grad_output, values):
        probs.add_(23)
        return probs

    @staticmethod
    def fused_softmax_grad_backward_out(probs, grad_output, values, out):
        out.copy_(probs + 23)
        return out


@pytest.fixture
def fake_ascendc(monkeypatch):
    monkeypatch.setattr(wrapper, "_impl", lambda: "ascendc")
    monkeypatch.setattr(wrapper, "_is_npu", lambda tensor: True)
    monkeypatch.setattr(wrapper, "_load_extension", lambda: _FakeExtension)


@pytest.mark.parametrize("noncontiguous", [False, True], ids=["contiguous", "transpose-view"])
def test_fused_bias_softmax_is_out_of_place_for_all_layouts(fake_ascendc, noncontiguous):
    base = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    x = base.transpose(0, 1) if noncontiguous else base
    before = x.clone()

    out = wrapper.fused_bias_softmax(x)

    assert torch.equal(x, before)
    assert torch.equal(out, before + 17)
    assert out is not x
    assert out.data_ptr() != x.data_ptr()


@pytest.mark.parametrize("noncontiguous", [False, True], ids=["contiguous", "transpose-view"])
def test_fused_softmax_grad_is_out_of_place_for_all_layouts(fake_ascendc, noncontiguous):
    base = torch.arange(120, dtype=torch.float32).reshape(2, 3, 4, 5)
    probs = base.transpose(1, 2) if noncontiguous else base
    before = probs.clone()
    batch, heads, rows, width = probs.shape
    grad_output = torch.zeros((batch, heads, rows, 2), dtype=probs.dtype)
    values = torch.zeros((batch, heads, width, 2), dtype=probs.dtype)

    out = wrapper.fused_softmax_grad(probs, grad_output, values)

    assert torch.equal(probs, before)
    assert torch.equal(out, before + 23)
    assert out is not probs
    assert out.data_ptr() != probs.data_ptr()


def test_fused_bias_softmax_inplace_requires_contiguous_input(fake_ascendc):
    x = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    out = wrapper.fused_bias_softmax_(x)

    assert torch.equal(x, torch.arange(24, dtype=torch.float32).reshape(2, 3, 4) + 17)
    assert out is x
    assert out.data_ptr() == x.data_ptr()
    with pytest.raises(ValueError, match="must be contiguous"):
        wrapper.fused_bias_softmax_(x.transpose(0, 1))


def test_fused_softmax_grad_inplace_requires_contiguous_input(fake_ascendc):
    probs = torch.arange(120, dtype=torch.float32).reshape(2, 3, 4, 5)
    grad_output = torch.zeros((2, 3, 4, 2), dtype=probs.dtype)
    values = torch.zeros((2, 3, 5, 2), dtype=probs.dtype)
    out = wrapper.fused_softmax_grad_(probs, grad_output, values)

    assert torch.equal(probs, torch.arange(120, dtype=torch.float32).reshape(2, 3, 4, 5) + 23)
    assert out is probs
    assert out.data_ptr() == probs.data_ptr()
    with pytest.raises(ValueError, match="must be contiguous"):
        wrapper.fused_softmax_grad_(probs.transpose(1, 2), grad_output.transpose(1, 2), values.transpose(1, 2))
