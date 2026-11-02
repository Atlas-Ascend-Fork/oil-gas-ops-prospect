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

"""Unit tests: CustomGRU v2 numerical stability.

Validates tanh = 2*sigmoid(2x)-1 stability for extreme input values.
No changes needed from review — original stability tests were correct.
"""
from __future__ import annotations

import torch


def _run_with_gate_scale(scale, H=8, F_in=3, T=10):
    from oil_gas_ops_prospect.custom_gru_v2 import custom_gru_v2
    torch.manual_seed(42)
    x = torch.randn(1, T, F_in) * scale
    wi = torch.randn(3 * H, F_in) * 0.1 * scale
    wh = torch.randn(3 * H, H) * 0.1 * scale
    bi = torch.randn(3 * H) * 0.1 * scale
    bh = torch.randn(3 * H) * 0.1 * scale
    y, h = custom_gru_v2(x, wi, wh, bi, bh, H, False)
    return y, h


def test_ut_num_001_tanh_large_positive():
    y, h = _run_with_gate_scale(20.0)
    assert not torch.isnan(y).any(), "NaN in output with large positive input"
    assert not torch.isinf(y).any(), "Inf in output with large positive input"


def test_ut_num_002_tanh_large_negative():
    y, h = _run_with_gate_scale(-20.0)
    assert not torch.isnan(y).any(), "NaN in output with large negative input"
    assert not torch.isinf(y).any(), "Inf in output with large negative input"


def test_ut_num_003_sigmoid_saturation():
    y, h = _run_with_gate_scale(50.0)
    assert not torch.isnan(y).any(), "NaN in output with saturated sigmoid"
    assert not torch.isinf(y).any(), "Inf in output with saturated sigmoid"


def test_ut_num_004_sigmoid_negative_saturation():
    y, h = _run_with_gate_scale(-50.0)
    assert not torch.isnan(y).any(), "NaN in output with negative saturated sigmoid"
    assert not torch.isinf(y).any(), "Inf in output with negative saturated sigmoid"


def test_ut_num_005_mixed_extreme_values():
    from oil_gas_ops_prospect.custom_gru_v2 import custom_gru_v2
    H, F_in, T = 8, 3, 10
    torch.manual_seed(123)
    x = torch.randn(1, T, F_in)
    for i in range(T):
        if i % 2 == 0:
            x[0, i, :] *= 100
    wi = torch.randn(3 * H, F_in) * 0.1
    wh = torch.randn(3 * H, H) * 0.1
    bi = torch.randn(3 * H) * 0.1
    bh = torch.randn(3 * H) * 0.1
    y, h = custom_gru_v2(x, wi, wh, bi, bh, H, False)
    assert not torch.isnan(y).any(), "NaN in output with mixed extreme values"
    assert not torch.isinf(y).any(), "Inf in output with mixed extreme values"
    assert y.abs().max().item() < 1e6, "Output exploded"
