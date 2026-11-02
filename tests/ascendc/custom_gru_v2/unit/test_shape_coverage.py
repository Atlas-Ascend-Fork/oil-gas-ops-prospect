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

"""Unit tests: CustomGRU v2 shape coverage.

Tests various B, T, H, F combinations including alignment edge cases.
Fixes from review:
- P0-4: H=1/T=1 cases previously triggered GM write overflow via DataCopy alignment.
        Now kernel uses SetData per element (no bulk DataCopy), so these are safe.
- P0-1: num_layers removed; no multi-layer shape tests.
"""
from __future__ import annotations

import torch


def _test_shape(B, T, H, F_in):
    from oil_gas_ops_prospect.custom_gru_v2 import custom_gru_v2
    torch.manual_seed(42)
    x = torch.randn(B, T, F_in)
    wi = torch.randn(3 * H, F_in) * 0.1
    wh = torch.randn(3 * H, H) * 0.1
    bi = torch.randn(3 * H) * 0.1
    bh = torch.randn(3 * H) * 0.1
    y, h = custom_gru_v2(x, wi, wh, bi, bh, H, False)
    assert y.shape == (1, B, T, H), f"y shape {y.shape} != (1,{B},{T},{H})"
    assert h.shape == (1, B, H), f"h shape {h.shape} != (1,{B},{H})"
    assert not torch.isnan(y).any()
    assert not torch.isnan(h).any()


def test_ut_shape_001_minimal():
    _test_shape(1, 1, 1, 1)


def test_ut_shape_002_aligned_h8():
    _test_shape(1, 10, 8, 3)


def test_ut_shape_003_unaligned_h7():
    _test_shape(1, 10, 7, 3)


def test_ut_shape_004_h16():
    _test_shape(1, 10, 16, 3)


def test_ut_shape_005_business_h24():
    _test_shape(1, 10, 24, 3)


def test_ut_shape_006_cross_align_h25():
    _test_shape(1, 10, 25, 3)


def test_ut_shape_007_train_batch():
    _test_shape(32, 100, 24, 3)


def test_ut_shape_008_long_seq():
    _test_shape(1, 1000, 24, 3)


def test_ut_shape_009_h1_boundary():
    _test_shape(1, 10, 1, 1)


def test_ut_shape_010_t2_minimal():
    _test_shape(1, 2, 8, 1)


def test_ut_shape_011_bidirectional_h8():
    """Bidirectional shape: verify y.shape == (2,B,T,8), NOT (2,B,T,16).

    This catches the P0-新-1 bug where hiddenSize = dim0/3 (without dividing
    by dirs) would compute H=2H in bidirectional mode, causing wrong shapes
    and GM read/write out of bounds.
    """
    from oil_gas_ops_prospect.custom_gru_v2 import custom_gru_v2
    H, F_in = 8, 3
    B, T = 2, 50
    torch.manual_seed(42)
    x = torch.randn(B, T, F_in)
    wi = torch.randn(2 * 3 * H, F_in) * 0.1
    wh = torch.randn(2 * 3 * H, H) * 0.1
    bi = torch.randn(2 * 3 * H) * 0.1
    bh = torch.randn(2 * 3 * H) * 0.1
    y, h = custom_gru_v2(x, wi, wh, bi, bh, H, True)
    assert y.shape == (2, B, T, H), f"Expected (2,{B},{T},{H}), got {y.shape}"
    assert h.shape == (2, B, H), f"Expected (2,{B},{H}), got {h.shape}"
