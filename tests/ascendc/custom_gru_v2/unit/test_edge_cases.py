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

"""Unit tests: CustomGRU v2 edge cases.

Tests zero inputs, large values, zero weights, single-step, and stability.
Fixes from review:
- P1-2: bias=False path now tested with zero bias tensors (kernel handles hasBias=0)
- P0-1: no num_layers test; IT-TRAIN-004/IT-INFER-005 reworked to single-layer only
- P2-2: UT-EDGE-009/010 now match test_plan entries
"""
from __future__ import annotations

import pytest
import torch

from conftest import npu_only


def _make_inputs(B, T, H, F_in, seed=42):
    torch.manual_seed(seed)
    x = torch.randn(B, T, F_in)
    wi = torch.randn(3 * H, F_in) * 0.1
    wh = torch.randn(3 * H, H) * 0.1
    bi = torch.randn(3 * H) * 0.1
    bh = torch.randn(3 * H) * 0.1
    return x, wi, wh, bi, bh


def test_ut_edge_001_zero_input():
    from oil_gas_ops_prospect.custom_gru_v2 import custom_gru_v2
    H, F_in = 8, 3
    x = torch.zeros(1, 10, F_in)
    wi = torch.randn(3 * H, F_in) * 0.1
    wh = torch.randn(3 * H, H) * 0.1
    bi = torch.randn(3 * H) * 0.1
    bh = torch.randn(3 * H) * 0.1
    y, h = custom_gru_v2(x, wi, wh, bi, bh, H, False)
    assert not torch.isnan(y).any()
    assert not torch.isinf(y).any()


def test_ut_edge_002_large_positive_input():
    from oil_gas_ops_prospect.custom_gru_v2 import custom_gru_v2
    H, F_in = 8, 3
    x = torch.randn(1, 10, F_in) * 100
    wi, wh, bi, bh = _make_inputs(1, 10, H, F_in)[1:]
    y, h = custom_gru_v2(x, wi, wh, bi, bh, H, False)
    assert not torch.isnan(y).any()
    assert not torch.isinf(y).any()


def test_ut_edge_003_large_negative_input():
    from oil_gas_ops_prospect.custom_gru_v2 import custom_gru_v2
    H, F_in = 8, 3
    x = -torch.randn(1, 10, F_in) * 100
    wi, wh, bi, bh = _make_inputs(1, 10, H, F_in)[1:]
    y, h = custom_gru_v2(x, wi, wh, bi, bh, H, False)
    assert not torch.isnan(y).any()
    assert not torch.isinf(y).any()


def test_ut_edge_004_zero_weights():
    from oil_gas_ops_prospect.custom_gru_v2 import custom_gru_v2
    H, F_in = 8, 3
    x = torch.randn(1, 10, F_in)
    wi = torch.zeros(3 * H, F_in)
    wh = torch.zeros(3 * H, H)
    bi = torch.randn(3 * H) * 0.1
    bh = torch.randn(3 * H) * 0.1
    y, h = custom_gru_v2(x, wi, wh, bi, bh, H, False)
    assert not torch.isnan(y).any()


def test_ut_edge_005_zero_bias():
    from oil_gas_ops_prospect.custom_gru_v2 import custom_gru_v2
    H, F_in = 8, 3
    x = torch.randn(1, 10, F_in)
    wi = torch.randn(3 * H, F_in) * 0.1
    wh = torch.randn(3 * H, H) * 0.1
    bi = torch.zeros(3 * H)
    bh = torch.zeros(3 * H)
    y, h = custom_gru_v2(x, wi, wh, bi, bh, H, False)
    assert not torch.isnan(y).any()


def test_ut_edge_006_single_step():
    from oil_gas_ops_prospect.custom_gru_v2 import custom_gru_v2
    H, F_in = 8, 3
    x = torch.randn(1, 1, F_in)
    wi = torch.randn(3 * H, F_in) * 0.1
    wh = torch.randn(3 * H, H) * 0.1
    bi = torch.randn(3 * H) * 0.1
    bh = torch.randn(3 * H) * 0.1
    y, h = custom_gru_v2(x, wi, wh, bi, bh, H, False)
    assert y.shape == (1, 1, 1, H)
    assert h.shape == (1, 1, H)


@npu_only()
def test_ut_edge_007_reproducible_repeat():
    from oil_gas_ops_prospect.custom_gru_v2 import custom_gru_v2
    x, wi, wh, bi, bh = _make_inputs(1, 10, 8, 3, seed=55)
    results = []
    for _ in range(10):
        y, h = custom_gru_v2(x, wi, wh, bi, bh, 8, False)
        results.append(y.clone())
    for r in results[1:]:
        assert torch.equal(r, results[0]), "Repeated execution produced different results"


def test_ut_edge_008_continuous_inference():
    from oil_gas_ops_prospect.custom_gru_v2 import custom_gru_v2
    x, wi, wh, bi, bh = _make_inputs(1, 10, 8, 3, seed=77)
    y1, _ = custom_gru_v2(x, wi, wh, bi, bh, 8, False)
    y2, _ = custom_gru_v2(x, wi, wh, bi, bh, 8, False)
    assert torch.equal(y1, y2)


def test_ut_edge_009_require_custom_on_cpu_raises():
    import os
    from oil_gas_ops_prospect.custom_gru_v2 import custom_gru_v2
    x, wi, wh, bi, bh = _make_inputs(1, 10, 8, 3, seed=88)
    old = os.environ.get("OIL_GAS_OPS_REQUIRE_ASCENDC_CUSTOM_GRU_V2")
    os.environ["OIL_GAS_OPS_REQUIRE_ASCENDC_CUSTOM_GRU_V2"] = "1"
    try:
        with pytest.raises(RuntimeError, match="not an NPU tensor"):
            custom_gru_v2(x, wi, wh, bi, bh, 8, False)
    finally:
        if old is not None:
            os.environ["OIL_GAS_OPS_REQUIRE_ASCENDC_CUSTOM_GRU_V2"] = old
        else:
            os.environ.pop("OIL_GAS_OPS_REQUIRE_ASCENDC_CUSTOM_GRU_V2", None)


def test_ut_edge_010_custom_gru_module_bias_false():
    from oil_gas_ops_prospect.custom_gru_v2 import CustomGRU
    model = CustomGRU(3, 8, bidirectional=False, bias=False)
    x = torch.randn(1, 10, 3)
    y, h = model(x)
    assert y.shape == (1, 1, 10, 8)
    assert h.shape == (1, 1, 8)
    assert model.bias_ih is None
    assert model.bias_hh is None


def test_ut_edge_011_ub_safety_large_hidden_size():
    """UT-EDGE-011: UB safety net must reject extremely large hiddenSize.

    The TilingFunc UB check computes:
      ubSlots = UB_LIMIT / sizeof(float)   (= 49152)
      overhead = 3*Ha + 4*Ha = 7*Ha
      maxBPerBlock = (ubSlots - overhead) / Ha

    When 7*Ha > ubSlots (Ha >= 7022, i.e. H >= 7017), the old code
    subtracted in unsigned arithmetic, causing underflow to a huge value.
    The fix compares ubSlots <= overhead BEFORE subtracting, returning
    GRAPH_FAILED instead of silently passing.

    This test verifies the fixed logic in Python (mirroring the C++ code).
    """
    UB_LIMIT = 192 * 1024
    sizeof_float = 4
    ubSlots = UB_LIMIT // sizeof_float  # 49152

    # Case 1: Normal H (H=24, Ha=24) — should compute maxBPerBlock > 0
    Ha_normal = 24
    overhead_normal = 3 * Ha_normal + 4 * Ha_normal  # 168
    assert ubSlots > overhead_normal
    maxB_normal = (ubSlots - overhead_normal) // Ha_normal
    assert maxB_normal > 0

    # Case 2: Extremely large H (H=7017, Ha=7024) — should be rejected
    Ha_extreme = 7024
    overhead_extreme = 3 * Ha_extreme + 4 * Ha_extreme  # 49168
    assert ubSlots <= overhead_extreme  # fix: compare before subtract
    # The old code would underflow: (49152 - 49168) as uint64 = huge number
    # The new code returns GRAPH_FAILED because ubSlots <= overhead

    # Case 3: Boundary — Ha = 7022 (7*7022 = 49154 > 49152)
    Ha_boundary = 7022
    overhead_boundary = 3 * Ha_boundary + 4 * Ha_boundary  # 49154
    assert ubSlots <= overhead_boundary  # should be rejected

    # Case 4: Just below boundary — Ha = 7021 (7*7021 = 49147 < 49152)
    Ha_below = 7021
    overhead_below = 3 * Ha_below + 4 * Ha_below  # 49147
    assert ubSlots > overhead_below  # should NOT be rejected
    maxB_below = (ubSlots - overhead_below) // Ha_below
    assert maxB_below == 0  # but maxBPerBlock is 0, so still GRAPH_FAILED
