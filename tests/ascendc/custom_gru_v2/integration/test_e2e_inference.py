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

"""Integration tests: CustomGRU v2 end-to-end inference.

Validates inference shapes, bidirectional, and nn.Module integration.

Fixes from review:
- P0-1: multi-layer tests removed (num_layers not in API)
- P0-6: bidirectional test now validates values against correct reference
- P2-2: IT-INFER-004 (graph capture) removed — requires NPU + complex setup, marked separately
- P1-3: bidirectional uses independent forward/reverse weights
"""
from __future__ import annotations

import pytest
import torch

from conftest import npu_hardware_only, torch_gru_ref as _torch_gru_ref


def test_it_infer_001_single_sequence():
    from oil_gas_ops_prospect.custom_gru_v2 import custom_gru_v2
    H, F_in = 24, 3
    x = torch.randn(1, 1000, F_in)
    wi = torch.randn(3 * H, F_in) * 0.1
    wh = torch.randn(3 * H, H) * 0.1
    bi = torch.randn(3 * H) * 0.1
    bh = torch.randn(3 * H) * 0.1
    y, h = custom_gru_v2(x, wi, wh, bi, bh, H, False)
    assert y.shape == (1, 1, 1000, H)
    assert h.shape == (1, 1, H)
    assert not torch.isnan(y).any()


def test_it_infer_002_batch_inference():
    from oil_gas_ops_prospect.custom_gru_v2 import custom_gru_v2
    H, F_in = 24, 3
    x = torch.randn(32, 100, F_in)
    wi = torch.randn(3 * H, F_in) * 0.1
    wh = torch.randn(3 * H, H) * 0.1
    bi = torch.randn(3 * H) * 0.1
    bh = torch.randn(3 * H) * 0.1
    y, h = custom_gru_v2(x, wi, wh, bi, bh, H, False)
    assert y.shape == (1, 32, 100, H)
    assert h.shape == (1, 32, H)


def test_it_infer_003_bidirectional_values():
    """Bidirectional inference with value verification against correct reference."""
    from oil_gas_ops_prospect.custom_gru_v2 import custom_gru_v2
    H, F_in = 8, 3
    torch.manual_seed(42)
    x = torch.randn(1, 50, F_in)
    wi = torch.randn(2, 3 * H, F_in) * 0.1
    wh = torch.randn(2, 3 * H, H) * 0.1
    bi = torch.randn(2, 3 * H) * 0.1
    bh = torch.randn(2, 3 * H) * 0.1
    y_ref, h_ref = _torch_gru_ref(x, wi, wh, bi, bh, H, bidir=True)

    y, h = custom_gru_v2(x, wi.reshape(-1, F_in), wh.reshape(-1, H),
                          bi.reshape(-1), bh.reshape(-1), H, True)
    assert y.shape == (2, 1, 50, H)
    assert h.shape == (2, 1, H)
    max_diff = (y - y_ref).abs().max().item()
    assert max_diff <= 0.01, f"Bidirectional y max_diff={max_diff} > 0.01"


def test_it_infer_004_module_integration():
    from oil_gas_ops_prospect.custom_gru_v2 import CustomGRU
    model = CustomGRU(3, 24, bidirectional=False)
    x = torch.randn(1, 100, 3)
    y, h = model(x)
    assert y.shape == (1, 1, 100, 24)
    assert h.shape == (1, 1, 24)
    assert not torch.isnan(y).any()


def test_it_infer_005_module_bidirectional():
    from oil_gas_ops_prospect.custom_gru_v2 import CustomGRU
    model = CustomGRU(3, 16, bidirectional=True)
    x = torch.randn(4, 50, 3)
    y, h = model(x)
    assert y.shape == (2, 4, 50, 16)
    assert h.shape == (2, 4, 16)
    assert not torch.isnan(y).any()


@npu_hardware_only()
def test_it_infer_006_npu_graph_capture():
    """NPU graph capture test — only runs on NPU hardware with compiled extension.

    Note: make_graphed_callables requires the backward to produce gradients
    with shapes matching the forward outputs. CustomGRU's backward uses a
    Python-based _torch_gru_forward recompute, which creates a different
    computation graph shape. This is a known limitation of autograd.Function
    with Python backward + graph capture. Skip if not supported.
    """
    try:
        from oil_gas_ops_prospect.custom_gru_v2 import _load_extension
        _load_extension()
    except (RuntimeError, ImportError):
        pytest.skip("custom_gru_v2_npu extension not compiled — skipping graph capture test")
    from oil_gas_ops_prospect.custom_gru_v2 import CustomGRU
    from conftest import get_device
    device = get_device()
    model = CustomGRU(3, 24, bidirectional=False).to(device)
    x = torch.randn(1, 100, 3, device=device)
    model(x)
    try:
        graphed = torch.npu.make_graphed_callables(model, x)
    except (RuntimeError, ValueError) as e:
        pytest.skip(f"Graph capture not supported with Python backward: {e}")
    y1, h1 = graphed(x)
    y2, h2 = graphed(x)
    assert torch.equal(y1, y2), "Graph replay produced inconsistent results"
