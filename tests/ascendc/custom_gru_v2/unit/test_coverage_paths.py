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

"""Targeted coverage tests for custom_gru_v2.py.

Ensures every executable code path in the Python wrapper is exercised:
- CPU fallback paths (bias=None, 3D weights with unidir, _torch_gru_forward direct calls)
- NPU-specific paths (extension load, autograd.Function forward/backward, mock import failure)
- CustomGRU module paths (bias=True/False, bidirectional combinations)

These tests complement the existing unit/integration/performance suites to achieve
100% line coverage in a dual-environment (CPU+NPU) combine strategy.
"""
from __future__ import annotations

import os

import pytest
import torch

from conftest import npu_only, get_device


# ============================================================================
# CPU-reachable paths: custom_gru_v2() with bias=None
# ============================================================================

def test_cov_bias_ih_none():
    """Cover custom_gru_v2() line 145-146: bias_ih is None → zeros."""
    from oil_gas_ops_prospect.custom_gru_v2 import custom_gru_v2
    H, F_in = 8, 3
    x = torch.randn(1, 10, F_in)
    wi = torch.randn(3 * H, F_in) * 0.1
    wh = torch.randn(3 * H, H) * 0.1
    bh = torch.randn(3 * H) * 0.1
    y, h = custom_gru_v2(x, wi, wh, None, bh, H, False)
    assert y.shape == (1, 1, 10, H)
    assert not torch.isnan(y).any()


def test_cov_bias_hh_none():
    """Cover custom_gru_v2() line 147-148: bias_hh is None → zeros."""
    from oil_gas_ops_prospect.custom_gru_v2 import custom_gru_v2
    H, F_in = 8, 3
    x = torch.randn(1, 10, F_in)
    wi = torch.randn(3 * H, F_in) * 0.1
    wh = torch.randn(3 * H, H) * 0.1
    bi = torch.randn(3 * H) * 0.1
    y, h = custom_gru_v2(x, wi, wh, bi, None, H, False)
    assert y.shape == (1, 1, 10, H)
    assert not torch.isnan(y).any()


def test_cov_both_bias_none():
    """Cover both bias_ih and bias_hh None simultaneously."""
    from oil_gas_ops_prospect.custom_gru_v2 import custom_gru_v2
    H, F_in = 8, 3
    x = torch.randn(1, 10, F_in)
    wi = torch.randn(3 * H, F_in) * 0.1
    wh = torch.randn(3 * H, H) * 0.1
    y, h = custom_gru_v2(x, wi, wh, None, None, H, False)
    assert y.shape == (1, 1, 10, H)
    assert not torch.isnan(y).any()


def test_cov_both_bias_none_bidirectional():
    """Cover bias=None with bidirectional=True."""
    from oil_gas_ops_prospect.custom_gru_v2 import custom_gru_v2
    H, F_in = 8, 3
    x = torch.randn(1, 10, F_in)
    wi = torch.randn(2, 3 * H, F_in) * 0.1
    wh = torch.randn(2, 3 * H, H) * 0.1
    y, h = custom_gru_v2(x, wi.reshape(-1, F_in), wh.reshape(-1, H), None, None, H, True)
    assert y.shape == (2, 1, 10, H)
    assert not torch.isnan(y).any()


# ============================================================================
# CPU-reachable paths: _torch_gru_forward() direct calls
# ============================================================================

def test_cov_torch_gru_forward_none_bias():
    """Cover _torch_gru_forward() line 58-59: bi_list[d] is None → zeros."""
    from oil_gas_ops_prospect.custom_gru_v2 import _torch_gru_forward
    H, F_in = 8, 3
    x = torch.randn(1, 10, F_in)
    wi = torch.randn(3 * H, F_in) * 0.1
    wh = torch.randn(3 * H, H) * 0.1
    y, h = _torch_gru_forward(x, wi, wh, None, None, H, bidirectional=False)
    assert y.shape == (1, 1, 10, H)
    assert not torch.isnan(y).any()


def test_cov_torch_gru_forward_none_bias_bidir():
    """Cover _torch_gru_forward() with None bias and bidirectional."""
    from oil_gas_ops_prospect.custom_gru_v2 import _torch_gru_forward
    H, F_in = 8, 3
    x = torch.randn(1, 10, F_in)
    wi = torch.randn(2, 3 * H, F_in) * 0.1
    wh = torch.randn(2, 3 * H, H) * 0.1
    y, h = _torch_gru_forward(x, wi, wh, None, None, H, bidirectional=True)
    assert y.shape == (2, 1, 10, H)
    assert not torch.isnan(y).any()


def test_cov_torch_gru_forward_3d_weights_unidir():
    """Cover _torch_gru_forward() weight_ih.dim()==3 with bidirectional=False."""
    from oil_gas_ops_prospect.custom_gru_v2 import _torch_gru_forward
    H, F_in = 8, 3
    x = torch.randn(1, 10, F_in)
    wi = torch.randn(1, 3 * H, F_in) * 0.1
    wh = torch.randn(1, 3 * H, H) * 0.1
    bi = torch.randn(1, 3 * H) * 0.1
    bh = torch.randn(1, 3 * H) * 0.1
    y, h = _torch_gru_forward(x, wi, wh, bi, bh, H, bidirectional=False)
    assert y.shape == (1, 1, 10, H)


def test_cov_torch_gru_forward_2d_weights_bidir():
    """Cover _torch_gru_forward() weight_ih.dim()==2 with bidirectional=True."""
    from oil_gas_ops_prospect.custom_gru_v2 import _torch_gru_forward
    H, F_in = 8, 3
    x = torch.randn(1, 10, F_in)
    wi = torch.randn(3 * H, F_in) * 0.1
    wh = torch.randn(3 * H, H) * 0.1
    bi = torch.randn(3 * H) * 0.1
    bh = torch.randn(3 * H) * 0.1
    y, h = _torch_gru_forward(x, wi, wh, bi, bh, H, bidirectional=True)
    assert y.shape == (2, 1, 10, H)


# ============================================================================
# CPU-reachable paths: CustomGRU module combinations
# ============================================================================

def test_cov_custom_gru_bias_true_bidir():
    """Cover CustomGRU(bias=True, bidirectional=True) forward path."""
    from oil_gas_ops_prospect.custom_gru_v2 import CustomGRU
    model = CustomGRU(3, 8, bidirectional=True, bias=True)
    assert model.bias_ih is not None
    assert model.bias_hh is not None
    x = torch.randn(2, 10, 3)
    y, h = model(x)
    assert y.shape == (2, 2, 10, 8)
    assert h.shape == (2, 2, 8)


def test_cov_custom_gru_bias_false_unidir():
    """Cover CustomGRU(bias=False, bidirectional=False) forward path."""
    from oil_gas_ops_prospect.custom_gru_v2 import CustomGRU
    model = CustomGRU(3, 8, bidirectional=False, bias=False)
    assert model.bias_ih is None
    assert model.bias_hh is None
    x = torch.randn(1, 10, 3)
    y, h = model(x)
    assert y.shape == (1, 1, 10, 8)
    assert h.shape == (1, 1, 8)


def test_cov_custom_gru_init_weights():
    """Cover _init_weights: weight gets xavier, bias gets zeros."""
    from oil_gas_ops_prospect.custom_gru_v2 import CustomGRU
    model = CustomGRU(3, 8, bidirectional=False, bias=True)
    assert model.weight_ih.std().item() > 0, "weight_ih not initialized (xavier)"
    assert model.bias_ih.abs().sum().item() == 0, "bias_ih not zeroed"


def test_cov_custom_gru_trainable():
    """Cover CustomGRU in a training step with loss.backward()."""
    from oil_gas_ops_prospect.custom_gru_v2 import CustomGRU
    model = CustomGRU(3, 8, bidirectional=False, bias=True)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    x = torch.randn(1, 10, 3)
    target = torch.randn(1, 1, 10, 8)
    opt.zero_grad()
    y, h = model(x)
    loss = ((y - target) ** 2).mean()
    loss.backward()
    opt.step()
    assert loss.item() > 0


# ============================================================================
# CPU-reachable paths: _require_custom and _is_npu_tensor
# ============================================================================

def test_cov_require_custom_false_default():
    """Cover _require_custom() returning False (default env)."""
    from oil_gas_ops_prospect.custom_gru_v2 import _require_custom, custom_gru_v2
    old = os.environ.get("OIL_GAS_OPS_REQUIRE_ASCENDC_CUSTOM_GRU_V2")
    os.environ.pop("OIL_GAS_OPS_REQUIRE_ASCENDC_CUSTOM_GRU_V2", None)
    try:
        assert _require_custom() is False
        x = torch.randn(1, 5, 3)
        wi = torch.randn(24, 3) * 0.1
        wh = torch.randn(24, 8) * 0.1
        bi = torch.randn(24) * 0.1
        bh = torch.randn(24) * 0.1
        y, h = custom_gru_v2(x, wi, wh, bi, bh, 8, False)
        assert y.shape == (1, 1, 5, 8)
    finally:
        if old is not None:
            os.environ["OIL_GAS_OPS_REQUIRE_ASCENDC_CUSTOM_GRU_V2"] = old
        else:
            os.environ.pop("OIL_GAS_OPS_REQUIRE_ASCENDC_CUSTOM_GRU_V2", None)


def test_cov_is_npu_tensor_cpu():
    """Cover _is_npu_tensor() returning False on CPU."""
    from oil_gas_ops_prospect.custom_gru_v2 import _is_npu_tensor
    x = torch.randn(1, 5, 3)
    assert _is_npu_tensor(x) is False


# ============================================================================
# NPU-only paths: _CustomGRUV2Function, extension loading, mock failures
# ============================================================================

@npu_only()
def test_cov_npu_forward_function():
    """Cover _CustomGRUV2Function.forward() on NPU."""
    from oil_gas_ops_prospect.custom_gru_v2 import custom_gru_v2
    device = get_device()
    H, F_in = 8, 3
    x = torch.randn(1, 10, F_in, device=device)
    wi = torch.randn(3 * H, F_in, device=device) * 0.1
    wh = torch.randn(3 * H, H, device=device) * 0.1
    bi = torch.randn(3 * H, device=device) * 0.1
    bh = torch.randn(3 * H, device=device) * 0.1
    y, h = custom_gru_v2(x, wi, wh, bi, bh, H, False)
    assert y.shape == (1, 1, 10, H)
    assert not torch.isnan(y).any()


@npu_only()
def test_cov_npu_forward_bidirectional():
    """Cover _CustomGRUV2Function.forward() with bidirectional=True on NPU."""
    from oil_gas_ops_prospect.custom_gru_v2 import custom_gru_v2
    device = get_device()
    H, F_in = 8, 3
    x = torch.randn(1, 10, F_in, device=device)
    wi = torch.randn(2, 3 * H, F_in, device=device) * 0.1
    wh = torch.randn(2, 3 * H, H, device=device) * 0.1
    bi = torch.randn(2, 3 * H, device=device) * 0.1
    bh = torch.randn(2, 3 * H, device=device) * 0.1
    y, h = custom_gru_v2(x, wi.reshape(-1, F_in), wh.reshape(-1, H),
                          bi.reshape(-1), bh.reshape(-1), H, True)
    assert y.shape == (2, 1, 10, H)


@npu_only()
def test_cov_npu_backward_function():
    """Cover _CustomGRUV2Function.backward() on NPU."""
    from oil_gas_ops_prospect.custom_gru_v2 import custom_gru_v2
    device = get_device()
    H, F_in = 8, 3
    x = torch.randn(1, 10, F_in, device=device, requires_grad=True)
    wi = (torch.randn(3 * H, F_in, device=device) * 0.1).requires_grad_(True)
    wh = (torch.randn(3 * H, H, device=device) * 0.1).requires_grad_(True)
    bi = (torch.randn(3 * H, device=device) * 0.1).requires_grad_(True)
    bh = (torch.randn(3 * H, device=device) * 0.1).requires_grad_(True)
    y, h = custom_gru_v2(x, wi, wh, bi, bh, H, False)
    (y.sum() + h.sum()).backward()
    assert x.grad is not None
    assert wi.grad is not None


@npu_only()
def test_cov_npu_backward_bias_false():
    """Cover _CustomGRUV2Function.backward() with bias not requiring grad."""
    from oil_gas_ops_prospect.custom_gru_v2 import CustomGRU
    device = get_device()
    model = CustomGRU(3, 8, bidirectional=False, bias=False).to(device)
    x = torch.randn(1, 10, 3, device=device)
    y, h = model(x)
    y.sum().backward()
    assert model.weight_ih.grad is not None


@npu_only()
def test_cov_npu_load_extension_fail_require():
    """Cover NPU path: _load_extension() fails + require_custom → RuntimeError."""
    import importlib
    mod = importlib.import_module('oil_gas_ops_prospect.custom_gru_v2')
    device = get_device()
    x = torch.randn(1, 5, 3, device=device)
    wi = torch.randn(24, 3, device=device) * 0.1
    wh = torch.randn(24, 8, device=device) * 0.1
    bi = torch.randn(24, device=device) * 0.1
    bh = torch.randn(24, device=device) * 0.1

    old_ext = mod._load_extension
    old_env = os.environ.get("OIL_GAS_OPS_REQUIRE_ASCENDC_CUSTOM_GRU_V2")
    os.environ["OIL_GAS_OPS_REQUIRE_ASCENDC_CUSTOM_GRU_V2"] = "1"
    mod._load_extension = mod._load_extension.__wrapped__ if hasattr(mod._load_extension, '__wrapped__') else mod._load_extension

    def _fail():
        raise ImportError("mocked")
    mod._load_extension = _fail
    try:
        with pytest.raises(RuntimeError, match="required but could not be imported"):
            mod.custom_gru_v2(x, wi, wh, bi, bh, 8, False)
    finally:
        mod._load_extension = old_ext
        if old_env is not None:
            os.environ["OIL_GAS_OPS_REQUIRE_ASCENDC_CUSTOM_GRU_V2"] = old_env
        else:
            os.environ.pop("OIL_GAS_OPS_REQUIRE_ASCENDC_CUSTOM_GRU_V2", None)


@npu_only()
def test_cov_npu_load_extension_fail_fallback():
    """Cover NPU path: _load_extension() fails + not require_custom → CPU fallback."""
    import importlib
    mod = importlib.import_module('oil_gas_ops_prospect.custom_gru_v2')
    device = get_device()
    x = torch.randn(1, 5, 3, device=device)
    wi = torch.randn(24, 3, device=device) * 0.1
    wh = torch.randn(24, 8, device=device) * 0.1
    bi = torch.randn(24, device=device) * 0.1
    bh = torch.randn(24, device=device) * 0.1

    old_ext = mod._load_extension
    old_env = os.environ.get("OIL_GAS_OPS_REQUIRE_ASCENDC_CUSTOM_GRU_V2")
    os.environ.pop("OIL_GAS_OPS_REQUIRE_ASCENDC_CUSTOM_GRU_V2", None)

    def _fail():
        raise ImportError("mocked")
    mod._load_extension = _fail
    try:
        y, h = mod.custom_gru_v2(x, wi, wh, bi, bh, 8, False)
        assert y.shape == (1, 1, 5, 8)
        assert not torch.isnan(y).any()
    finally:
        mod._load_extension = old_ext
        if old_env is not None:
            os.environ["OIL_GAS_OPS_REQUIRE_ASCENDC_CUSTOM_GRU_V2"] = old_env
        else:
            os.environ.pop("OIL_GAS_OPS_REQUIRE_ASCENDC_CUSTOM_GRU_V2", None)


@npu_only()
def test_cov_npu_is_npu_tensor_true():
    """Cover _is_npu_tensor() returning True on NPU."""
    from oil_gas_ops_prospect.custom_gru_v2 import _is_npu_tensor
    device = get_device()
    x = torch.randn(1, 5, 3, device=device)
    assert _is_npu_tensor(x) is True


@npu_only()
def test_cov_npu_custom_gru_module():
    """Cover CustomGRU module forward on NPU."""
    from oil_gas_ops_prospect.custom_gru_v2 import CustomGRU
    device = get_device()
    model = CustomGRU(3, 8, bidirectional=False, bias=True).to(device)
    x = torch.randn(1, 10, 3, device=device)
    y, h = model(x)
    assert y.shape == (1, 1, 10, 8)


@npu_only()
def test_cov_npu_custom_gru_module_bias_false():
    """Cover CustomGRU(bias=False) forward on NPU."""
    from oil_gas_ops_prospect.custom_gru_v2 import CustomGRU
    device = get_device()
    model = CustomGRU(3, 8, bidirectional=True, bias=False).to(device)
    x = torch.randn(1, 10, 3, device=device)
    y, h = model(x)
    assert y.shape == (2, 1, 10, 8)
