# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import os

import pytest
import torch


def test_oil_gas_ops_prospect_distribution_and_public_imports():
    assert importlib.metadata.version("oil-gas-ops-prospect")

    from oil_gas_ops_prospect.ascendc import (
        attention_core,
        complex_mul,
        fused_bias_softmax,
        fused_softmax_grad,
        torch_complex_mul,
        torch_complex_mul_grad,
        torch_fused_bias_softmax,
        torch_fused_softmax_grad,
    )
    from oil_gas_ops_prospect.triton.real_fft import real_irfft, real_rfft

    assert callable(complex_mul)
    assert callable(torch_complex_mul)
    assert callable(torch_complex_mul_grad)
    assert callable(fused_bias_softmax)
    assert callable(fused_softmax_grad)
    assert callable(attention_core)
    assert callable(torch_fused_bias_softmax)
    assert callable(torch_fused_softmax_grad)
    assert callable(real_rfft)
    assert callable(real_irfft)


def test_removed_packages_are_not_in_wheel():
    """已下线的算子不应残留在 whl 里。"""
    for name in (
        "oil_gas_ops_prospect.ascendc.conv3d_gelu",
        "oil_gas_ops_prospect.ascendc.out_layer_3d_gelu",
        "oil_gas_ops_prospect.triton.out_layer_3d_gelu",
        "oil_gas_ops_prospect.triton.axis_spectral",
    ):
        assert importlib.util.find_spec(name) is None, f"{name} 仍在 whl 中"


def test_complex_mul_whl_cpu_fallback_forward_backward():
    """CPU 上走 torch 回退，验证 autograd 封装的前反向接线与数学一致性。"""
    from oil_gas_ops_prospect.ascendc import complex_mul

    torch.manual_seed(11)
    shape = (2, 3, 4, 5)
    leaves = [torch.randn(shape, requires_grad=True) for _ in range(4)]
    ur, ui, vr, vi = leaves

    wr, wi = complex_mul(ur, ui, vr, vi)
    assert wr.shape == shape and wi.shape == shape

    # 前向对齐 torch 复数乘
    ref = torch.complex(ur.detach(), ui.detach()) * torch.complex(vr.detach(), vi.detach())
    assert torch.allclose(wr, ref.real, rtol=1e-6, atol=1e-6)
    assert torch.allclose(wi, ref.imag, rtol=1e-6, atol=1e-6)

    # 反向对齐 autograd
    gwr = torch.randn(shape)
    gwi = torch.randn(shape)
    (wr * gwr + wi * gwi).sum().backward()

    ref_leaves = [t.detach().clone().requires_grad_(True) for t in leaves]
    rur, rui, rvr, rvi = ref_leaves
    ref_wr = rur * rvr - rui * rvi
    ref_wi = rur * rvi + rui * rvr
    (ref_wr * gwr + ref_wi * gwi).sum().backward()

    for got, expected, name in zip(leaves, ref_leaves, ("ur", "ui", "vr", "vi")):
        assert got.grad is not None, f"grad[{name}] 为空"
        assert torch.allclose(got.grad, expected.grad, rtol=1e-6, atol=1e-6), f"grad[{name}] 不一致"


def test_complex_mul_grad_reference_matches_autograd():
    """torch_complex_mul_grad 的闭式解必须与 autograd 结果一致。"""
    from oil_gas_ops_prospect.ascendc import torch_complex_mul, torch_complex_mul_grad

    torch.manual_seed(12)
    shape = (3, 7)
    ur, ui, vr, vi = (torch.randn(shape) for _ in range(4))
    gwr, gwi = torch.randn(shape), torch.randn(shape)

    closed = torch_complex_mul_grad(gwr, gwi, ur, ui, vr, vi)

    leaves = [t.detach().clone().requires_grad_(True) for t in (ur, ui, vr, vi)]
    wr, wi = torch_complex_mul(*leaves)
    autograd = torch.autograd.grad((wr, wi), leaves, grad_outputs=(gwr, gwi))

    for got, expected, name in zip(closed, autograd, ("gur", "gui", "gvr", "gvi")):
        assert torch.allclose(got, expected, rtol=1e-6, atol=1e-6), f"{name} 不一致"


def test_real_fft_whl_cpu_fallback_matches_torch_fft():
    from oil_gas_ops_prospect.triton.real_fft import real_irfft, real_rfft

    torch.manual_seed(12)
    x = torch.randn(2, 7, 3)
    xr, xi = real_rfft(x, dim=1, norm="ortho")
    ref = torch.fft.rfft(x, dim=1, norm="ortho")
    assert torch.allclose(xr, ref.real, rtol=1e-5, atol=1e-5)
    assert torch.allclose(xi, ref.imag, rtol=1e-5, atol=1e-5)

    y = real_irfft(xr, xi, n=x.shape[1], dim=1, norm="ortho")
    assert torch.allclose(y, x, rtol=1e-5, atol=1e-5)


def test_complex_mul_npu_extension_import_when_required():
    if os.environ.get("OIL_GAS_OPS_WHL_REQUIRE_NPU_EXT") != "1":
        pytest.skip("set OIL_GAS_OPS_WHL_REQUIRE_NPU_EXT=1 to require the compiled NPU extension")

    ext = importlib.import_module("complex_mul_npu")
    assert callable(ext.complex_mul_forward)
    assert callable(ext.complex_mul_backward)
