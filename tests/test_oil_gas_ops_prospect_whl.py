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

import importlib
import importlib.metadata
import importlib.util
import os
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]

# 源码树没有 `oil_gas_ops_prospect/` 目录，但 tests/conftest.py 会把源码挂到该导入名下。
# 专门验证 whl 的入口脚本置 OIL_GAS_OPS_WHL_TEST=1，此时不挂源码：
#   - 置了：解析到仓内路径就是硬失败；
#   - 没置（例如顺手跑 `pytest tests/`）：本模块跳过，不误报。
_WHEEL_VERIFICATION = os.environ.get("OIL_GAS_OPS_WHL_TEST") == "1"


def _package_origin() -> Path | None:
    spec = importlib.util.find_spec("oil_gas_ops_prospect.complex_mul")
    if spec is None or not spec.origin:
        return None
    return Path(spec.origin).resolve()


_ORIGIN = _package_origin()
_SHADOWED = _ORIGIN is not None and REPO_ROOT in _ORIGIN.parents

if not _WHEEL_VERIFICATION and (_ORIGIN is None or _SHADOWED):
    pytest.skip(
        "本文件验证的是已安装的 whl，当前 oil_gas_ops_prospect 未安装或被仓库内的"
        "源码树遮蔽。请用 `bash tests/run_whl_tests.sh`。",
        allow_module_level=True,
    )


def test_imports_resolve_to_the_installed_wheel():
    """守卫：本文件必须测已安装的 whl，而不是仓库里的源码树。"""
    assert _ORIGIN is not None, "oil_gas_ops_prospect 未安装，无法验证 whl"
    assert not _SHADOWED, (
        f"oil_gas_ops_prospect 解析到了源码树 {_ORIGIN}，而不是已安装的 whl。"
        f"请在仓库根之外的目录运行本用例。"
    )


def test_oil_gas_ops_prospect_distribution_and_public_imports():
    assert importlib.metadata.version("oil-gas-ops-prospect")

    from oil_gas_ops_prospect.complex_mul import complex_mul, torch_complex_mul, torch_complex_mul_grad
    from oil_gas_ops_prospect.complex_exp import complex_exp
    from oil_gas_ops_prospect.complex_reciprocal import complex_reciprocal
    from oil_gas_ops_prospect.conv3d_custom import Conv3d
    from oil_gas_ops_prospect.conv3d_custom.functional import conv3d
    from oil_gas_ops_prospect.fused_bias_softmax import (
        attention_core,
        fused_bias_softmax,
        fused_softmax_grad,
        torch_fused_bias_softmax,
        torch_fused_softmax_grad,
    )
    from oil_gas_ops_prospect.interpolate import bilinear, interpolate, trilinear
    from oil_gas_ops_prospect.real_fft import real_irfft, real_rfft

    assert callable(complex_mul)
    assert callable(torch_complex_mul)
    assert callable(torch_complex_mul_grad)
    assert callable(complex_reciprocal)
    assert callable(complex_exp)
    assert callable(Conv3d)
    assert callable(conv3d)
    assert callable(fused_bias_softmax)
    assert callable(fused_softmax_grad)
    assert callable(attention_core)
    assert callable(torch_fused_bias_softmax)
    assert callable(torch_fused_softmax_grad)
    assert callable(real_rfft)
    assert callable(real_irfft)
    assert callable(interpolate)
    assert callable(bilinear)
    assert callable(trilinear)


def test_removed_packages_are_not_in_wheel():
    """已下线的算子不应残留在 whl 里。"""
    for name in (
        "oil_gas_ops_prospect.ascendc",
        "oil_gas_ops_prospect.triton",
        "oil_gas_ops_prospect.ascendc.conv3d_gelu",
        "oil_gas_ops_prospect.triton.real_fft",
        "oil_gas_ops_prospect.triton.out_layer_3d_gelu",
        "oil_gas_ops_prospect.triton.axis_spectral",
    ):
        try:
            spec = importlib.util.find_spec(name)
        except ModuleNotFoundError:
            spec = None
        assert spec is None, f"{name} 仍在 whl 中"


def test_complex_mul_whl_cpu_fallback_forward_backward():
    """CPU 上走 torch 回退，验证 autograd 封装的前反向接线与数学一致性。"""
    from oil_gas_ops_prospect.complex_mul import complex_mul

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
    from oil_gas_ops_prospect.complex_mul import torch_complex_mul, torch_complex_mul_grad

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


def test_fused_bias_softmax_whl_cpu_fallback():
    """CPU 上走 torch 回退，验证融合 softmax 前反向与 attention_core 接线。"""
    from oil_gas_ops_prospect.fused_bias_softmax import (
        attention_core,
        fused_bias_softmax,
        fused_softmax_grad,
        torch_fused_bias_softmax,
        torch_fused_softmax_grad,
    )

    torch.manual_seed(13)
    x = torch.randn(2, 4, 8, 16)
    bias1 = torch.randn(2, 4, 1, 16)
    bias2 = torch.randn(2, 4, 8, 16)
    y = fused_bias_softmax(x, bias1, bias2)
    ref = torch_fused_bias_softmax(x, bias1, bias2)
    assert torch.allclose(y, ref, rtol=1e-6, atol=1e-6)

    probs = torch.softmax(torch.randn(2, 4, 8, 16), dim=-1)
    go = torch.randn(2, 4, 8, 32)
    values = torch.randn(2, 4, 16, 32)
    g = fused_softmax_grad(probs, go, values)
    g_ref = torch_fused_softmax_grad(probs, go, values)
    assert torch.allclose(g, g_ref, rtol=1e-6, atol=1e-6)

    q = torch.randn(2, 4, 8, 32, requires_grad=True)
    k = torch.randn(2, 4, 16, 32, requires_grad=True)
    v = torch.randn(2, 4, 16, 32, requires_grad=True)
    out = attention_core(q, k, v, bias1, bias2)
    assert out.shape == (2, 4, 8, 32)
    out.sum().backward()
    assert q.grad is not None and k.grad is not None and v.grad is not None


def test_real_fft_whl_cpu_fallback_matches_torch_fft():
    from oil_gas_ops_prospect.real_fft import real_irfft, real_rfft

    torch.manual_seed(12)
    x = torch.randn(2, 7, 3)
    xr, xi = real_rfft(x, dim=1, norm="ortho")
    ref = torch.fft.rfft(x, dim=1, norm="ortho")
    assert torch.allclose(xr, ref.real, rtol=1e-5, atol=1e-5)
    assert torch.allclose(xi, ref.imag, rtol=1e-5, atol=1e-5)

    y = real_irfft(xr, xi, n=x.shape[1], dim=1, norm="ortho")
    assert torch.allclose(y, x, rtol=1e-5, atol=1e-5)


def test_complex_reciprocal_whl_cpu_fallback_matches_torch():
    """CPU 上走 torch 回退，验证复数倒数算子的接线与 autograd。"""
    from oil_gas_ops_prospect.complex_reciprocal import complex_reciprocal

    torch.manual_seed(13)
    z = torch.randn(2, 7, 3) + 1j * torch.randn(2, 7, 3)
    assert torch.equal(complex_reciprocal(z), 1 / z)

    xr = torch.randn(2, 7, 3)
    xi = torch.randn(2, 7, 3)
    ref = 1 / torch.complex(xr, xi)
    out = complex_reciprocal(xr, xi)
    assert torch.allclose(out, ref, rtol=1e-6, atol=1e-6)
    out2 = complex_reciprocal(xr, -0.25)
    assert torch.allclose(out2, 1 / torch.complex(xr, torch.full_like(xr, -0.25)), rtol=1e-6, atol=1e-6)

    z_g = z.clone().requires_grad_(True)
    g = torch.randn_like(z_g.real) + 1j * torch.randn_like(z_g.real)
    out_g = complex_reciprocal(z_g)
    (out_g.real * g.real + out_g.imag * g.imag).sum().backward()
    ref_z = z.detach().clone().requires_grad_(True)
    ref_out = 1 / ref_z
    (ref_out.real * g.real + ref_out.imag * g.imag).sum().backward()
    assert torch.allclose(z_g.grad, ref_z.grad, rtol=1e-5, atol=1e-5)


def test_complex_exp_whl_cpu_fallback_matches_torch():
    """CPU 上走 torch 回退，验证复数指数算子的接线与 autograd。"""
    from oil_gas_ops_prospect.complex_exp import complex_exp

    torch.manual_seed(13)
    z = torch.randn(2, 7, 3) + 1j * torch.randn(2, 7, 3)
    assert torch.equal(complex_exp(z), torch.exp(z))

    z_g = z.clone().requires_grad_(True)
    g = torch.randn_like(z_g.real) + 1j * torch.randn_like(z_g.real)
    out = complex_exp(z_g)
    (out.real * g.real + out.imag * g.imag).sum().backward()
    ref_z = z.detach().clone().requires_grad_(True)
    ref = torch.exp(ref_z)
    (ref.real * g.real + ref.imag * g.imag).sum().backward()
    assert torch.allclose(z_g.grad, ref_z.grad, rtol=1e-5, atol=1e-5)


def test_conv3d_custom_whl_cpu_forward_backward():
    """CPU 上验证 Conv3d / conv3d 前反向接线与数学一致性。"""
    import torch.nn.functional as F

    from oil_gas_ops_prospect.conv3d_custom import Conv3d
    from oil_gas_ops_prospect.conv3d_custom.functional import conv3d

    torch.manual_seed(14)
    dtype = torch.float64
    x = torch.randn(1, 3, 6, 6, 6, dtype=dtype, requires_grad=True)
    w = torch.randn(4, 3, 3, 3, 3, dtype=dtype, requires_grad=True)
    out = conv3d(x, w, None, 1, 1, 1, 1)
    ref_leaves = [t.detach().clone().requires_grad_(True) for t in (x, w)]
    ref = F.conv3d(ref_leaves[0], ref_leaves[1], None, 1, 1, 1, 1)
    assert torch.allclose(out, ref, rtol=1e-6, atol=1e-6)

    gout = torch.randn_like(out)
    gx, gw = torch.autograd.grad(out, (x, w), gout)
    rgx, rgw = torch.autograd.grad(ref, ref_leaves, gout)
    assert torch.allclose(gx, rgx, rtol=1e-6, atol=1e-6)
    assert torch.allclose(gw, rgw, rtol=1e-6, atol=1e-6)

    m = Conv3d(3, 4, 3, padding=1).double()
    m.use_custom = True  # CPU 上强制走自定义前/反向路径
    xm = torch.randn(1, 3, 5, 5, 5, dtype=dtype, requires_grad=True)
    y = m(xm)
    y.sum().backward()
    assert y.shape == (1, 4, 5, 5, 5)
    assert xm.grad is not None and xm.grad.shape == xm.shape
    assert m.weight.grad is not None and m.weight.grad.shape == m.weight.shape


def test_complex_mul_npu_extension_import_when_required():
    if os.environ.get("OIL_GAS_OPS_WHL_REQUIRE_NPU_EXT") != "1":
        pytest.skip("set OIL_GAS_OPS_WHL_REQUIRE_NPU_EXT=1 to require the compiled NPU extension")

    ext = importlib.import_module("complex_mul_npu")
    assert callable(ext.complex_mul_forward)
    assert callable(ext.complex_mul_backward)
