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

import importlib
import sys
from pathlib import Path

import pytest
import torch

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import _bootstrap  # noqa: F401, E402  (side effect: 源码树里解析 oil_gas_ops_prospect)
from oil_gas_ops_prospect.complex_reciprocal import (  # noqa: E402
    complex_reciprocal,
)
from oil_gas_ops_prospect.complex_reciprocal import (  # noqa: E402
    triton_complex_reciprocal as recip_impl,
)

# 本环境的 torch（torch_npu 2.7.1）对 complex 张量调用 .double() 会静默丢弃虚部，
# 比较时必须用 .to(torch.complex128)。
REL_TOL = 1e-6

# 业务大形状（profiler 抓取，304x304x176 complex64，16.3M 元素）
PROD_SHAPE = (304, 304, 176)


def _maybe_npu_device():
    try:
        importlib.import_module("torch_npu")
    except ImportError:
        return None
    if hasattr(torch, "npu") and torch.npu.is_available():
        return torch.device("npu:0")
    return None


def _to_cpu(z: torch.Tensor) -> torch.Tensor:
    return z.detach().cpu().to(torch.complex128)


def _to_npu(z: torch.Tensor, device) -> torch.Tensor:
    flat = z.detach().contiguous().reshape(-1).view(torch.float32).to(device)
    return flat.view(torch.complex64).reshape(z.shape)


def _relerr(got: torch.Tensor, ref: torch.Tensor) -> float:
    got = _to_cpu(got)
    ref = ref.to(torch.complex128)
    diff = (got - ref).abs()
    return (diff / ref.abs().clamp_min(1e-300)).max().item()


# --------------------------------------------------------------------------- #
# CPU 回退：验证 API 接线与参考实现（kernel 数值必须上卡验证，见 NPU 用例）
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "shape", [(13,), (2, 3, 5), (0,), (1,), (1000,)]
)
def test_cpu_fallback_matches_torch_reciprocal(shape):
    torch.manual_seed(0)
    z = torch.randn(shape) + 1j * torch.randn(shape)
    assert torch.equal(complex_reciprocal(z), 1 / z)
    zt = z.t().conj() if z.dim() == 2 else z  # 非连续输入
    assert torch.allclose(complex_reciprocal(zt), 1 / zt, rtol=0, atol=0)


@pytest.mark.parametrize(
    "xr_type,xi_type", [("tensor", "scalar"), ("scalar", "tensor"), ("tensor", "tensor"),
                        ("tensor", "0dim"), ("0dim", "tensor"), ("0dim", "0dim")]
)
def test_cpu_fallback_matches_torch_reciprocal_parts(xr_type, xi_type):
    torch.manual_seed(1)
    n = 17
    xr = torch.randn(n) if xr_type == "tensor" else (
        torch.randn(()) if xr_type == "0dim" else 0.5)
    xi = torch.randn(n) if xi_type == "tensor" else (
        torch.randn(()) if xi_type == "0dim" else -0.25)
    out = complex_reciprocal(xr, xi)
    xr_b = xr if xr_type == "tensor" else torch.full((n,), float(xr))
    xi_b = xi if xi_type == "tensor" else torch.full((n,), float(xi))
    ref = 1 / torch.complex(xr_b, xi_b)
    assert torch.allclose(out, ref, rtol=1e-7, atol=1e-7)
    assert out.dtype == torch.complex64


def test_cpu_fallback_autograd():
    torch.manual_seed(2)
    a = torch.randn(9, requires_grad=True)
    b = torch.randn(9, requires_grad=True)
    u = torch.randn(9)
    v = torch.randn(9)
    z = torch.complex(a, b)
    out = complex_reciprocal(z)
    (out.real * u + out.imag * v).sum().backward()

    ref_a = a.detach().clone().requires_grad_(True)
    ref_b = b.detach().clone().requires_grad_(True)
    ref = 1 / torch.complex(ref_a, ref_b)
    (ref.real * u + ref.imag * v).sum().backward()
    assert torch.allclose(a.grad, ref_a.grad, rtol=1e-6, atol=1e-6)
    assert torch.allclose(b.grad, ref_b.grad, rtol=1e-6, atol=1e-6)

    xr = torch.randn(9, requires_grad=True)
    xi = torch.randn(9, requires_grad=True)
    out2 = complex_reciprocal(xr, xi)
    (out2.real * u + out2.imag * v).sum().backward()
    ref_xr = xr.detach().clone().requires_grad_(True)
    ref_xi = xi.detach().clone().requires_grad_(True)
    ref2 = 1 / torch.complex(ref_xr, ref_xi)
    (ref2.real * u + ref2.imag * v).sum().backward()
    assert torch.allclose(xr.grad, ref_xr.grad, rtol=1e-6, atol=1e-6)
    assert torch.allclose(xi.grad, ref_xi.grad, rtol=1e-6, atol=1e-6)


def test_reciprocal_parts_input_validation():
    with pytest.raises(TypeError):
        complex_reciprocal(0.5, -0.25)
    with pytest.raises(ValueError):
        complex_reciprocal(torch.randn(4), torch.randn(5))
    with pytest.raises(ValueError):
        complex_reciprocal(torch.randn(4), torch.randn(5, dtype=torch.float64))
    # 单元素张量可广播
    out = complex_reciprocal(torch.randn(4), torch.tensor(0.5))
    assert out.shape == (4,)


def member_of(name: str):
    """按名字取实现模块的成员（如 ``_resolve_block``）。

    下划线前缀是实现细节，测试是它们的门禁；直接书写 ``recip_impl._xxx``
    会触发 protected-access 告警，按仓库 ``kernel_of``（aot_compile_check.py）
    的先例用 ``getattr`` 按名访问。
    """
    return getattr(recip_impl, name)


def test_resolve_block_clamps_to_power_of_two(monkeypatch):
    resolve_block = member_of("_resolve_block")
    assert resolve_block(None) == 2048
    assert resolve_block(100) == 64
    assert resolve_block(3000) == 2048
    assert resolve_block(8) == 16
    monkeypatch.setenv("OIL_GAS_OPS_RECIP_BLOCK", "256")
    assert resolve_block(None) == 256
    assert resolve_block(1024) == 1024  # 显式参数优先


# --------------------------------------------------------------------------- #
# NPU：kernel 数值验证（前向 vs float64 参考、反向 vs autograd）
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "shape", [(1,), (13,), (2, 3, 5), (1000001,), PROD_SHAPE]
)
@pytest.mark.parametrize("stable", [True, False])
def test_npu_forward_matches_float64_reference(shape, stable):
    device = _maybe_npu_device()
    if device is None:
        pytest.skip("NPU is not available")

    torch.manual_seed(3)
    zc = (torch.randn(shape) + 1j * torch.randn(shape)).to(torch.complex64)
    ref = 1 / zc.to(torch.complex128)
    z = _to_npu(zc, device)
    out = complex_reciprocal(z, stable=stable)
    torch.npu.synchronize()
    assert out.dtype == torch.complex64
    assert tuple(out.shape) == shape
    assert _relerr(out, ref) < REL_TOL


def test_npu_forward_noncontiguous_and_empty():
    device = _maybe_npu_device()
    if device is None:
        pytest.skip("NPU is not available")

    torch.manual_seed(4)
    zc = (torch.randn(6, 8) + 1j * torch.randn(6, 8)).to(torch.complex64)
    ref = 1 / zc.to(torch.complex128)
    z = _to_npu(zc, device).t()  # 非连续
    out = complex_reciprocal(z)
    torch.npu.synchronize()
    assert _relerr(out.t(), ref) < REL_TOL

    out_empty = complex_reciprocal(torch.empty(0, dtype=torch.complex64, device=device))
    assert out_empty.numel() == 0


@pytest.mark.parametrize("scale", [1e-20, 1e-10, 1.0, 1e10, 1e20])
def test_npu_forward_stable_at_extreme_magnitudes(scale):
    """naive 的 a^2+b^2 在 |z| 超出 ~[1e-19, 1.8e19] 时上/下溢，stable 保持正确。"""
    device = _maybe_npu_device()
    if device is None:
        pytest.skip("NPU is not available")

    torch.manual_seed(5)
    zc = (scale * (torch.randn(4096) + 1j * torch.randn(4096))).to(torch.complex64)
    ref = 1 / zc.to(torch.complex128)
    out = complex_reciprocal(_to_npu(zc, device))
    torch.npu.synchronize()
    assert _relerr(out, ref) < REL_TOL


def test_npu_backward_matches_torch_grad_wrt_z():
    device = _maybe_npu_device()
    if device is None:
        pytest.skip("NPU is not available")

    torch.manual_seed(6)
    n = 100003
    zc = (torch.randn(n) + 1j * torch.randn(n)).to(torch.complex64)
    u, v = torch.randn(n), torch.randn(n)

    z_ref = zc.clone().requires_grad_(True)
    g_ref = torch.autograd.grad(1 / z_ref, z_ref, grad_outputs=torch.complex(u, v))[0]

    z_npu = _to_npu(zc, device).requires_grad_(True)
    g_npu = torch.complex(u, v).to(device)
    out = complex_reciprocal(z_npu)
    gz = torch.autograd.grad(out, z_npu, grad_outputs=g_npu)[0]
    torch.npu.synchronize()
    assert _relerr(gz, g_ref) < REL_TOL


def test_npu_backward_end_to_end_matches_float64():
    """经 torch.complex(a, b) 构造的端到端反传（float32 消放大约 1e-3 量级）。"""
    device = _maybe_npu_device()
    if device is None:
        pytest.skip("NPU is not available")

    torch.manual_seed(7)
    n = 8192
    a, b = torch.randn(n), torch.randn(n)
    u, v = torch.randn(n), torch.randn(n)

    ad = a.double().requires_grad_(True)
    bd = b.double().requires_grad_(True)
    ref = 1 / torch.complex(ad, bd)
    (ref.real * u.double() + ref.imag * v.double()).sum().backward()

    a_npu = a.to(device).requires_grad_(True)
    b_npu = b.to(device).requires_grad_(True)
    z = torch.complex(a_npu, b_npu)
    out = complex_reciprocal(z)
    g = torch.complex(u, v).to(device)
    (out * g.conj()).real.sum().backward()
    torch.npu.synchronize()
    rel_a = ((a_npu.grad.cpu() - ad.grad.float()).abs() / ad.grad.float().abs().clamp_min(1e-30)).max().item()
    rel_b = ((b_npu.grad.cpu() - bd.grad.float()).abs() / bd.grad.float().abs().clamp_min(1e-30)).max().item()
    assert rel_a < 5e-2 and rel_b < 5e-2


@pytest.mark.parametrize(
    "xr_type,xi_type", [("tensor", "scalar"), ("scalar", "tensor"),
                        ("tensor", "tensor"), ("tensor", "0dim"), ("0dim", "tensor")]
)
def test_npu_reciprocal_parts_forward(xr_type, xi_type):
    device = _maybe_npu_device()
    if device is None:
        pytest.skip("NPU is not available")

    torch.manual_seed(8)
    n = 100003
    xc = torch.randn(n)
    xr = xc if xr_type == "tensor" else (torch.randn(()) if xr_type == "0dim" else 0.5)
    xi = -xc if xi_type == "tensor" else (torch.randn(()) if xi_type == "0dim" else -0.05)
    xr_b = xc if xr_type == "tensor" else torch.full((n,), float(xr))
    xi_b = -xc if xi_type == "tensor" else torch.full((n,), float(xi))
    ref = 1 / torch.complex(xr_b, xi_b).to(torch.complex128)

    def to_npu(t):
        return t.to(device) if isinstance(t, torch.Tensor) else t

    out = complex_reciprocal(to_npu(xr), to_npu(xi))
    torch.npu.synchronize()
    assert out.dtype == torch.complex64
    assert _relerr(out, ref) < REL_TOL


def test_npu_reciprocal_parts_backward():
    device = _maybe_npu_device()
    if device is None:
        pytest.skip("NPU is not available")

    torch.manual_seed(9)
    n = 4096
    xc = torch.randn(n)
    u, v = torch.randn(n), torch.randn(n)

    # 参考梯度：float64 CPU autograd（含标量 xi 的梯度）
    xr64 = xc.double().requires_grad_(True)
    xi64 = torch.tensor(-0.05, dtype=torch.float64, requires_grad=True)
    ref = 1 / (xr64 + 1j * xi64)
    (ref.real * u.double() + ref.imag * v.double()).sum().backward()

    xr = xc.to(device).requires_grad_(True)
    xi = torch.tensor(-0.05, device=device, requires_grad=True)
    out = complex_reciprocal(xr, xi)
    g = torch.complex(u, v).to(device)
    (out * g.conj()).real.sum().backward()
    torch.npu.synchronize()
    rel_xr = ((xr.grad.cpu() - xr64.grad.float()).abs() / xr64.grad.float().abs().clamp_min(1e-30)).max().item()
    assert rel_xr < 5e-2
    assert xi.grad.numel() == 1
    assert abs(xi.grad.item() - xi64.grad.item()) / abs(xi64.grad.item()) < 5e-3


def test_npu_int64_offset_path(monkeypatch):
    """强制走 int64 偏移分支（默认只在 numel >= 2^30 时启用）。"""
    device = _maybe_npu_device()
    if device is None:
        pytest.skip("NPU is not available")

    monkeypatch.setattr(recip_impl, "_INT64_THRESHOLD", 16)
    torch.manual_seed(10)
    n = 100003
    zc = (torch.randn(n) + 1j * torch.randn(n)).to(torch.complex64)
    ref = 1 / zc.to(torch.complex128)
    out = complex_reciprocal(_to_npu(zc, device))
    torch.npu.synchronize()
    assert _relerr(out, ref) < REL_TOL

    xc = torch.randn(n)
    ref2 = 1 / (xc.to(torch.complex128) - 0.05j)
    out2 = complex_reciprocal(xc.to(device), -0.05)
    torch.npu.synchronize()
    assert _relerr(out2, ref2) < REL_TOL

    z = _to_npu(zc, device).requires_grad_(True)
    g = torch.complex(torch.randn(n), torch.randn(n)).to(device)
    gz = torch.autograd.grad(complex_reciprocal(z), z, grad_outputs=g)[0]
    torch.npu.synchronize()
    z_ref = zc.clone().requires_grad_(True)
    g_ref = torch.autograd.grad(1 / z_ref, z_ref, grad_outputs=torch.complex(
        g.cpu().real, g.cpu().imag))[0]
    assert _relerr(gz, g_ref) < REL_TOL


def test_npu_fused_tensor_scalar_expression():
    """端到端：实部张量 + 常数虚部的复数倒数 1 / (a - i*c)。"""
    device = _maybe_npu_device()
    if device is None:
        pytest.skip("NPU is not available")

    torch.manual_seed(11)
    a = torch.rand(PROD_SHAPE) * 4.0 - 0.25
    c = 1e-2
    a_dev = a.to(device)
    out = complex_reciprocal(a_dev, -c)
    torch.npu.synchronize()

    ref = 1 / (a.to(torch.complex128) - 1j * c)
    assert _relerr(out, ref) < REL_TOL
    # 与先构造 complex64 的原始 torch 写法逐元素接近（同为 float32，允许 ulp 级差异）
    baseline = 1 / (a_dev - 1j * c)
    assert _relerr(out, _to_cpu(baseline)) < 1e-6


def test_npu_grad_requires_grad_flow():
    """requires_grad=False 的输入不应触发 backward kernel 也能正常前向。"""
    device = _maybe_npu_device()
    if device is None:
        pytest.skip("NPU is not available")

    torch.manual_seed(12)
    z = _to_npu((torch.randn(64) + 1j * torch.randn(64)).to(torch.complex64), device)
    out = complex_reciprocal(z)
    assert not out.requires_grad

    z2 = z.clone().requires_grad_(True)
    out2 = complex_reciprocal(z2)
    assert out2.requires_grad
