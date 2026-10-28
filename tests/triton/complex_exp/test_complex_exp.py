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

"""complex_exp 测试：CPU 回退逐位一致、NPU 前/反向精度、int64 分支、极值。

NPU 用例在无 NPU 环境自动 skip；CPU 用例在任何环境都跑。
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
import torch

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import _bootstrap  # noqa: F401, E402
from oil_gas_ops_prospect.complex_exp import (  # noqa: E402
    complex_exp,
    triton_complex_exp as exp_impl,
)

# torch_npu 的 complex 张量 .to(torch.complex128) 会静默丢虚部（见
# complex_reciprocal 测试），比较时统一在 CPU 侧转换。
REL_TOL = 1e-5

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


def _relerr(got: torch.Tensor, ref: torch.Tensor) -> float:
    got = _to_cpu(got)
    ref = ref.to(torch.complex128)
    diff = (got - ref).abs()
    return (diff / ref.abs().clamp_min(1e-30)).max().item()


# ------------------------------------------------------------------ #
# CPU 回退
# ------------------------------------------------------------------ #

@pytest.mark.parametrize(
    "shape", [(), (1,), (13,), (2, 3, 5), (1000001,), (8, 16, 32, 7)]
)
def test_cpu_fallback_matches_torch_exp(shape):
    torch.manual_seed(0)
    z = (torch.randn(shape) + 1j * torch.randn(shape)).to(torch.complex64)
    assert torch.equal(complex_exp(z), torch.exp(z))


def test_cpu_fallback_autograd():
    torch.manual_seed(1)
    z = (torch.randn(4, 7) + 1j * torch.randn(4, 7)).to(torch.complex64)
    z_g = z.clone().requires_grad_(True)
    g = torch.randn_like(z_g.real) + 1j * torch.randn_like(z_g.real)

    out = complex_exp(z_g)
    (out.real * g.real + out.imag * g.imag).sum().backward()

    z_r = z.clone().requires_grad_(True)
    ref = torch.exp(z_r)
    (ref.real * g.real + ref.imag * g.imag).sum().backward()

    assert _relerr(out, ref.to(torch.complex64)) < 1e-7
    assert _relerr(z_g.grad, z_r.grad) < 1e-7


def test_resolve_block_clamps_to_power_of_two(monkeypatch):
    monkeypatch.setenv("OIL_GAS_OPS_EXP_BLOCK", "256")
    assert exp_impl._resolve_block(None) == 256
    monkeypatch.setenv("OIL_GAS_OPS_EXP_BLOCK", "4096")
    assert exp_impl._resolve_block(None) == exp_impl._MAX_BLOCK
    monkeypatch.setenv("OIL_GAS_OPS_EXP_BLOCK", "3")
    assert exp_impl._resolve_block(None) == 16  # 下限钳制
    monkeypatch.setenv("OIL_GAS_OPS_EXP_BLOCK", "0")
    assert exp_impl._resolve_block(None) == exp_impl._DEFAULT_BLOCK


# ------------------------------------------------------------------ #
# NPU 精度
# ------------------------------------------------------------------ #

@pytest.mark.parametrize(
    "shape",
    [(), (1,), (13,), (1000001,), (64, 64, 64), PROD_SHAPE, (4096, 4096)],
)
def test_npu_forward_matches_float64_reference(shape):
    device = _maybe_npu_device()
    if device is None:
        pytest.skip("NPU is not available")

    torch.manual_seed(2)
    z = (torch.randn(shape) + 1j * torch.randn(shape)).to(torch.complex64)
    out = complex_exp(z.to(device))
    ref = torch.exp(z.to(torch.complex128))
    assert _relerr(out, ref) < REL_TOL


def test_npu_forward_noncontiguous_and_empty():
    device = _maybe_npu_device()
    if device is None:
        pytest.skip("NPU is not available")

    torch.manual_seed(3)
    base = (torch.randn(4, 8) + 1j * torch.randn(4, 8)).to(torch.complex64)
    # 在设备上取转置视图，构造非连续输入（避免把非连续张量 .to(device) 触发
    # torch_npu 的 internal format 告警）。
    z_nc = base.to(device).t()
    out = complex_exp(z_nc)
    ref = torch.exp(base.t().to(torch.complex128))
    assert _relerr(out, ref) < REL_TOL

    z_empty = torch.empty(0, 3, dtype=torch.complex64, device=device)
    assert complex_exp(z_empty).shape == (0, 3)

    # 空张量也要保持 autograd 连接：反向得到同形状空梯度而非断链。
    z_empty_g = torch.empty(0, 3, dtype=torch.complex64, device=device, requires_grad=True)
    out_empty_g = complex_exp(z_empty_g)
    out_empty_g.real.sum().backward()
    assert z_empty_g.grad is not None and z_empty_g.grad.shape == (0, 3)


def test_npu_zero_dim_forward_backward():
    """0-dim（标量）复数输入：view(dtype) 需先 reshape，前向与反向都要走通。"""
    device = _maybe_npu_device()
    if device is None:
        pytest.skip("NPU is not available")

    z = torch.tensor(1.0 + 2.0j, dtype=torch.complex64, device=device, requires_grad=True)
    out = complex_exp(z)
    assert out.shape == ()
    assert out.dtype == torch.complex64
    ref = torch.exp(torch.tensor(1.0 + 2.0j, dtype=torch.complex128))
    assert _relerr(out, ref) < REL_TOL

    (out.real + 0.5 * out.imag).sum().backward()
    z_ref = torch.tensor(1.0 + 2.0j, dtype=torch.complex64, requires_grad=True)
    ref_out = torch.exp(z_ref)
    (ref_out.real + 0.5 * ref_out.imag).sum().backward()
    assert z.grad is not None and z.grad.shape == ()
    assert _relerr(z.grad, z_ref.grad) < REL_TOL


@pytest.mark.parametrize("re", [0.0, -1.0, 1.0, 80.0, -80.0])
@pytest.mark.parametrize("im", [0.0, 1.0, -1.0, 10.0, 100.0])
def test_npu_forward_value_domain(re, im):
    """特定取值域（含纯虚指数 e^{iθ} 与大实部）与 torch 逐元素一致。"""
    device = _maybe_npu_device()
    if device is None:
        pytest.skip("NPU is not available")

    z = torch.full((17,), complex(re, im), dtype=torch.complex64)
    out = complex_exp(z.to(device))
    ref = torch.exp(z.to(torch.complex128))
    assert _relerr(out, ref) < REL_TOL


def test_npu_forward_extreme_overflow_semantics():
    """实部超出 float32 exp 值域时与 torch complex64 同步上溢/下溢。"""
    device = _maybe_npu_device()
    if device is None:
        pytest.skip("NPU is not available")

    z = torch.tensor(
        [complex(100.0, 0.5), complex(-100.0, 0.5)], dtype=torch.complex64
    ).to(device)
    out = complex_exp(z)
    ref = torch.exp(z.cpu())
    assert torch.equal(out.cpu().real.isinf(), ref.real.isinf())
    assert torch.equal(out.cpu().abs() == 0, ref.abs() == 0)


def _assert_parts_equal(got: torch.Tensor, ref: torch.Tensor, atol: float = 0.0) -> None:
    """逐元素比较实部/虚部：NaN 视为相等，其余要求逐位相等（atol>0 时放宽）。"""
    got = got.detach().cpu()
    ref = ref.detach().cpu()
    for g, r in ((got.real, ref.real), (got.imag, ref.imag)):
        g64, r64 = g.to(torch.float64), r.to(torch.float64)
        same_nan = torch.isnan(g64) & torch.isnan(r64)
        ok = same_nan | (g64 == r64)
        if atol > 0:
            finite = torch.isfinite(g64) & torch.isfinite(r64)
            ok = ok | (finite & ((g64 - r64).abs() <= atol * r64.abs().clamp_min(1e-30)))
        assert torch.equal(ok, torch.ones_like(ok)), (
            f"mismatch at {torch.nonzero(~ok).flatten()[:5].tolist()}: "
            f"got={g[~ok][:5].tolist()} ref={r[~ok][:5].tolist()}"
        )


def test_npu_forward_pure_real_overflow_keeps_imag_zero():
    """纯实数上溢时虚部必须是 +0，不能是 inf*0 得到的 NaN。"""
    device = _maybe_npu_device()
    if device is None:
        pytest.skip("NPU is not available")

    im = torch.tensor([0.0, -0.0, 0.0, -0.0, 0.0], dtype=torch.float32)
    z = torch.complex(torch.tensor([89.0, 100.0, 152.0, 300.0, float("inf")]), im)
    out = complex_exp(z.to(torch.complex64).to(device))
    ref = torch.exp(z.to(torch.complex64))
    _assert_parts_equal(out, ref)
    assert torch.equal(out.cpu().imag, im), "虚部的符号位也要与 torch 一致"


def test_npu_forward_overflow_band_matches_torch():
    """88.7 < a < 192 的缩放带：实部可能有限，不能先溢出成 inf。"""
    device = _maybe_npu_device()
    if device is None:
        pytest.skip("NPU is not available")

    a = torch.tensor([89.0, 100.0, 150.0, 191.0], dtype=torch.float32)
    b = torch.tensor([1.0, 1e-30, 1.5707964, 3.1415927], dtype=torch.float32)
    z = torch.complex(a, b)
    out = complex_exp(z.to(device))
    ref = torch.exp(z)
    _assert_parts_equal(out, ref, atol=1e-6)


def test_npu_forward_special_values_match_torch():
    """±inf / NaN 组合按 cexpf 的语义特判，与 torch 逐位一致。"""
    device = _maybe_npu_device()
    if device is None:
        pytest.skip("NPU is not available")

    inf, nan = float("inf"), float("nan")
    avals = [inf, -inf, nan, 0.0, -0.0, 1.0, -104.0]
    bvals = [inf, -inf, nan, 0.0, -0.0, 1.0]
    grid = torch.tensor([[complex(a, b) for b in bvals] for a in avals], dtype=torch.complex64)
    flat = grid.reshape(-1)
    out = complex_exp(flat.to(device))
    # 有限值上 triton 的 cos/sin 与 torch 有 1 ulp 差异，故给 atol；
    # NaN / ±inf 的判定仍是逐位严格（atol 分支只覆盖双有限的情况）。
    _assert_parts_equal(out, torch.exp(flat), atol=1e-6)


def test_npu_backward_accepts_conj_view_grad():
    """二阶导链路上 grad_out 可能是 conjugate view，kernel 入口要能处理。"""
    device = _maybe_npu_device()
    if device is None:
        pytest.skip("NPU is not available")

    torch.manual_seed(8)
    z = (torch.randn(9, 7) + 1j * torch.randn(9, 7)).to(torch.complex64)
    z_g = z.clone().to(device).requires_grad_(True)
    w = complex_exp(z_g)
    gz, = torch.autograd.grad(w, z_g, grad_outputs=torch.ones_like(w), create_graph=True)
    # 生成一个 conjugate view 作为上游梯度
    assert gz.is_conj() or gz.conj().is_conj()
    gz2, = torch.autograd.grad((gz * gz.conj()).real.sum(), z_g)
    assert gz2.shape == z.shape


def test_npu_double_backward_matches_float64():
    """二阶导（grad of grad）与 CPU float64 参考一致。"""
    device = _maybe_npu_device()
    if device is None:
        pytest.skip("NPU is not available")

    torch.manual_seed(9)
    z0 = (torch.randn(6, 5) + 1j * torch.randn(6, 5)).to(torch.complex64)
    g0 = (torch.randn(6, 5) + 1j * torch.randn(6, 5)).to(torch.complex64)
    h0 = (torch.randn(6, 5) + 1j * torch.randn(6, 5)).to(torch.complex64)

    def second_order(fn, device, dtype):
        z = z0.to(device).to(dtype).requires_grad_(True)
        g = g0.to(device).to(dtype)
        h = h0.to(device).to(dtype)
        gz, = torch.autograd.grad(fn(z), z, grad_outputs=g, create_graph=True)
        gg, = torch.autograd.grad((gz * h.conj()).real.sum(), z)
        return gz.detach().cpu(), gg.detach().cpu()

    gz_npu, gg_npu = second_order(complex_exp, device, torch.complex64)
    gz_ref, gg_ref = second_order(torch.exp, "cpu", torch.complex128)
    assert _relerr(gz_npu, gz_ref) < REL_TOL
    assert _relerr(gg_npu, gg_ref) < REL_TOL



def test_npu_backward_matches_torch_grad_wrt_z():
    device = _maybe_npu_device()
    if device is None:
        pytest.skip("NPU is not available")

    torch.manual_seed(4)
    z = (torch.randn(64, 31) + 1j * torch.randn(64, 31)).to(torch.complex64)
    z_g = z.clone().to(device).requires_grad_(True)
    g = (torch.randn_like(z.real) + 1j * torch.randn_like(z.real)).to(torch.complex64).to(device)

    out = complex_exp(z_g)
    (out.real * g.real + out.imag * g.imag).sum().backward()

    z_r = z.clone().requires_grad_(True)
    ref = torch.exp(z_r)
    g_r = g.cpu()
    (ref.real * g_r.real + ref.imag * g_r.imag).sum().backward()
    assert _relerr(z_g.grad, z_r.grad) < 1e-5


def test_npu_backward_end_to_end_matches_float64():
    """端到端（exp → abs² → sum）链路反向 vs float64 参考。"""
    device = _maybe_npu_device()
    if device is None:
        pytest.skip("NPU is not available")

    torch.manual_seed(5)
    z = (torch.randn(33, 17) + 1j * torch.randn(33, 17)).to(torch.complex64)
    z_g = z.clone().to(device).requires_grad_(True)
    w = complex_exp(z_g)
    loss = (w.real ** 2 + w.imag ** 2).sum()
    loss.backward()

    z64 = z.to(torch.complex128).requires_grad_(True)
    w64 = torch.exp(z64)
    loss64 = (w64.real ** 2 + w64.imag ** 2).sum()
    loss64.backward()
    assert _relerr(z_g.grad, z64.grad) < 1e-4
    assert abs(loss.item() - loss64.item()) / loss64.item() < 1e-5


def test_npu_int64_offset_path(monkeypatch):
    """把 int64 阈值调小，强制走 int64 偏移分支验证与 int32 分支一致。"""
    device = _maybe_npu_device()
    if device is None:
        pytest.skip("NPU is not available")

    monkeypatch.setattr(exp_impl, "_INT64_THRESHOLD", 16)
    torch.manual_seed(6)
    z = (torch.randn(128) + 1j * torch.randn(128)).to(torch.complex64)
    out64 = complex_exp(z.to(device))

    monkeypatch.setattr(exp_impl, "_INT64_THRESHOLD", 1 << 30)
    out32 = complex_exp(z.to(device))
    assert torch.equal(out64.cpu(), out32.cpu())


def test_npu_grad_requires_grad_flow():
    """requires_grad=False 的输入不触发 backward kernel 也能正常前向。"""
    device = _maybe_npu_device()
    if device is None:
        pytest.skip("NPU is not available")

    torch.manual_seed(7)
    z = (torch.randn(64) + 1j * torch.randn(64)).to(torch.complex64)
    out = complex_exp(z.to(device))
    assert _relerr(out, torch.exp(z.to(torch.complex128))) < REL_TOL
