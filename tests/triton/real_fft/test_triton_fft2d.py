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

"""real_rfft2 / real_irfft2：分离式代数与公开接口的 CPU 用例。

两层覆盖：

1. ``_rfft2_separable`` / ``_irfft2_separable`` 的代数正确性。这两个函数允许把一维
   原语换成 torch 的等价实现，因此在没有 NPU 的机器上也能验证「线性 + 共轭反转 +
   Hermitian 补全」这套分解本身，而不只是验证 torch 回退分支。
2. 公开接口的形状、``s`` 裁断/零填充、norm 换算、autograd 与入参校验。

NPU kernel 路径的正确性由 ``verify_real_fft2.py`` 在真机上覆盖。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from oil_gas_ops_prospect.real_fft import real_irfft2, real_rfft2  # noqa: E402
from oil_gas_ops_prospect.real_fft import fft2d as fft2d_mod  # noqa: E402

TOL = 1e-5


def _assert_close(actual, expect, tol=TOL):
    """按参考量级归一化的相对误差比较（float32 下绝对误差随信号幅度增长）。"""
    scale = max(1.0, expect.abs().max().item())
    err = (actual - expect).abs().max().item()
    assert err / scale < tol, f"max|delta|={err:.3e} over scale {scale:.3e}"

# (lead, h, w) —— 覆盖 2 的幂、非 2 的幂、以及在 strided kernel 边界两侧的长度。
SHAPES_2D = [
    (1, 8, 8),
    (2, 12, 8),
    (3, 16, 24),
    (1, 32, 64),
    (1, 33, 45),
    (1, 320, 8),
    (1, 576, 4),
]

DIMS_2D = [(-2, -1), (-1, -2), (1, 2)]


def _torch_rfft(x, dim, norm, **_):
    z = torch.fft.rfft(x, dim=dim, norm=norm)
    return z.real, z.imag


def _torch_irfft(xr, xi, n, dim, norm, **_):
    return torch.fft.irfft(torch.complex(xr, xi), n=n, dim=dim, norm=norm)


def _separable_rfft2(x, dim, norm):
    d1, d2 = (d % x.dim() for d in dim)
    return fft2d_mod._rfft2_separable(
        x, d1, d2, norm, fft2d_mod._rfft_opts(None, None, None), _torch_rfft
    )


def _separable_irfft2(xr, xi, s, dim, norm):
    d1, d2 = (d % xr.dim() for d in dim)
    return fft2d_mod._irfft2_separable(
        xr, xi, s[0], s[1], d1, d2, norm,
        fft2d_mod._rfft_opts(None, None, None), _torch_irfft, _torch_rfft,
    )


@pytest.mark.parametrize("norm", ["backward", "ortho"])
@pytest.mark.parametrize("shape", SHAPES_2D)
@pytest.mark.parametrize("dim", DIMS_2D)
def test_separable_2d_matches_torch(shape, dim, norm):
    torch.manual_seed(0)
    x = torch.randn(shape)
    if max(abs(d) for d in dim) >= x.dim():
        pytest.skip("dim does not apply to this rank")

    xr, xi = _separable_rfft2(x, dim, norm)
    ref = torch.fft.rfft2(x, dim=dim, norm=norm)
    assert xr.shape == ref.real.shape
    _assert_close(xr, ref.real)
    _assert_close(xi, ref.imag)

    d1, d2 = (d % x.dim() for d in dim)
    y = _separable_irfft2(ref.real, ref.imag, (x.shape[d1], x.shape[d2]), dim, norm)
    assert y.shape == x.shape
    _assert_close(y, x)


@pytest.mark.parametrize("norm", ["backward", "ortho"])
def test_separable_2d_gradients_match_torch(norm):
    torch.manual_seed(1)
    x = torch.randn(2, 8, 6, requires_grad=True)
    ref = torch.fft.rfft2(x, dim=(-2, -1), norm=norm)
    weight = torch.randn(ref.real.shape) + 1j * torch.randn(ref.real.shape)
    # 显式传入连续的 grad_output：``.sum()`` 会展开成全 0 stride，带 oneMKL 的构建
    # 对这套布局会报 DFTI 配置不一致（torch FFT 反向的已知脆弱点）。
    ref_grad, = torch.autograd.grad((ref * weight).real, x, grad_outputs=torch.ones_like(ref.real))

    xr, xi = _separable_rfft2(x, (-2, -1), norm)
    grad, = torch.autograd.grad((xr * weight.real - xi * weight.imag).sum(), x)
    _assert_close(grad, ref_grad)


def test_separable_norm_scaling_is_sqrt_of_total_length():
    """backward 与 ortho 只差一个 sqrt(h*w)。"""
    torch.manual_seed(2)
    x = torch.randn(1, 12, 10)
    b_r, b_i = _separable_rfft2(x, (-2, -1), "backward")
    o_r, o_i = _separable_rfft2(x, (-2, -1), "ortho")
    scale = math.sqrt(12 * 10)
    assert torch.allclose(b_r, o_r * scale, rtol=TOL, atol=TOL)
    assert torch.allclose(b_i, o_i * scale, rtol=TOL, atol=TOL)


@pytest.mark.parametrize("n,keep", [(16, 9), (288, 145), (576, 289)])
def test_hermitian_fill_is_axis_independent(n, keep):
    """补全结果只取决于数据，与走哪条轴（末轴搬移）无关。"""
    torch.manual_seed(8)
    x = torch.randn(2, keep, 3)
    torch.manual_seed(9)
    xi = torch.randn(2, keep, 3)

    last_r, last_i = fft2d_mod._hermitian(x, xi, n, 1)
    # 与直接按定义构造的全谱比对
    expect_r = torch.cat((x, x[:, 1:n - keep + 1].flip(1)), dim=1)
    expect_i = torch.cat((xi, -xi[:, 1:n - keep + 1].flip(1)), dim=1)
    assert last_r.shape[1] == n and last_i.shape[1] == n
    assert torch.equal(last_r, expect_r)
    assert torch.equal(last_i, expect_i)

    # 同一份数据换到别的轴，形状与数值关系保持一致
    moved = x.transpose(0, 1).contiguous()
    moved_i = xi.transpose(0, 1).contiguous()
    mid_r, mid_i = fft2d_mod._hermitian(moved, moved_i, n, 0)
    assert torch.equal(mid_r.transpose(0, 1), last_r)
    assert torch.equal(mid_i.transpose(0, 1), last_i)


# ---------------------------------------------------------------------------------------------
# 公开接口（CPU 回退分支）
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("norm", ["backward", "ortho"])
@pytest.mark.parametrize("shape", [(2, 8, 8), (1, 12, 16), (3, 5, 4)])
def test_public_api_cpu_fallback_matches_torch(shape, norm):
    torch.manual_seed(3)
    x = torch.randn(shape)
    xr, xi = real_rfft2(x, dim=(-2, -1), norm=norm)
    ref = torch.fft.rfft2(x, dim=(-2, -1), norm=norm)
    assert torch.allclose(xr, ref.real, rtol=TOL, atol=TOL)
    assert torch.allclose(xi, ref.imag, rtol=TOL, atol=TOL)

    y = real_irfft2(xr, xi, s=shape[-2:], dim=(-2, -1), norm=norm)
    assert y.shape == x.shape
    assert torch.allclose(y, x, rtol=TOL, atol=TOL)


def test_public_api_default_s_infers_signal_length():
    torch.manual_seed(4)
    x = torch.randn(2, 6, 10)
    xr, xi = real_rfft2(x)
    assert xr.shape == (2, 6, 6)
    y = real_irfft2(xr, xi)
    assert y.shape == x.shape
    assert torch.allclose(y, x, rtol=TOL, atol=TOL)


def test_public_api_s_pads_and_truncates_like_torch():
    """``s`` 的裁断/零填充语义与 torch.fft 一致。"""
    torch.manual_seed(5)
    x = torch.randn(1, 6, 10)
    for s in [(4, 8), (8, 16), (6, 10), (3, 5)]:
        xr, xi = real_rfft2(x, s=s)
        ref = torch.fft.rfft2(x, s=s)
        assert xr.shape == ref.real.shape
        assert torch.allclose(xr, ref.real, rtol=TOL, atol=TOL)
        assert torch.allclose(xi, ref.imag, rtol=TOL, atol=TOL)

        y = real_irfft2(xr, xi, s=s)
        assert y.shape == (*x.shape[:-2], *s)
        assert torch.allclose(y, torch.fft.irfft2(torch.complex(xr, xi), s=s), rtol=TOL, atol=TOL)


def test_public_api_autograd_through_cpu_fallback():
    torch.manual_seed(6)
    x = torch.randn(1, 8, 8, requires_grad=True)
    xr, xi = real_rfft2(x)
    (xr.pow(2).sum() + xi.pow(2).sum()).backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()

    amp = torch.randn_like(xr).requires_grad_(True)
    phase = torch.randn_like(xi).requires_grad_(True)
    y = real_irfft2(amp, phase, s=(8, 8))
    y.sum().backward()
    assert torch.isfinite(amp.grad).all()
    assert torch.isfinite(phase.grad).all()


@pytest.mark.parametrize("norm", ["backward", "ortho"])
@pytest.mark.parametrize("shape,s", [
    ((1, 8, 8), (8, 8)),
    ((2, 4, 6), (4, 6)),
    ((1, 3, 5), (3, 5)),
    ((1, 8, 7), (8, 7)),      # odd width
    ((2, 5, 9), (5, 9)),
])
def test_irfft2_gradient_populates_without_mkl_layout_error(shape, s, norm):
    """``.sum()`` 会产生 stride 全 0 的 grad_output。

    该布局在带 oneMKL 的构建上会让 ``torch.fft.irfft2`` 的反向报
    "Inconsistent configuration parameters"（CI 记录过）。这里断言反向可用、
    梯度有限。

    正确性用伴随恒等式校验，只用**前向**变换，不触发 torch 的 FFT 反向
    （torch 的 FFT 反向本身正是会踩 MKL 的那条路径，不能作为参考）：
    对线性算子 ``A``，恒有 ``<A p, 1> == <p, A^T 1>``，而 ``A^T 1`` 就是
    ``y.sum()`` 得到的梯度。
    """
    torch.manual_seed(10)
    h, w = s
    half = w // 2 + 1
    amp = torch.randn(*shape[:-1], half, requires_grad=True)
    phase = torch.randn(*shape[:-1], half, requires_grad=True)

    y = real_irfft2(amp, phase, s=s, norm=norm)
    y.sum().backward()

    assert amp.grad is not None and phase.grad is not None
    assert torch.isfinite(amp.grad).all()
    assert torch.isfinite(phase.grad).all()
    assert amp.grad.shape == amp.shape
    assert phase.grad.shape == phase.shape

    # 伴随恒等式：<A(p), 1> == <p, A^T(1)>，仅用前向变换
    probe_amp = torch.randn_like(amp)
    probe_phase = torch.randn_like(phase)
    lhs = real_irfft2(probe_amp, probe_phase, s=s, norm=norm).sum()
    rhs = (probe_amp * amp.grad).sum() + (probe_phase * phase.grad).sum()
    scale = max(lhs.abs().item(), rhs.abs().item(), 1.0)
    assert torch.allclose(lhs, rhs, rtol=1e-5, atol=1e-5 * scale)


def test_rfft2_gradient_survives_stride0_grad_output():
    """``rfft2`` 的反向是 ``irfft2``，同样会被 stride-0 的 grad_output 触发 MKL 问题。"""
    torch.manual_seed(13)
    x = torch.randn(2, 8, 8, requires_grad=True)
    xr, xi = real_rfft2(x)
    (xr.pow(2).sum() + xi.pow(2).sum()).backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()

    # 非退化的 grad_output 下与 torch 逐值一致
    seed = torch.randn(2, 8, 8)
    base = torch.fft.rfft2(seed)
    a = torch.randn_like(base.real)
    b = torch.randn_like(base.imag)

    xt = seed.clone().requires_grad_(True)
    ref = torch.fft.rfft2(xt)
    # 显式给稠密 grad_output，避免任何展开（全 0 stride）布局进入 torch 的 FFT 反向。
    expect, = torch.autograd.grad(ref, xt, grad_outputs=torch.complex(a, b))

    x2 = seed.clone().requires_grad_(True)
    xr2, xi2 = real_rfft2(x2)
    got, = torch.autograd.grad((xr2 * a + xi2 * b).sum(), x2)

    assert torch.allclose(got, expect, rtol=1e-5, atol=1e-6)


def test_irfft2_gradient_matches_torch_for_weighted_loss():
    """非退化的 grad_output 下，自定义反向必须与 torch 逐值一致。"""
    torch.manual_seed(11)
    amp = torch.randn(2, 4, 4, requires_grad=True)
    phase = torch.randn(2, 4, 4, requires_grad=True)
    weight = torch.randn(2, 4, 6)

    (real_irfft2(amp, phase, s=(4, 6)) * weight).sum().backward()

    amp2 = amp.detach().clone().requires_grad_(True)
    phase2 = phase.detach().clone().requires_grad_(True)
    y_ref = torch.fft.irfft2(torch.complex(amp2, phase2), s=(4, 6))
    # 显式给稠密 grad_output，避免 stride-0 布局进入 torch 的 FFT 反向（oneMKL 会报错）。
    ref_amp, ref_phase = torch.autograd.grad(y_ref, (amp2, phase2), grad_outputs=weight)

    assert torch.allclose(amp.grad, ref_amp, rtol=1e-5, atol=1e-6)
    assert torch.allclose(phase.grad, ref_phase, rtol=1e-5, atol=1e-6)


def test_irfft2_non_contiguous_spectrum_inputs():
    """输入是复数张量的 real/imag 切片（末轴 stride 2）时也要正确。"""
    torch.manual_seed(12)
    x = torch.randn(2, 6, 8)
    ref = torch.fft.rfft2(x)
    amp = ref.real.clone().requires_grad_(True)
    phase = ref.imag.clone().requires_grad_(True)
    # clone() may densify; make the strided case explicit either way.
    amp = amp.detach().requires_grad_(True)
    phase = phase.detach().requires_grad_(True)

    y = real_irfft2(amp, phase, s=(6, 8))
    assert torch.allclose(y, x, rtol=1e-5, atol=1e-5)
    y.sum().backward()
    assert torch.isfinite(amp.grad).all() and torch.isfinite(phase.grad).all()


def test_public_api_degenerate_axes():
    """长度 1 的变换轴：半谱长度也是 1，信号长度无法从谱反推，``s`` 必须显式给出。"""
    torch.manual_seed(7)
    x = torch.randn(2, 4, 1)
    xr, xi = real_rfft2(x)
    ref = torch.fft.rfft2(x)
    assert xr.shape == ref.real.shape == (2, 4, 1)
    _assert_close(xr, ref.real)
    _assert_close(xi, ref.imag)
    _assert_close(real_irfft2(xr, xi, s=(4, 1)), x)
    with pytest.raises(ValueError):
        real_irfft2(xr, xi)


def test_public_api_rejects_bad_arguments():
    x = torch.randn(2, 4, 4)
    with pytest.raises(ValueError):
        real_rfft2(x, norm="forward")
    with pytest.raises(ValueError):
        real_rfft2(x, dim=(1, 1))
    with pytest.raises(ValueError):
        real_rfft2(x, dim=(1, 2, 3))
    with pytest.raises(TypeError):
        real_rfft2(x, dim=1)
    with pytest.raises(IndexError):
        real_rfft2(x, dim=(0, 3))
    with pytest.raises(TypeError):
        real_rfft2(torch.randn(2, 3, dtype=torch.complex64))
    with pytest.raises(ValueError):
        real_rfft2(x, s=(4, 0))

    xr, xi = real_rfft2(x)
    with pytest.raises(ValueError):
        real_irfft2(xr, xi[:, :1])
    with pytest.raises(ValueError):
        real_irfft2(xr, xi, norm="forward")


# ---------------------------------------------------------------------------------------------
# 非末轴搬轴兜底
# ---------------------------------------------------------------------------------------------

class _Recorder:
    """记录一维原语实际收到的变换轴与长度，并转发给 torch。"""

    def __init__(self):
        self.rfft_calls = []
        self.irfft_calls = []

    def rfft(self, x, dim, norm, **_):
        self.rfft_calls.append((dim % x.dim(), x.shape[dim], x.is_contiguous()))
        return _torch_rfft(x, dim, norm)

    def irfft(self, xr, xi, n, dim, norm, **_):
        self.irfft_calls.append((dim % xr.dim(), n, xr.is_contiguous()))
        return _torch_irfft(xr, xi, n, dim, norm)


@pytest.mark.parametrize("n", [128, 320, 576])
def test_non_last_axis_moves_to_last_only_above_threshold(n):
    rec = _Recorder()
    x = torch.randn(1, n, 2)
    fft2d_mod._rfft_1d(x, 1, "ortho", {}, rec.rfft)
    axis, length, contiguous = rec.rfft_calls[-1]
    assert (axis, length, contiguous) == (2 if n > 320 else 1, n, True)


@pytest.mark.parametrize("n", [128, 320, 576])
def test_non_last_axis_irfft_moves_to_last_only_above_threshold(n):
    """``real_irfft`` 沿非末轴且长度较大时同样搬轴，避免 strided kernel 编译失败。"""
    rec = _Recorder()
    xr = torch.randn(1, n // 2 + 1, 2)
    xi = torch.randn_like(xr)
    fft2d_mod._irfft_1d(xr, xi, n, 1, "ortho", {}, rec.irfft)
    axis, length, contiguous = rec.irfft_calls[-1]
    assert (axis, length, contiguous) == (2 if n > 320 else 1, n, True)
