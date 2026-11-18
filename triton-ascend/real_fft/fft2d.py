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

"""二维实数 FFT：``real_rfft2`` / ``real_irfft2``。

本模块不引入新的 Triton kernel，而是把已有的 1-D ``real_rfft`` / ``real_irfft``
组合成二维变换，语义对齐 ``torch.fft.rfft2`` / ``torch.fft.irfft2``。返回值与
一维入口一致：实部、虚部两个实数张量，不构造 complex dtype（NPU 上复数 kernel
支持有限）。

组合依赖两个恒等式，二者都只要求「实序列的 FFT」：

    FFT(a + i b) = FFT(a) + i FFT(b)                  线性
    IFFT_ortho(z) = conj( FFT_ortho( conj(z) ) )       共轭反转（ortho 归一化下不除 n）

``real_rfft`` 只返回半谱（长度 ``n//2+1``），所以进入恒等式之前先按
``X[n-k] = conj(X[k])`` 补成全谱，补全长度为 ``n - (n//2+1)``。

变换顺序与 ``rfft2`` 一致：正向先在 ``dim[1]``（半谱轴）做实数 FFT，再在
``dim[0]`` 做完整复数 FFT；反向逆序执行。内部一律按 ``norm="ortho"`` 计算
（这是 1-D 算子唯一支持的归一化），``norm="backward"`` 通过整体乘/除
``sqrt(h*w)`` 换算：

    rfft2  backward = ortho * sqrt(h*w)
    irfft2 backward = ortho / sqrt(h*w)

一维算子沿非末轴时走 strided kernel，变换长度较大时该 kernel 编译会失败
（实测边界在 336/352 之间）。因此本模块在长度超过 ``_AXIS_TRANSPOSE_MIN_N``
时先把轴搬到末轴（设备内 transpose）。这层兜底对正向、反向一致内置，调用方
不需要感知；``_AXIS_TRANSPOSE_MIN_N`` 是内部常量，不暴露环境变量。

非 NPU、Triton 不可用或输入不是 ``float32`` 时回退到 ``torch.fft``，便于在
CPU 上做精度自检。
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn.functional as F

from .triton_fft_real import _has_triton, _is_npu_tensor, real_irfft, real_rfft

__all__ = ["real_rfft2", "real_irfft2"]

# 非末轴 1-D 变换超过此长度时先把轴搬到末轴。strided（非末轴）kernel 在更大的
# 变换长度上会 cbuf overflow 编译失败，末轴 kernel 在同一长度上正确。
_AXIS_TRANSPOSE_MIN_N = 320


# ---------------------------------------------------------------------------------------------
# 参数校验与 shape 处理
# ---------------------------------------------------------------------------------------------

def _resolve_dims(dim, ndim: int) -> tuple[int, int]:
    """把 ``dim`` 归一化成两个非负、互不相同的轴下标。"""
    if isinstance(dim, int):
        raise TypeError(f"dim must be a 2-element sequence of ints, got {dim!r}")
    try:
        dims = tuple(dim)
    except TypeError:
        raise TypeError(f"dim must be a 2-element sequence of ints, got {dim!r}") from None
    if len(dims) != 2:
        raise ValueError(f"dim must have exactly 2 elements, got {dims!r}")
    out: list[int] = []
    for d in dims:
        if not isinstance(d, int):
            raise TypeError(f"dim elements must be int, got {d!r}")
        if not -ndim <= d < ndim:
            raise IndexError(f"dim {d} is out of range for a {ndim}-D input")
        out.append(d % ndim)
    if out[0] == out[1]:
        raise ValueError(f"transform dimensions must be distinct, got {dims!r}")
    return out[0], out[1]


def _resolve_sizes(s: Sequence[int] | None) -> tuple[int, int] | None:
    """校验 ``s``。``None`` 表示沿用输入尺寸。"""
    if s is None:
        return None
    sizes = tuple(int(v) for v in s)
    if len(sizes) != 2:
        raise ValueError(f"s must have exactly 2 elements, got {sizes!r}")
    if any(v <= 0 for v in sizes):
        raise ValueError(f"s must be strictly positive, got {sizes!r}")
    return sizes[0], sizes[1]


def _resize_dim(t: torch.Tensor, dim: int, size: int) -> torch.Tensor:
    """把 ``t`` 的 ``dim`` 轴裁到或零填充到 ``size``（与 ``torch.fft`` 的 ``s`` 语义一致）。"""
    cur = t.shape[dim]
    if cur == size:
        return t
    if cur > size:
        return t.narrow(dim, 0, size)
    pads = [0] * (2 * t.dim())
    pads[2 * (t.dim() - 1 - dim) + 1] = size - cur
    return F.pad(t, pads)


# ---------------------------------------------------------------------------------------------
# 1-D 原语：非末轴且长度较大时先搬轴
# ---------------------------------------------------------------------------------------------

def _rfft_1d(x: torch.Tensor, dim: int, norm: str, opts: dict, rfft=None):
    """``rfft`` 沿 ``dim``；非末轴且长度较大时先 ``movedim`` 到末轴再变换。"""
    rfft = rfft or real_rfft
    ax = dim % x.dim()
    n = x.shape[ax]
    if ax == x.dim() - 1 or n <= _AXIS_TRANSPOSE_MIN_N:
        return rfft(x, dim=ax, norm=norm, **opts)
    xt = x.movedim(ax, -1).contiguous()
    re, im = rfft(xt, dim=-1, norm=norm, **opts)
    return re.movedim(-1, ax), im.movedim(-1, ax)


def _irfft_1d(xr: torch.Tensor, xi: torch.Tensor, n: int, dim: int, norm: str, opts: dict,
              irfft=None):
    """``irfft`` 沿 ``dim``；与 :func:`_rfft_1d` 相同的搬轴兜底。"""
    irfft = irfft or real_irfft
    ax = dim % xr.dim()
    if ax == xr.dim() - 1 or n <= _AXIS_TRANSPOSE_MIN_N:
        return irfft(xr, xi, n=n, dim=ax, norm=norm, **opts)
    xrt = xr.movedim(ax, -1).contiguous()
    xit = xi.movedim(ax, -1).contiguous()
    out = irfft(xrt, xit, n=n, dim=-1, norm=norm, **opts)
    return out.movedim(-1, ax)


# ---------------------------------------------------------------------------------------------
# 全谱补全与复数 FFT（全部基于实序列的 1-D rFFT）
# ---------------------------------------------------------------------------------------------

def _fill_half_spectrum(xr: torch.Tensor, xi: torch.Tensor, n: int, dim: int):
    """在 ``dim`` 上把半谱补成全谱（不归一化、不改 dtype）。"""
    keep = xr.shape[dim]
    tail = n - keep
    if tail <= 0:
        return xr, xi
    return (
        torch.cat((xr, xr.narrow(dim, 1, tail).flip((dim,))), dim=dim),
        torch.cat((xi, (-xi.narrow(dim, 1, tail)).flip((dim,))), dim=dim),
    )


def _hermitian(xr: torch.Tensor, xi: torch.Tensor, n: int, dim: int):
    """半谱（长度 ``n//2+1``）补成全谱（长度 ``n``）：``X[n-k] = conj(X[k])``。

    沿末轴（stride 1 轴）``flip`` 在 910B 上会退化成一个很慢的 kernel（同形状下比
    沿非末轴慢两个数量级），而 ``movedim`` 本身很便宜。所以末轴的情形先把该轴搬到
    0 再做镜像，最后搬回。形状与数值完全一致。
    """
    if xr.dim() == 1 or dim != xr.dim() - 1:
        return _fill_half_spectrum(xr, xi, n, dim)
    r0, i0 = xr.movedim(dim, 0), xi.movedim(dim, 0)
    out_r, out_i = _fill_half_spectrum(r0, i0, n, 0)
    return out_r.movedim(0, dim), out_i.movedim(0, dim)


def _cfft(a: torch.Tensor, b: torch.Tensor, n: int, dim: int, opts: dict, rfft=None):
    """``a + i b``（两者均为实序列，长度 ``n``）的完整复数 FFT，``ortho`` 归一化。"""
    ar, ai = _hermitian(*_rfft_1d(a, dim, "ortho", opts, rfft), n, dim)
    br, bi = _hermitian(*_rfft_1d(b, dim, "ortho", opts, rfft), n, dim)
    return ar - bi, ai + br


def _cifft(zr: torch.Tensor, zi: torch.Tensor, n: int, dim: int, opts: dict, rfft=None):
    """``zr + i zi``（长度 ``n`` 的完整谱）的复数逆，``ortho`` 归一化。"""
    # IFFT_ortho(z) = conj(FFT_ortho(conj(z)))，ortho 下不需要除以 n。
    fr, fi = _cfft(zr, -zi, n, dim, opts, rfft)
    return fr, -fi


# ---------------------------------------------------------------------------------------------
# 分离式二维核心（rfft / irfft 可替换，便于在 CPU 上验证同一套代数）
# ---------------------------------------------------------------------------------------------

def _rfft2_separable(x: torch.Tensor, dim1: int, dim2: int, norm: str, opts: dict, rfft=None):
    """``torch.fft.rfft2`` 的分离式实现，输出半谱落在 ``dim2``。"""
    h, w = x.shape[dim1], x.shape[dim2]
    fr, fi = _rfft_1d(x, dim2, "ortho", opts, rfft)   # 实输入 -> 半谱
    xr, xi = _cfft(fr, fi, h, dim1, opts, rfft)       # 复数 -> 全谱
    if norm == "backward":
        scale = math.sqrt(h * w)
        xr, xi = xr * scale, xi * scale
    return xr, xi


def _irfft2_separable(xr: torch.Tensor, xi: torch.Tensor, h: int, w: int, dim1: int, dim2: int,
                      norm: str, opts: dict, irfft=None, rfft=None):  # noqa: PLR0913
    """``torch.fft.irfft2`` 的分离式实现，输入半谱落在 ``dim2``。"""
    ar, ai = _cifft(xr, xi, h, dim1, opts, rfft)       # 全谱复数逆
    out = _irfft_1d(ar, ai, w, dim2, "ortho", opts, irfft)   # 半谱 -> 实数
    if norm == "backward":
        out = out / math.sqrt(h * w)
    return out


# ---------------------------------------------------------------------------------------------
# 公开接口
# ---------------------------------------------------------------------------------------------

class _Rfft2CpuFunction(torch.autograd.Function):
    """``torch.fft.rfft2`` on CPU with a materialised gradient.

    ``rfft2``'s own backward is an ``irfft2``, which plans an MKL DFTI transform. oneMKL rejects
    the layout a ``.sum()`` produces upstream (stride-0 grad_output), so the differentiable case is
    routed here and the grad_output is made dense before the adjoint runs.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, h: int, w: int, dim1: int, dim2: int, norm: str):
        ctx.h, ctx.w, ctx.dim1, ctx.dim2, ctx.norm = h, w, dim1, dim2, norm
        ref = torch.fft.rfft2(x, s=(h, w), dim=(dim1, dim2), norm=norm)
        return ref.real, ref.imag

    @staticmethod
    def backward(ctx, grad_r, grad_i):
        h, w, d1, d2, norm = ctx.h, ctx.w, ctx.dim1, ctx.dim2, ctx.norm
        grad_r, grad_i = grad_r.contiguous(), grad_i.contiguous()
        # ad_rfft2(g) = irfft2(g * w) * h*w for "backward", with w = 1 at DC/Nyquist and 0.5
        # elsewhere (the inverse of irfft's mirror-bin weights).
        weights = grad_r.new_full((w // 2 + 1,), 0.5)
        weights[0] = 1.0
        if w % 2 == 0:
            weights[-1] = 1.0
        shape = [1] * grad_r.dim()
        shape[d2] = w // 2 + 1
        spec = torch.complex(grad_r, grad_i) * weights.view(shape)
        out = torch.fft.irfft2(spec, s=(h, w), dim=(d1, d2), norm=norm)
        if norm == "backward":
            out = out * (h * w)
        return out, None, None, None, None, None


def _use_npu_path(t: torch.Tensor) -> bool:
    return _is_npu_tensor(t) and _has_triton() and t.dtype == torch.float32


class _Irfft2CpuFunction(torch.autograd.Function):
    """``torch.fft.irfft2`` on CPU with a materialised gradient.

    The CPU fallback runs ``torch.fft.irfft2`` directly, so its backward is PyTorch's own and
    uses an MKL DFTI plan. oneMKL rejects some layouts that reach it: ``.sum()`` hands the
    transform a stride-0 grad_output, and the transform's own output gradient inherits that
    layout, which is what raises "Inconsistent configuration parameters". ``.contiguous()``
    here makes both the forward input and the backward result dense, so the DFTI plan always
    sees the same shape/strides in both directions.
    """

    @staticmethod
    def forward(ctx, spectrum: torch.Tensor, h: int, w: int, dim1: int, dim2: int, norm: str):
        ctx.h, ctx.w, ctx.dim1, ctx.dim2, ctx.norm = h, w, dim1, dim2, norm
        return torch.fft.irfft2(spectrum.contiguous(), s=(h, w), dim=(dim1, dim2), norm=norm)

    @staticmethod
    def backward(ctx, grad_y):
        h, w, d1, d2, norm = ctx.h, ctx.w, ctx.dim1, ctx.dim2, ctx.norm
        # The adjoint of irfft2 is a full fft2 of grad_output, restricted to the half spectrum and
        # weighted by irfft's sum weights (1 at DC and Nyquist, 2 elsewhere for the mirrored bins).
        grad_y = grad_y.contiguous()
        full = torch.fft.fft2(grad_y, dim=(d1, d2), norm=norm)
        spec = full.narrow(d2, 0, w // 2 + 1)
        # fft2 with the same ``norm`` already matches irfft2's scaling for "ortho"; "backward"
        # additionally leaves the h*w factor to undo.
        if norm == "backward":
            spec = spec / (h * w)
        weights = spec.new_full((w // 2 + 1,), 2.0)
        weights[0] = 1.0
        if w % 2 == 0:
            weights[-1] = 1.0
        shape = [1] * spec.dim()
        shape[d2] = w // 2 + 1
        return spec * weights.view(shape), None, None, None, None, None


def _check_norm(norm: str) -> None:
    if norm not in ("backward", "ortho"):
        raise ValueError(f"norm must be 'backward' or 'ortho', got {norm!r}")


def _rfft_opts(block_n: int | None, block_m: int | None, block_k: int | None) -> dict:
    return {"block_n": block_n, "block_m": block_m, "block_k": block_k}


def real_rfft2(
    x: torch.Tensor,
    s: Sequence[int] | None = None,
    dim: tuple[int, int] | Sequence[int] = (-2, -1),
    norm: str = "backward",
    block_n: int | None = None,
    block_m: int | None = None,
    block_k: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """二维实数正向 FFT，返回半谱的 ``(real, imag)``。

    语义与 ``torch.fft.rfft2(x, s=s, dim=dim, norm=norm)`` 一致：``dim[1]`` 轴的
    长度变为 ``n // 2 + 1``，``dim[0]`` 轴保持全谱长度。支持 autograd。

    | 参数 | 说明 |
    |------|------|
    | ``x`` | 实数张量；NPU 路径要求 ``float32`` |
    | ``s`` | 两个变换轴的信号长度，``None`` 表示沿用输入尺寸；长度不符时按 ``torch.fft`` 语义裁断或零填充 |
    | ``dim`` | 两个变换轴，必须互不相同 |
    | ``norm`` | ``"backward"``（默认，与 torch 一致）或 ``"ortho"`` |
    | ``block_n`` / ``block_m`` / ``block_k`` | 透传给一维 kernel 的分块参数 |

    非 NPU、Triton 不可用或输入不是 ``float32`` 时回退 ``torch.fft.rfft2``。
    """
    _check_norm(norm)
    if x.is_complex():
        raise TypeError(f"real_rfft2 expects a real input, got dtype {x.dtype}")
    dim1, dim2 = _resolve_dims(dim, x.dim())
    sizes = _resolve_sizes(s)
    if sizes is not None:
        x = _resize_dim(_resize_dim(x, dim1, sizes[0]), dim2, sizes[1])
    h, w = x.shape[dim1], x.shape[dim2]

    if not _use_npu_path(x):
        if not x.requires_grad:
            ref = torch.fft.rfft2(x, s=(h, w), dim=(dim1, dim2), norm=norm)
            return ref.real, ref.imag
        # Same MKL DFTI pitfall as real_irfft2's backward (rfft2's own backward is an irfft2,
        # which chokes on the stride-0 grad a ``.sum()`` produces), so go through a Function
        # that keeps the gradient dense in both directions.
        return _Rfft2CpuFunction.apply(x.contiguous(), h, w, dim1, dim2, norm)

    return _rfft2_separable(x, dim1, dim2, norm, _rfft_opts(block_n, block_m, block_k))


def real_irfft2(
    xr: torch.Tensor,
    xi: torch.Tensor,
    s: Sequence[int] | None = None,
    dim: tuple[int, int] | Sequence[int] = (-2, -1),
    norm: str = "backward",
    block_k: int | None = None,
    block_m: int | None = None,
    block_n: int | None = None,
) -> torch.Tensor:
    """二维实数逆 FFT：半谱 ``(xr, xi)`` -> 实数张量。

    语义与 ``torch.fft.irfft2(complex(xr, xi), s=s, dim=dim, norm=norm)`` 一致，
    包括 ``s`` 对半谱轴的裁断/零填充。支持 autograd。

    | 参数 | 说明 |
    |------|------|
    | ``xr`` / ``xi`` | 半谱实部、虚部，shape 必须相同 |
    | ``s`` | 重建后的两个轴长度，``None`` 时 ``s = (xr.shape[dim0], (xr.shape[dim1] - 1) * 2)`` |
    | ``dim`` | 两个变换轴，必须互不相同；半谱在 ``dim[1]`` 上 |
    | ``norm`` | ``"backward"``（默认，与 torch 一致）或 ``"ortho"`` |
    | ``block_k`` / ``block_m`` / ``block_n`` | 透传给一维 kernel 的分块参数 |

    非 NPU、Triton 不可用或输入不是 ``float32`` 时回退 ``torch.fft.irfft2``。
    """
    _check_norm(norm)
    if xr.shape != xi.shape:
        raise ValueError(f"xr and xi must have the same shape, got {xr.shape} and {xi.shape}")
    if xr.is_complex():
        raise TypeError(f"real_irfft2 expects real tensors, got dtype {xr.dtype}")
    dim1, dim2 = _resolve_dims(dim, xr.dim())
    sizes = _resolve_sizes(s)
    if sizes is None:
        h = xr.shape[dim1]
        w = (xr.shape[dim2] - 1) * 2
        if w <= 0:
            raise ValueError(
                f"cannot infer the output length from a half-spectrum of length "
                f"{xr.shape[dim2]} on dim[1]; pass s explicitly"
            )
    else:
        h, w = sizes
    # ``s`` 与 torch 一致地裁断/零填充两条谱轴。
    xr = _resize_dim(xr, dim1, h)
    xi = _resize_dim(xi, dim1, h)
    half = w // 2 + 1
    xr = _resize_dim(xr, dim2, half)
    xi = _resize_dim(xi, dim2, half)

    if not _use_npu_path(xr):
        # The CPU fallback is a plain ``torch.fft.irfft2``, so its backward is PyTorch's own and
        # plans an MKL DFTI transform. oneMKL rejects some layouts that reach that backward: a
        # ``.sum()`` upstream hands it a stride-0 grad_output, which then disagrees with the
        # forward plan's configuration. Route the differentiable case through a Function that
        # materialises the gradient, so the DFTI plan always sees dense, matching strides.
        if not (xr.requires_grad or xi.requires_grad):
            return torch.fft.irfft2(
                torch.complex(xr.contiguous(), xi.contiguous()), s=(h, w),
                dim=(dim1, dim2), norm=norm,
            )
        return _Irfft2CpuFunction.apply(
            torch.complex(xr, xi), h, w, dim1, dim2, norm
        )

    return _irfft2_separable(
        xr, xi, h, w, dim1, dim2, norm,
        _rfft_opts(block_n, block_m, block_k), real_irfft, real_rfft,
    )
