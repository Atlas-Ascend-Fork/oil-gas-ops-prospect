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

"""2D/3D linear interpolation as separable batched GEMMs on Ascend.

The operator is a drop-in subset of :func:`torch.nn.functional.interpolate` for
``mode='bilinear'`` (4D input) and ``mode='trilinear'`` (5D input).  Instead of
gathering and blending neighbouring samples with vector instructions, the
interpolation is expressed as one GEMM per spatial axis::

    Y = X @ W_axis

where ``W_axis`` is the ``[in_size, out_size]`` linear-interpolation matrix.
Each pass is a batched GEMM with ``M = batch * remaining_spatial``,
``K = in_size`` and ``N = out_size``, so the contraction runs on the Cube unit
through ``tl.dot``.  The three passes are applied last-axis first (W -> H -> D),
which keeps every pass a contiguous row-major GEMM.

The weight matrices are built by interpolating an identity matrix with PyTorch
itself, so the effective weights match ``F.interpolate`` bit-for-bit at fp32 and
the only remaining difference is the GEMM accumulation order.

Notes
-----
- NPU fast path: 5D ``float32`` trilinear, ``align_corners`` either value.
  The default is ``auto`` (Cube for outputs >=128 MiB).
- Everything else (CPU, float64/fp16/bf16, 4D bilinear, nearest/bicubic,
  ``scale_factor``, ``antialias``) falls back to ``F.interpolate``.
- The op is intentionally conservative for small tensors: the native vector
  kernel is already memory-bound, so the Cube path is used only when the output
  is at least 128 MiB.
- The NPU kernel uses a persistent grid of ``num_aicore`` programs. Tile counts
  and block sizes are ``tl.constexpr``; the actual shapes/strides are runtime
  arguments.
- Large outputs are chunked along ``B * C`` so a single kernel launch never
  addresses more than 1 GiB.

Environment variables
---------------------
- ``OIL_GAS_OPS_INTERPOLATE_IMPL=auto|triton|torch`` (default ``auto``)

Usage::

    from oil_gas_ops_prospect.interpolate import interpolate

    y = interpolate(x, size=(128, 128, 128), mode="trilinear", align_corners=False)

    # Convenience wrappers
    from oil_gas_ops_prospect.interpolate import bilinear, trilinear
    y2 = bilinear(x2, size=(256, 256))
    y3 = trilinear(x3, size=(160, 160, 160))
"""

from __future__ import annotations

import os
from typing import Sequence

import torch
import torch.nn.functional as F

try:  # pragma: no cover - CPU-only environments can still import this module.
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover
    triton = None
    tl = None

_IMPL_ENV = "OIL_GAS_OPS_INTERPOLATE_IMPL"

_MIN_BYTES = 128 << 20
_CHUNK_BYTES = 1 << 30
_REQUIRE_CUSTOM = False
_SUPPORTED_MODES = ("bilinear", "trilinear")

_WEIGHT_CACHE: dict[tuple[int, int, bool, str, bool], torch.Tensor] = {}


def _impl() -> str:
    return os.environ.get(_IMPL_ENV, "auto").strip().lower()


def _require_custom() -> bool:
    return _REQUIRE_CUSTOM


def _set_test_options(
    *,
    require_custom: bool | None = None,
    min_bytes: int | None = None,
    chunk_bytes: int | None = None,
) -> None:
    """Test-only knobs; not part of the public environment-variable surface."""
    global _REQUIRE_CUSTOM, _MIN_BYTES, _CHUNK_BYTES
    if require_custom is not None:
        _REQUIRE_CUSTOM = bool(require_custom)
    if min_bytes is not None:
        _MIN_BYTES = int(min_bytes)
    if chunk_bytes is not None:
        _CHUNK_BYTES = int(chunk_bytes)


def _has_triton() -> bool:
    return triton is not None and tl is not None


def _is_npu(x: torch.Tensor) -> bool:
    return x.device.type == "npu"


def _spatial_ndim(x: torch.Tensor) -> int:
    return x.dim() - 2


def _normalize_size(size: int | Sequence[int], ndim: int) -> tuple[int, ...]:
    if isinstance(size, int):
        return (size,) * ndim
    values = tuple(int(v) for v in size)
    if len(values) != ndim:
        raise ValueError(
            f"size must have {ndim} elements for a {ndim + 2}D input, got {size!r}"
        )
    return values


def _out_bytes(x: torch.Tensor, size: tuple[int, ...]) -> int:
    elems = 1
    for value in size:
        elems *= value
    elems *= x.shape[0] * x.shape[1]
    return elems * x.element_size()


def _torch_interpolate(
    x, size, scale_factor, mode, align_corners, antialias, recompute_scale_factor
):
    kwargs = {
        "mode": mode,
        "align_corners": align_corners if mode in _SUPPORTED_MODES else None,
        "antialias": antialias,
    }
    if size is not None:
        kwargs["size"] = size
    if scale_factor is not None:
        kwargs["scale_factor"] = scale_factor
    if recompute_scale_factor is not None:
        kwargs["recompute_scale_factor"] = recompute_scale_factor
    return F.interpolate(x, **kwargs)


def _make_weight(nin: int, nout: int, align_corners: bool) -> torch.Tensor:
    """Return the exact ``[nin, nout]`` linear-interpolation matrix PyTorch uses.

    Interpolating an identity matrix with ``F.interpolate`` is intentional: it
    reproduces PyTorch's float32 source-index / weight arithmetic exactly,
    whereas rebuilding the formula by hand differs by up to ~2e-6 and pushes the
    final error above 1e-5 for large shapes.
    """
    if nin == nout:
        return torch.eye(nin, dtype=torch.float32)
    eye = torch.eye(nin, dtype=torch.float32).view(nin, 1, 1, 1, nin)
    weight = F.interpolate(
        eye,
        size=(1, 1, nout),
        mode="trilinear",
        align_corners=align_corners,
    )
    return weight.reshape(nin, nout).contiguous()


def _get_weight(
    nin: int,
    nout: int,
    align_corners: bool,
    device: torch.device,
    transpose: bool = False,
) -> torch.Tensor:
    key = (nin, nout, bool(align_corners), str(device), bool(transpose))
    cached = _WEIGHT_CACHE.get(key)
    if cached is not None:
        return cached
    weight = _make_weight(nin, nout, align_corners)
    if transpose:
        weight = weight.t().contiguous()
    weight = weight.to(device)
    _WEIGHT_CACHE[key] = weight
    return weight


def _pow2(value: int) -> int:
    value = max(16, int(value))
    return 1 << (value - 1).bit_length()


def _blocks(kind: str, m_size: int, n_out: int, n_in: int) -> tuple[int, int, int]:
    """Pick GEMM tiles for one axis pass.

    ``last`` is a contiguous-row GEMM (interpolating the innermost axis),
    ``mid`` is the middle axis in a 3D pass and ``first`` is the outermost axis.
    The caps were tuned on 910B2 for the large 5D shapes (17/33/41 -> 64/128/160);
    the L0C accumulator must stay <= 128 KiB.
    """
    if kind == "last":
        bm = min(64, _pow2(m_size))
        bn = min(256, _pow2(n_out))
        bk = min(128, _pow2(n_in))
    elif kind == "mid":
        bm = min(256, _pow2(m_size))
        bn = min(128, _pow2(n_out))
        bk = min(64, _pow2(n_in))
    elif kind == "first":
        # With the persistent grid, 256x128 is faster than 512x64 and keeps
        # the fp32 result within 1e-6 on 910B2.
        bm = min(256, _pow2(m_size))
        bn = min(128, _pow2(n_out))
        bk = min(64, _pow2(n_in))
    else:
        raise ValueError(f"unknown axis kind: {kind!r}")
    return bm, bn, bk


_CORE_COUNT = None


def _num_cores() -> int:
    """Return the AI-core count used as the persistent-grid width."""
    global _CORE_COUNT
    if _CORE_COUNT is not None:
        return _CORE_COUNT
    count = 24
    if _has_triton():
        try:
            props = triton.runtime.driver.active.utils.get_device_properties(0)
            for key in ("num_aicore", "cube_core_num", "aicore_num"):
                if key in props:
                    count = int(props[key])
                    break
        except Exception:
            pass
    _CORE_COUNT = count
    return count


if _has_triton():

    @triton.jit
    def _interp_axis_kernel(
        x_ptr,
        w_ptr,
        y_ptr,
        gsx,
        gsy,
        sm_in,
        kstride,
        msize,
        nin,
        nout,
        sm_out,
        sn_out,
        NG: tl.constexpr,
        NM: tl.constexpr,
        NN: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """Persistent grid: one program per AI core, looping over tiles.

        A full ``(groups, tiles_m, tiles_n)`` grid launches tens of thousands of
        tiny programs for large 5D shapes; the launch and tile-decode overhead is
        larger than the Cube work itself. Keeping one program per AI core and
        walking the tile space gives ~1.4-1.8x over the per-tile launch while
        preserving fp32 accuracy.
        """
        pid = tl.program_id(0)
        nprog = tl.num_programs(0)
        total: tl.constexpr = NG * NM * NN
        offs_m0 = tl.arange(0, BLOCK_M)
        offs_n0 = tl.arange(0, BLOCK_N)
        offs_k0 = tl.arange(0, BLOCK_K)
        for tile in range(pid, total, nprog):
            nb = tile % NN
            t = tile // NN
            mb = t % NM
            g = t // NM
            offs_m = (mb * BLOCK_M + offs_m0).to(tl.int64)
            offs_n = (nb * BLOCK_N + offs_n0).to(tl.int64)
            offs_k = offs_k0.to(tl.int64)
            xb = g.to(tl.int64) * gsx
            yb = g.to(tl.int64) * gsy
            acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
            for k0 in range(0, nin, BLOCK_K):
                k = k0 + offs_k
                x = tl.load(
                    x_ptr + xb + offs_m[:, None] * sm_in + k[None, :] * kstride,
                    mask=(offs_m[:, None] < msize) & (k[None, :] < nin),
                    other=0.0,
                )
                w = tl.load(
                    w_ptr + k[:, None] * nout + offs_n[None, :],
                    mask=(k[:, None] < nin) & (offs_n[None, :] < nout),
                    other=0.0,
                )
                acc = tl.dot(x, w, acc)
            tl.store(
                y_ptr + yb + offs_m[:, None] * sm_out + offs_n[None, :] * sn_out,
                acc,
                mask=(offs_m[:, None] < msize) & (offs_n[None, :] < nout),
            )


def _axis_pass(
    x: torch.Tensor,
    weight: torch.Tensor,
    y: torch.Tensor,
    *,
    gsx: int,
    gsy: int,
    sm_in: int,
    kstride: int,
    msize: int,
    nin: int,
    nout: int,
    sm_out: int,
    sn_out: int,
    groups: int,
    kind: str,
    blocks: tuple[int, int, int] | None = None,
) -> None:
    if blocks is None:
        bm, bn, bk = _blocks(kind, msize, nout, nin)
    else:
        bm, bn, bk = blocks
    nm = triton.cdiv(msize, bm)
    nn = triton.cdiv(nout, bn)
    total = groups * nm * nn
    grid = (min(total, _num_cores()),)
    _interp_axis_kernel[grid](
        x,
        weight,
        y,
        gsx,
        gsy,
        sm_in,
        kstride,
        msize,
        nin,
        nout,
        sm_out,
        sn_out,
        NG=groups,
        NM=nm,
        NN=nn,
        BLOCK_M=bm,
        BLOCK_N=bn,
        BLOCK_K=bk,
    )


_KERNEL_CHECK_OK = None
_KERNEL_CHECK_ERROR = None


def _kernel_fp32_ok(
    x: torch.Tensor, size: tuple[int, int, int], align_corners: bool
) -> bool:
    """Check the real W/H/D kernel once on a one-channel slice of this input.

    A plain 16x16/64x64 ``tl.dot`` is not representative: on this compiler it
    can take the HF32 tiny-dot path even when the tiled interpolation kernel is
    IEEE.  Using ``x[:1, :1]`` exercises the exact spatial shape/tiles and
    reuses the compiled kernel for the real call, so the only extra cost is one
    native interpolation of a one-channel slice.
    """
    global _KERNEL_CHECK_OK
    if _KERNEL_CHECK_OK is not None:
        return _KERNEL_CHECK_OK
    if not _has_triton():
        return False
    if x.dim() != 5:
        return True
    try:
        probe = x[:1, :1]
        ref = F.interpolate(
            probe.cpu(), size=size, mode="trilinear", align_corners=align_corners
        )
        error = float("inf")
        for _ in range(2):
            got = _forward_3d(probe, size, align_corners)
            error = (got.cpu() - ref).abs().max().item()
            if error <= 1e-5:
                break
        _KERNEL_CHECK_ERROR = f"max_abs={error:.6e}"
        _KERNEL_CHECK_OK = error <= 1e-5
    except Exception:
        _KERNEL_CHECK_OK = False
    return _KERNEL_CHECK_OK


def _chunk_size(n_rows: int, out_elems_per_row: int, element_size: int) -> int:
    budget = _CHUNK_BYTES
    per_row = max(1, out_elems_per_row * element_size)
    return max(1, min(int(n_rows), budget // per_row))


def _forward_3d(
    x: torch.Tensor, size: tuple[int, int, int], align_corners: bool
) -> torch.Tensor:
    batch, channels = x.shape[0], x.shape[1]
    d_in, h_in, w_in = x.shape[2:]
    d_out, h_out, w_out = size
    x = x.contiguous().reshape(batch * channels, d_in, h_in, w_in)
    y = torch.empty(
        (batch * channels, d_out, h_out, w_out), device=x.device, dtype=x.dtype
    )
    w_w = _get_weight(w_in, w_out, align_corners, x.device)
    w_h = _get_weight(h_in, h_out, align_corners, x.device)
    w_d = _get_weight(d_in, d_out, align_corners, x.device)
    chunk = _chunk_size(x.shape[0], d_out * h_out * w_out, x.element_size())
    for n0 in range(0, x.shape[0], chunk):
        n1 = min(x.shape[0], n0 + chunk)
        count = n1 - n0
        xc = x[n0:n1]
        yc = y[n0:n1]

        if w_out == w_in:
            y1 = xc
        else:
            y1 = torch.empty((count, d_in, h_in, w_out), device=x.device, dtype=x.dtype)
            _axis_pass(
                xc,
                w_w,
                y1,
                gsx=h_in * w_in,
                gsy=h_in * w_out,
                sm_in=w_in,
                kstride=1,
                msize=h_in,
                nin=w_in,
                nout=w_out,
                sm_out=w_out,
                sn_out=1,
                groups=count * d_in,
                kind="last",
            )

        if h_out == h_in:
            y2 = y1
        else:
            y2 = torch.empty(
                (count, d_in, h_out, w_out), device=x.device, dtype=x.dtype
            )
            _axis_pass(
                y1,
                w_h,
                y2,
                gsx=h_in * w_out,
                gsy=h_out * w_out,
                sm_in=1,
                kstride=w_out,
                msize=w_out,
                nin=h_in,
                nout=h_out,
                sm_out=1,
                sn_out=w_out,
                groups=count * d_in,
                kind="mid",
            )

        if d_out == d_in:
            yc.copy_(y2)
        else:
            _axis_pass(
                y2,
                w_d,
                yc,
                gsx=d_in * h_out * w_out,
                gsy=d_out * h_out * w_out,
                sm_in=1,
                kstride=h_out * w_out,
                msize=h_out * w_out,
                nin=d_in,
                nout=d_out,
                sm_out=1,
                sn_out=h_out * w_out,
                groups=count,
                kind="first",
            )
    return y.reshape(batch, channels, d_out, h_out, w_out)


def _backward_3d(
    grad: torch.Tensor,
    input_size: tuple[int, int, int],
    out_size: tuple[int, int, int],
    align_corners: bool,
) -> torch.Tensor:
    batch, channels = grad.shape[0], grad.shape[1]
    d_in, h_in, w_in = input_size
    d_out, h_out, w_out = out_size
    grad = grad.contiguous().reshape(batch * channels, d_out, h_out, w_out)
    grad_x = torch.empty(
        (batch * channels, d_in, h_in, w_in), device=grad.device, dtype=grad.dtype
    )
    w_d = _get_weight(d_in, d_out, align_corners, grad.device, transpose=True)
    w_h = _get_weight(h_in, h_out, align_corners, grad.device, transpose=True)
    w_w = _get_weight(w_in, w_out, align_corners, grad.device, transpose=True)
    # Chunk on the larger of the grad-output and grad-input row footprints so
    # the int32 offset arithmetic stays inside one 1 GiB window.
    row_elems = max(d_out * h_out * w_out, d_in * h_in * w_in)
    chunk = _chunk_size(grad.shape[0], row_elems, grad.element_size())
    for n0 in range(0, grad.shape[0], chunk):
        n1 = min(grad.shape[0], n0 + chunk)
        count = n1 - n0
        gc = grad[n0:n1]
        gxc = grad_x[n0:n1]

        # W^T pass on the output tensor is the adjoint of the last forward pass.
        if w_out == w_in:
            g1 = gc
        else:
            g1 = torch.empty(
                (count, d_out, h_out, w_in), device=grad.device, dtype=grad.dtype
            )
            # Adjoint W pass.  Merge Dout*Hout into M (like the forward W
            # pass) so the dot has M=512 instead of one small group per
            # (d, out-chunk).  This is the largest backward win.
            _axis_pass(
                gc,
                w_w,
                g1,
                gsx=d_out * h_out * w_out,
                gsy=d_out * h_out * w_in,
                sm_in=w_out,
                kstride=1,
                msize=d_out * h_out,
                nin=w_out,
                nout=w_in,
                sm_out=w_in,
                sn_out=1,
                groups=count,
                kind="last",
                blocks=(
                    min(512, _pow2(d_out * h_out)),
                    min(64, _pow2(w_in)),
                    min(64, _pow2(w_out)),
                ),
            )

        if h_out == h_in:
            g2 = g1
        else:
            g2 = torch.empty(
                (count, d_out, h_in, w_in), device=grad.device, dtype=grad.dtype
            )
            _axis_pass(
                g1,
                w_h,
                g2,
                gsx=h_out * w_in,
                gsy=h_in * w_in,
                sm_in=1,
                kstride=w_in,
                msize=w_in,
                nin=h_out,
                nout=h_in,
                sm_out=1,
                sn_out=w_in,
                groups=count * d_out,
                kind="mid",
                # K=Hout fits in one K tile; N and M are only 33/41 here.
                blocks=(
                    min(64, _pow2(w_in)),
                    min(64, _pow2(h_in)),
                    min(256, _pow2(h_out)),
                ),
            )

        if d_out == d_in:
            gxc.copy_(g2)
        else:
            _axis_pass(
                g2,
                w_d,
                gxc,
                gsx=d_out * h_in * w_in,
                gsy=d_in * h_in * w_in,
                sm_in=1,
                kstride=h_in * w_in,
                msize=h_in * w_in,
                nin=d_out,
                nout=d_in,
                sm_out=1,
                sn_out=h_in * w_in,
                groups=count,
                kind="first",
                # The final adjoint has a tiny N=Din; a taller M tile is
                # marginally better than the forward D-pass block.
                blocks=(
                    min(512, _pow2(h_in * w_in)),
                    min(64, _pow2(d_in)),
                    min(64, _pow2(d_out)),
                ),
            )
    return grad_x.reshape(batch, channels, d_in, h_in, w_in)


class _InterpolateFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, size, mode, align_corners):
        ctx.input_size = tuple(x.shape[2:])
        ctx.out_size = tuple(size)
        ctx.align_corners = bool(align_corners)
        return _forward_3d(x, ctx.out_size, ctx.align_corners)

    @staticmethod
    def backward(ctx, grad):
        grad_x = _backward_3d(grad, ctx.input_size, ctx.out_size, ctx.align_corners)
        return grad_x, None, None, None


def interpolate(
    x: torch.Tensor,
    size: int | Sequence[int] | None = None,
    scale_factor: float | Sequence[float] | None = None,
    mode: str = "trilinear",
    align_corners: bool = False,
    antialias: bool = False,
    recompute_scale_factor: bool | None = None,
) -> torch.Tensor:
    """Drop-in subset of ``F.interpolate`` backed by separable Cube GEMMs.

    ``size`` selects the exact output shape.  ``scale_factor`` and the other
    rare signatures are forwarded to ``F.interpolate`` and use the native
    kernel; the fast path is intentionally limited to the exact-size linear
    interpolation of 5D float32 tensors.
    """
    ndim = _spatial_ndim(x)
    if ndim not in (2, 3):
        raise ValueError(
            f"interpolate expects a 4D or 5D input, got shape {tuple(x.shape)}"
        )
    expected_mode = "bilinear" if ndim == 2 else "trilinear"
    if mode != expected_mode:
        return _torch_interpolate(
            x,
            size,
            scale_factor,
            mode,
            align_corners,
            antialias,
            recompute_scale_factor,
        )
    if size is None or scale_factor is not None:
        return _torch_interpolate(
            x,
            size,
            scale_factor,
            mode,
            align_corners,
            antialias,
            recompute_scale_factor,
        )
    size = _normalize_size(size, ndim)

    use_custom = (
        _impl() in ("auto", "triton")
        and _is_npu(x)
        and _has_triton()
        and x.is_floating_point()
        and x.dtype == torch.float32
        and not antialias
    )
    if ndim == 2:
        use_custom = False
    if use_custom and _out_bytes(x, size) < _MIN_BYTES:
        use_custom = False
    if use_custom and not _kernel_fp32_ok(x, size, align_corners):
        if _require_custom():
            raise RuntimeError(
                "Triton interpolation kernel failed the fp32 self-check; "
                "refusing to use the Cube interpolate path"
            )
        use_custom = False

    if use_custom:
        try:
            if x.requires_grad:
                return _InterpolateFn.apply(x, size, mode, align_corners)
            return _forward_3d(x, size, align_corners)
        except Exception:
            if _require_custom():
                raise
    return _torch_interpolate(
        x,
        size,
        scale_factor,
        mode,
        align_corners,
        antialias,
        recompute_scale_factor,
    )


def bilinear(
    x: torch.Tensor,
    size: int | Sequence[int],
    align_corners: bool = False,
) -> torch.Tensor:
    """4D bilinear interpolation; falls back to ``F.interpolate`` on non-NPU."""
    return interpolate(x, size=size, mode="bilinear", align_corners=align_corners)


def trilinear(
    x: torch.Tensor,
    size: int | Sequence[int],
    align_corners: bool = False,
) -> torch.Tensor:
    """5D trilinear interpolation; uses Cube GEMMs on NPU for large outputs."""
    return interpolate(x, size=size, mode="trilinear", align_corners=align_corners)


__all__ = ["bilinear", "interpolate", "trilinear"]
