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

"""complex64 倒数的 Triton-Ascend 算子。

背景：910B2 的 AI Core 不支持 complex64，``1/z`` 原生走 ``aclnnReciprocal``
（Ai_Cpu），大张量耗时高且训练不稳定；拆解为 float32 的写法也要多 kernel 往返。
本算子把 ``1/z`` 拆解为 AI Core / Vector Core 上的单个 Triton kernel：

    1/(a + i*b) = (a - i*b) / (a^2 + b^2)

输入输出均为 interleaved complex64（实部在偶数 float 槽、虚部在奇数槽），kernel 以
``(BLOCK, 2)`` 平铺加载后 ``tl.split`` 拆实虚、计算后 ``tl.join`` 写回——纯连续
访存，实测带宽 ~400GB/s（910B2）。

数值上默认走 ``stable`` 路径：以 ``t = |a| + |b|`` 缩放分量后再求倒数，等价于

    1/z = (a/t - i*b/t) * (1/t) / ((a/t)^2 + (b/t)^2)

避免 ``a^2 + b^2`` 中间量在 |z| 超出 [~1e-19, ~1.8e19] 时上/下溢（naive 路径在这些
幅值下结果完全错误，stable 路径保持 ~2ulp）。``stable=False`` 退回朴素公式
（少一次除法，带宽高 ~10%）。

反向：f(z) = 1/z 全纯，f'(z) = -1/z^2，torch 复数 autograd 约定为
``grad_in = grad_out * conj(f')``，即 ``grad_z = -g * conj(r^2)``（r 为前向输出，
kernel 从保存的 r 直接计算）。
"""

from __future__ import annotations

import os
from typing import Union

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - CPU-only environments can still import this module.
    triton = None
    tl = None


Number = Union[int, float]

# 2*numel 起过 int32（尾块 foffs = offs*2 + 1 最大到 2*numel-1）。numel >= 2^30 时
# kernel 里的 float 槽偏移必须按 int64 计算。测试可调小该阈值以覆盖 int64 分支。
_INT64_THRESHOLD = 1 << 30

# 单个 complex64 张量超过 2^31 个元素（17GB+）不支持：块内 ``block_idx * BLOCK``
# 的 int32 算术本身会回绕。
_MAX_NUMEL = (1 << 31) - 1

# 每 tile 的复数元素数上限。910B2 的 Vector Core UB 约 196KB：stable 路径 ~10 个
# live float32 tile，多缓冲下 2048 是实测可用上限（3072 运行时 UB 溢出，表现为
# vector core exception 且进程无法恢复，故只能在 host 侧钳制，不做运行时回退）。
_MAX_BLOCK = 2048
_DEFAULT_BLOCK = 2048

_BLOCK_ENV = "OIL_GAS_OPS_RECIP_BLOCK"


def _has_triton() -> bool:
    return triton is not None and tl is not None


def _is_npu_tensor(x: torch.Tensor) -> bool:
    return x.device.type == "npu"


_VECTORCORE_CACHE: dict = {}


def _get_vectorcore_num(default: int = 48) -> int:
    """当前 NPU 的 Vector Core 数，作为 persistent kernel 的 grid。"""
    key = "vectorcore"
    cached = _VECTORCORE_CACHE.get(key)
    if cached is not None:
        return cached
    try:
        import torch_npu  # type: ignore[import-not-found]
        import triton.runtime.driver as driver
    except ImportError:
        return default
    if not (hasattr(torch, "npu") and torch.npu.is_available()):
        return default
    try:
        device = torch_npu.npu.current_device()
        properties = driver.active.utils.get_device_properties(device)
        num = int(properties.get("num_vectorcore", default))
    except Exception:
        return default
    _VECTORCORE_CACHE[key] = num
    return num


def _resolve_block(block_size: int | None) -> int:
    """解析 tile 大小：显式参数 > ``OIL_GAS_OPS_RECIP_BLOCK`` > 默认；钳制到
    ``[16, 2048]`` 内的 2 的幂（``tl.arange`` 要求）。"""
    if block_size is None:
        block_size = _DEFAULT_BLOCK
        try:
            env = int(os.environ.get(_BLOCK_ENV, ""))
            if env > 0:
                block_size = env
        except (TypeError, ValueError):
            pass
    block_size = int(block_size)
    if block_size < 16:
        return 16
    if block_size >= _MAX_BLOCK:
        return _MAX_BLOCK
    return 1 << (block_size.bit_length() - 1)


def _check_numel(numel: int) -> None:
    if numel > _MAX_NUMEL:
        raise ValueError(
            f"complex_reciprocal 最多支持 {_MAX_NUMEL} 个元素，得到 {numel}"
        )


if _has_triton():

    @triton.jit
    def _complex_reciprocal_kernel(
        z_ptr,
        out_ptr,
        n_elements,
        block_size: tl.constexpr,
        index64: tl.constexpr,
        stable: tl.constexpr,
    ):
        """out = 1/z，z/out 均为 interleaved complex64 的 float32 视图。"""
        pid = tl.program_id(0)
        num_core = tl.num_programs(0)
        num_blocks = tl.cdiv(n_elements, block_size)
        pair = tl.arange(0, 2)[None, :]
        for block_idx in range(pid, num_blocks, num_core):
            offs = block_idx * block_size + tl.arange(0, block_size)
            mask = offs < n_elements
            if index64:
                foffs = offs.to(tl.int64)[:, None] * 2 + pair
            else:
                foffs = offs[:, None] * 2 + pair
            m2 = mask[:, None]
            v = tl.load(z_ptr + foffs, mask=m2, other=0.0)
            a, b = tl.split(v)
            if stable:
                t = tl.abs(a) + tl.abs(b)
                inv_t = 1.0 / t
                sa = a * inv_t
                sb = b * inv_t
                w = inv_t / (sa * sa + sb * sb)
                out_r = sa * w
                out_i = -(sb * w)
            else:
                inv = 1.0 / (a * a + b * b)
                out_r = a * inv
                out_i = -b * inv
            tl.store(out_ptr + foffs, tl.join(out_r, out_i), mask=m2)

    @triton.jit
    def _complex_reciprocal_of_kernel(
        xr_ptr,
        xi_ptr,
        out_ptr,
        n_elements,
        xr_scalar,
        xi_scalar,
        block_size: tl.constexpr,
        index64: tl.constexpr,
        stable: tl.constexpr,
        xr_is_tensor: tl.constexpr,
        xi_is_tensor: tl.constexpr,
    ):
        """out = 1/(xr + i*xi)，xr/xi 为 planar float32（或标量），out interleaved。"""
        pid = tl.program_id(0)
        num_core = tl.num_programs(0)
        num_blocks = tl.cdiv(n_elements, block_size)
        pair = tl.arange(0, 2)[None, :]
        for block_idx in range(pid, num_blocks, num_core):
            offs = block_idx * block_size + tl.arange(0, block_size)
            mask = offs < n_elements
            if xr_is_tensor:
                a = tl.load(xr_ptr + offs, mask=mask, other=0.0)
            else:
                a = xr_scalar
            if xi_is_tensor:
                b = tl.load(xi_ptr + offs, mask=mask, other=0.0)
            else:
                b = xi_scalar
            if stable:
                t = tl.abs(a) + tl.abs(b)
                inv_t = 1.0 / t
                sa = a * inv_t
                sb = b * inv_t
                w = inv_t / (sa * sa + sb * sb)
                out_r = sa * w
                out_i = -(sb * w)
            else:
                inv = 1.0 / (a * a + b * b)
                out_r = a * inv
                out_i = -b * inv
            if index64:
                foffs = offs.to(tl.int64)[:, None] * 2 + pair
            else:
                foffs = offs[:, None] * 2 + pair
            tl.store(out_ptr + foffs, tl.join(out_r, out_i), mask=mask[:, None])

    @triton.jit
    def _complex_reciprocal_grad_kernel(
        r_ptr,
        g_ptr,
        gz_ptr,
        gxr_ptr,
        gxi_ptr,
        n_elements,
        block_size: tl.constexpr,
        index64: tl.constexpr,
        interleaved_output: tl.constexpr,
        write_gxi: tl.constexpr,
    ):
        """grad = -g * conj(r^2)。

        ``r``/``g`` 为前向输出与 grad_out（interleaved complex64 的 float32 视图）。
        ``interleaved_output`` 时写 ``gz_ptr``（对 z 的梯度，interleaved）；
        否则写 ``gxr_ptr``（对 xr 的梯度，planar float32），``write_gxi`` 时再写
        ``gxi_ptr``（对 xi 的梯度）。planar 模式下 d/dxr = Re(gz)、d/dxi = Im(gz)
        （``torch.complex`` 的反传不做 2 倍缩放）。
        """
        pid = tl.program_id(0)
        num_core = tl.num_programs(0)
        num_blocks = tl.cdiv(n_elements, block_size)
        pair = tl.arange(0, 2)[None, :]
        for block_idx in range(pid, num_blocks, num_core):
            offs = block_idx * block_size + tl.arange(0, block_size)
            mask = offs < n_elements
            if index64:
                foffs = offs.to(tl.int64)[:, None] * 2 + pair
            else:
                foffs = offs[:, None] * 2 + pair
            m2 = mask[:, None]
            rv = tl.load(r_ptr + foffs, mask=m2, other=0.0)
            gv = tl.load(g_ptr + foffs, mask=m2, other=0.0)
            rr, ri = tl.split(rv)
            gr, gi = tl.split(gv)
            rr2 = rr * rr - ri * ri
            ri2 = 2.0 * rr * ri
            gzr = -(gr * rr2 + gi * ri2)
            gzi = -(gi * rr2 - gr * ri2)
            if interleaved_output:
                tl.store(gz_ptr + foffs, tl.join(gzr, gzi), mask=m2)
            else:
                tl.store(gxr_ptr + offs, gzr, mask=mask)
                if write_gxi:
                    tl.store(gxi_ptr + offs, gzi, mask=mask)


def _flat(t: torch.Tensor) -> torch.Tensor:
    """展平为一维连续张量。

    ``reshape(-1)`` 对一维 stride-0（expand 出来的广播视图）张量是恒等操作，
    传给 kernel 会越界读单元素存储，因此非连续输入必须先真实拷贝。
    """
    if not t.is_contiguous():
        t = t.contiguous()
    return t.reshape(-1)


def _launch_reciprocal(z: torch.Tensor, out: torch.Tensor, n: int, stable: bool, block: int):
    _complex_reciprocal_kernel[(_get_vectorcore_num(),)](
        z.view(torch.float32),
        out.view(torch.float32),
        n,
        block_size=block,
        index64=n >= _INT64_THRESHOLD,
        stable=stable,
    )


def _launch_reciprocal_of(
    xr: torch.Tensor | None,
    xi: torch.Tensor | None,
    out: torch.Tensor,
    n: int,
    xr_scalar: float,
    xi_scalar: float,
    stable: bool,
    block: int,
):
    xr_flat = _flat(xr) if xr is not None else None
    xi_flat = _flat(xi) if xi is not None else None
    # 标量分支下对应指针不会被解引用，传另一个张量占位即可
    dummy = xr_flat if xr_flat is not None else (xi_flat if xi_flat is not None else _flat(out).view(torch.float32))
    _complex_reciprocal_of_kernel[(_get_vectorcore_num(),)](
        xr_flat if xr_flat is not None else dummy,
        xi_flat if xi_flat is not None else dummy,
        out.view(torch.float32),
        n,
        float(xr_scalar),
        float(xi_scalar),
        block_size=block,
        index64=n >= _INT64_THRESHOLD,
        stable=stable,
        xr_is_tensor=xr_flat is not None,
        xi_is_tensor=xi_flat is not None,
    )


def _launch_grad(r: torch.Tensor, g: torch.Tensor, gz: torch.Tensor | None,
                 gxr: torch.Tensor | None, gxi: torch.Tensor | None, n: int, block: int):
    interleaved = gz is not None
    # 未用到的输出指针以 r 的 float32 视图占位（constexpr 分支不会解引用）
    r_flat = _flat(r).view(torch.float32)
    _complex_reciprocal_grad_kernel[(_get_vectorcore_num(),)](
        r_flat,
        _flat(g).view(torch.float32),
        _flat(gz).view(torch.float32) if interleaved else r_flat,
        _flat(gxr) if gxr is not None else r_flat,
        _flat(gxi) if gxi is not None else r_flat,
        n,
        block_size=block,
        index64=n >= _INT64_THRESHOLD,
        interleaved_output=interleaved,
        write_gxi=gxi is not None,
    )


def _reciprocal_forward_impl(z: torch.Tensor, stable: bool, block: int) -> torch.Tensor:
    n = z.numel()
    _check_numel(n)
    z = z.contiguous()
    out = torch.empty_like(z)
    if n == 0:
        return out
    _launch_reciprocal(z, out, n, stable, block)
    return out


def _reciprocal_of_forward_impl(
    xr: torch.Tensor | None,
    xi: torch.Tensor | None,
    out_shape: torch.Size,
    n: int,
    xr_scalar: float,
    xi_scalar: float,
    stable: bool,
    block: int,
) -> torch.Tensor:
    _check_numel(n)
    out = torch.empty(out_shape, dtype=torch.complex64, device=xr.device if xr is not None else xi.device)
    if n == 0:
        return out
    _launch_reciprocal_of(xr, xi, out, n, xr_scalar, xi_scalar, stable, block)
    return out


if _has_triton():

    class _ComplexReciprocalFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, z, stable, block):
            ctx.stable = stable
            ctx.block = block
            out = _reciprocal_forward_impl(z, stable, block)
            ctx.save_for_backward(out)
            return out

        @staticmethod
        def backward(ctx, grad_out):
            (r,) = ctx.saved_tensors
            n = r.numel()
            gz = torch.empty_like(r)
            if n > 0:
                _launch_grad(r, grad_out.contiguous(), gz, None, None, n, ctx.block)
            # trailing None: stable / block 是非张量参数
            return gz, None, None

    class _ComplexReciprocalOfFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, xr, xi, stable, block):
            # 单元素张量按标量传给 kernel（读取其存储会越界），梯度走平面求和
            def as_full(t):
                return t if isinstance(t, torch.Tensor) and t.numel() > 1 else None

            def as_scalar(t):
                if isinstance(t, torch.Tensor):
                    return float(t.reshape(-1)[0].item())
                return float(t)

            xr_t = as_full(xr)
            xi_t = as_full(xi)
            ctx.xr_input = xr
            ctx.xi_input = xi
            ctx.block = block
            tensors = [t for t in (xr_t, xi_t) if t is not None]
            shape = tensors[0].shape if tensors else torch.Size(())
            n = tensors[0].numel() if tensors else 1
            out = _reciprocal_of_forward_impl(
                xr_t,
                xi_t,
                shape,
                n,
                as_scalar(xr) if xr_t is None else 0.0,
                as_scalar(xi) if xi_t is None else 0.0,
                stable,
                block,
            )
            ctx.save_for_backward(out)
            return out

        @staticmethod
        def backward(ctx, grad_out):
            (r,) = ctx.saved_tensors
            n = r.numel()
            grads: list[torch.Tensor | None] = [None, None]
            gxr = gxi = None
            if ctx.needs_input_grad[0] and isinstance(ctx.xr_input, torch.Tensor):
                gxr = torch.empty(n, dtype=torch.float32, device=r.device)
            if ctx.needs_input_grad[1] and isinstance(ctx.xi_input, torch.Tensor):
                gxi = torch.empty(n, dtype=torch.float32, device=r.device)
            if gxr is not None or gxi is not None:
                _launch_grad(r, grad_out.contiguous(), None, gxr, gxi, n, ctx.block)
                for idx, gx in ((0, gxr), (1, gxi)):
                    if gx is None:
                        continue
                    src = ctx.xr_input if idx == 0 else ctx.xi_input
                    if src.numel() == 1:
                        # 单元素张量（标量广播）输入：梯度是平面梯度的和
                        grads[idx] = gx.sum().to(src.device).reshape(src.shape)
                    else:
                        grads[idx] = gx.reshape(src.shape)
            # trailing None: stable / block 是非张量参数
            return grads[0], grads[1], None, None


def complex_reciprocal(
    z,
    xi=None,
    *,
    stable: bool = True,
    block_size: int | None = None,
) -> torch.Tensor:
    """complex64 倒数，按输入形式自动分发：

    - ``complex_reciprocal(z)``: ``1/z``，``z`` 为 complex64 张量。NPU 上由单个
      Triton kernel（Vector Core）完成；CPU 或无 triton 环境回退 ``1 / z``。
    - ``complex_reciprocal(xr, xi)``: ``1 / (xr + i*xi)``，实虚部分开给出，
      省去先构造 complex64 的 Ai_Cpu 算子；分母常见形态为「实张量 + 常数虚部」
      （如 ``1/(a - i*c)``），虚部可为标量。

    :param z: complex64 张量（单参形式）；或 float32 实部张量/标量（双参形式）。
    :param xi: 双参形式下的虚部（float32 张量/标量，可广播），单参形式不传。
    :param stable: True（默认）用 ``t=|a|+|b|`` 缩放避免 ``a^2+b^2`` 上下溢；
        False 用朴素公式（带宽高约 10%，|z| 超出 ~[1e-19, 1.8e19] 时会错）。
    :param block_size: 每 tile 复数元素数（2 的幂，钳制到 [16, 2048]），
        默认 2048，可用 ``OIL_GAS_OPS_RECIP_BLOCK`` 覆盖。
    """
    if xi is not None:
        return _complex_reciprocal_parts(z, xi, stable=stable, block_size=block_size)
    if not isinstance(z, torch.Tensor):
        raise TypeError(
            f"complex_reciprocal 期望 torch.Tensor，得到 {type(z).__name__}；"
            f"实虚分离输入请用 complex_reciprocal(xr, xi)"
        )
    if z.dtype != torch.complex64 or not (_is_npu_tensor(z) and _has_triton()):
        return 1 / z
    return _ComplexReciprocalFunction.apply(z, stable, _resolve_block(block_size))


def _complex_reciprocal_parts(
    xr: torch.Tensor | Number,
    xi: torch.Tensor | Number,
    *,
    stable: bool = True,
    block_size: int | None = None,
) -> torch.Tensor:
    """``1 / (xr + i*xi)`` 的双参实现（经 ``complex_reciprocal(xr, xi)`` 分发）。"""
    xr_is_tensor = isinstance(xr, torch.Tensor)
    xi_is_tensor = isinstance(xi, torch.Tensor)
    if not (xr_is_tensor or xi_is_tensor):
        raise TypeError("complex_reciprocal(xr, xi) 的 xr/xi 至少需要一个 torch.Tensor")
    full_tensors = [t for t in (xr, xi) if isinstance(t, torch.Tensor) and t.numel() > 1]
    if len(full_tensors) == 2:
        xr_t, xi_t = full_tensors
        if xr_t.shape != xi_t.shape:
            raise ValueError(
                f"xr/xi 形状需一致（单元素张量按标量广播），得到 "
                f"{tuple(xr_t.shape)} 与 {tuple(xi_t.shape)}"
            )
        if xr_t.device != xi_t.device:
            raise ValueError(f"xr/xi 需在同一设备，得到 {xr_t.device} 与 {xi_t.device}")
        if xr_t.dtype != xi_t.dtype:
            raise ValueError(f"xr/xi dtype 需一致，得到 {xr_t.dtype} 与 {xi_t.dtype}")

    block = _resolve_block(block_size)
    # NPU kernel 路径只看"完整张量"输入的设备与 dtype（单元素张量按标量取值）
    if full_tensors:
        t = full_tensors[0]
        device_ok = _is_npu_tensor(t) and _has_triton()
        dtype_ok = t.dtype == torch.float32
    else:
        # 两个单元素张量：走 torch 标量路径即可
        device_ok = False
        dtype_ok = False
    if not (device_ok and dtype_ok):
        return _reciprocal_of_torch(xr, xi)
    return _ComplexReciprocalOfFunction.apply(xr, xi, stable, block)


def _reciprocal_of_torch(xr: torch.Tensor | Number, xi: torch.Tensor | Number) -> torch.Tensor:
    """``complex_reciprocal(xr, xi)`` 双参形式的 torch 参考实现（CPU / 非 NPU / 非 float32）。"""

    def as_full(t):
        return t if isinstance(t, torch.Tensor) and t.numel() > 1 else None

    def as_scalar_tensor(t, ref: torch.Tensor) -> torch.Tensor:
        if isinstance(t, torch.Tensor):
            # reshape(()) 把 (1,) 归一成 0-dim，expand 保持 autograd 连接
            return t.reshape(()) if t.numel() == 1 else t
        return ref.new_full((), float(t))

    xr_t = as_full(xr)
    xi_t = as_full(xi)
    if xr_t is not None and xi_t is not None:
        return _reciprocal_of_torch_full(xr_t, xi_t)
    if xr_t is None and xi_t is None:
        # 两个标量/单元素张量：直接按复标量计算
        ref = xr if isinstance(xr, torch.Tensor) else xi
        vals = [float(t.reshape(-1)[0].item()) if isinstance(t, torch.Tensor) else float(t)
                for t in (xr, xi)]
        value = 1 / complex(vals[0], vals[1])
        return torch.tensor(value, dtype=torch.complex64, device=ref.device)
    full = xr_t if xr_t is not None else xi_t
    other = xi if xr_t is not None else xr
    other_t = as_scalar_tensor(other, full).to(full.device)
    if full.numel() == 0:
        return torch.empty(full.shape, dtype=torch.complex64, device=full.device)
    if xr_t is not None:
        return _reciprocal_of_torch_full(xr_t, other_t.expand(full.shape))
    return _reciprocal_of_torch_full(other_t.expand(full.shape), xi_t)


def _reciprocal_of_torch_full(xr: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
    if xr.numel() == 0:
        return torch.empty(xr.shape, dtype=torch.complex64, device=xr.device)
    return 1 / torch.complex(xr, xi)
