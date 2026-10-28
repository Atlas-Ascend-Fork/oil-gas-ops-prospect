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

"""complex64 指数的 Triton-Ascend 算子。

背景：910B2 的 AI Core 不支持 complex64，``torch.exp`` 原生走
``aclnnExp``（Ai_Cpu 标量协处理器），大张量耗时高（实测 4M 元素 66ms，
比 f32 慢三个数量级）。本算子把 ``exp`` 拆解为 Vector Core 上的单个
Triton kernel：

    e^(a + i*b) = e^a * (cos b + i*sin b)

输入输出均为 interleaved complex64（实部在偶数 float 槽、虚部在奇数槽），
kernel 以 ``(BLOCK, 2)`` 平铺加载后 ``tl.split`` 拆实虚、计算后 ``tl.join``
写回——纯连续访存。

数值语义对齐 CPU ``cexpf``（即 ``torch.exp(complex64)``）。朴素
``e^a * cos b / e^a * sin b`` 在两处与 ``cexpf`` 不一致，kernel 里都做了处理：

* ``b == 0``（含 ``-0.0``）：``inf * 0`` 会得到 NaN，而 ``cexpf`` 直接返回
  ``exp(a) + i*b``。故该分支跳过 cos/sin，虚部直接取 ``b``（保留 ``-0.0``）。
* ``a`` 落在 float32 exp 值域外（``a > ln(FLT_MAX)``）：``exp(a)`` 先溢出成
  inf，再乘 ``cos b``（可能远小于 1）就丢失了「实部不上溢」的正确结果。
  改为 ``s = exp(a/4)`` 后连乘四次——除以 4 是精确的 2 的幂缩放，四次部分积
  单调不减，故 "部分积已溢出 ⇔ 真值已溢出"，溢出语义与 ``cexpf`` 一致。
* ``b`` 为 ``±inf`` / NaN 时 ``cos/sin`` 返回 NaN，需按 ``cexpf`` 的
  ``(a, b)`` 组合特判（``-inf`` 得 ``0``，``+inf`` 得 ``inf``，其余 NaN）。

反向：f(z) = e^z 全纯，f'(z) = e^z = w（前向输出）。torch 复数 autograd
约定 ``grad_in = grad_out * conj(f')``，即 ``grad_z = g * conj(w)``，kernel
从保存的前向输出直接计算，无需重算 exp。

二阶导：反向 kernel 只能算一阶 ``g * conj(w)``，其结果不带 autograd 图。
``backward`` 在自身处于 grad 模式（``create_graph=True`` / ``gradcheck``）时改用
等价的 torch 组合式 ``g * conj(w)``，这样二阶导（以及更高阶）由 PyTorch 自动
求导接管；不需要高阶导的常规训练仍走 kernel。

已知限制：triton-ascend 后端的 ``tl.sin`` / ``tl.cos`` 在 ``|b| >= 2^25`` 之后
没有参数归约，结果从该量级起与 ``torch.exp`` 不一致（后端固有限制）。
"""

from __future__ import annotations

import os

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - CPU-only environments can still import this module.
    triton = None
    tl = None


# 2*numel 起过 int32（尾块 foffs = offs*2 + 1 最大到 2*numel-1）。numel >= 2^30 时
# kernel 里的 float 槽偏移必须按 int64 计算。测试可调小该阈值以覆盖 int64 分支。
_INT64_THRESHOLD = 1 << 30

# 单个 complex64 张量超过 2^31 个元素（17GB+）不支持：块内 ``block_idx * BLOCK``
# 的 int32 算术本身会回绕。
_MAX_NUMEL = (1 << 31) - 1

# 每 tile 的复数元素数上限。910B2 的 Vector Core UB 约 196KB：exp 前向 ~6 个
# live float32 tile，多缓冲下 2048 是实测可用上限（与 complex_reciprocal 一致，
# 超限会 UB 溢出且进程无法恢复，故只能在 host 侧钳制）。
_MAX_BLOCK = 2048
_DEFAULT_BLOCK = 2048

_BLOCK_ENV = "OIL_GAS_OPS_EXP_BLOCK"

# ``cexpf`` 的上溢分界：``ln(FLT_MAX)``，float32 位模式 0x42b17218。
# ``a`` 超过它时 ``exp(a)`` 先溢出，但 ``exp(a)·cos b`` 的真值仍可能有限。
_EXP_OVFL = 88.72283935546875


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
    """解析 tile 大小：显式参数 > ``OIL_GAS_OPS_EXP_BLOCK`` > 默认；钳制到
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
        raise ValueError(f"complex_exp 最多支持 {_MAX_NUMEL} 个元素，得到 {numel}")


if _has_triton():
    # kernel 内只能引用 constexpr 全局量（见 triton 的 NameError 提示）。
    # NaN 不进 constexpr：triton 用 ``!=`` 判断全局量是否变化，NaN 恒不等会让
    # 缓存键每次都「已变化」，故 NaN/inf 以字面量写在 kernel 体内。
    _TL_EXP_OVFL = tl.constexpr(_EXP_OVFL)

    @triton.jit
    def _complex_exp_kernel(
        z_ptr,
        out_ptr,
        n_elements,
        block_size: tl.constexpr,
        index64: tl.constexpr,
    ):
        """out = exp(z)，z/out 均为 interleaved complex64 的 float32 视图。"""
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
            ea = tl.exp(a)
            cr = tl.cos(b)
            si = tl.sin(b)
            out_r = ea * cr
            out_i = ea * si
            # 上溢带：exp(a) 已溢出但真值可能有限。用 s = exp(a/4) 连乘四次，
            # 部分积单调不减，故「部分积溢出 ⇔ 真值溢出」，与 cexpf 的缩放
            # 路径语义一致；除以 4 是精确的 2 的幂缩放，不引入额外误差。
            ovf = a > _TL_EXP_OVFL
            if tl.max(ovf.to(tl.int32), axis=0) > 0:
                s = tl.exp(a * 0.25)
                sc = ((cr * s) * s) * s * s
                ss = ((si * s) * s) * s * s
                out_r = tl.where(ovf, sc, out_r)
                out_i = tl.where(ovf, ss, out_i)
            # b == 0（含 -0.0）：cexpf 直接返回 exp(a) + i*b，避免 inf*0 -> NaN。
            bz = b == 0.0
            out_r = tl.where(bz, ea, out_r)
            out_i = tl.where(bz, b, out_i)
            # b 非有限：cos/sin 为 NaN，cexpf 按实部特判。
            b_nan = b != b
            b_inf = (b == float("inf")) | (b == float("-inf"))
            b_bad = b_nan | b_inf
            if tl.max(b_bad.to(tl.int32), axis=0) > 0:
                a_pos_inf = a == float("inf")
                a_neg_inf = a == float("-inf")
                nan_r = tl.full(a.shape, float("nan"), tl.float32)
                zero_r = tl.zeros(a.shape, tl.float32)
                inf_r = tl.full(a.shape, float("inf"), tl.float32)
                nr = tl.where(a_neg_inf, zero_r, tl.where(a_pos_inf, inf_r, nan_r))
                # -inf 时虚部为带符号 0（符号取 b），其余 NaN。
                ni = tl.where(a_neg_inf, tl.where(b < 0.0, -0.0, 0.0), nan_r)
                out_r = tl.where(b_bad, nr, out_r)
                out_i = tl.where(b_bad, ni, out_i)
            tl.store(out_ptr + foffs, tl.join(out_r, out_i), mask=m2)

    @triton.jit
    def _complex_exp_grad_kernel(
        w_ptr,
        g_ptr,
        gz_ptr,
        n_elements,
        block_size: tl.constexpr,
        index64: tl.constexpr,
    ):
        """grad = g * conj(w)，w 为前向输出 exp(z)，g 为 grad_out。

        ``w``/``g``/``gz`` 均为 interleaved complex64 的 float32 视图：
        g*conj(w) = (gr + i*gi)(wr - i*wi) = (gr*wr + gi*wi) + i*(gi*wr - gr*wi)。
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
            wv = tl.load(w_ptr + foffs, mask=m2, other=0.0)
            gv = tl.load(g_ptr + foffs, mask=m2, other=0.0)
            wr, wi = tl.split(wv)
            gr, gi = tl.split(gv)
            gzr = gr * wr + gi * wi
            gzi = gi * wr - gr * wi
            tl.store(gz_ptr + foffs, tl.join(gzr, gzi), mask=m2)


def _flat(t: torch.Tensor) -> torch.Tensor:
    """展平为一维连续张量。

    ``reshape(-1)`` 对一维 stride-0（expand 出来的广播视图）张量是恒等操作，
    传给 kernel 会越界读单元素存储，因此非连续输入必须先真实拷贝。

    conjugate / neg bit 的视图（复数反向里很常见）也必须先物化：``contiguous()``
    不会清掉这两个 bit，后续 ``view(torch.float32)`` 会直接报错。
    """
    if t.is_conj():
        t = t.resolve_conj()
    if t.is_neg():
        t = t.resolve_neg()
    if not t.is_contiguous():
        t = t.contiguous()
    return t.reshape(-1)


def _launch_exp(z: torch.Tensor, out: torch.Tensor, n: int, block: int):
    # 经 _flat 先 reshape(-1)：0-dim（标量）复数的 view(torch.float32) 要求 dim >= 1，
    # 直接 view 会报 "self.dim() cannot be 0 to view ComplexFloat as Float"。
    _complex_exp_kernel[(_get_vectorcore_num(),)](
        _flat(z).view(torch.float32),
        _flat(out).view(torch.float32),
        n,
        block_size=block,
        index64=n >= _INT64_THRESHOLD,
    )


def _launch_grad(w: torch.Tensor, g: torch.Tensor, gz: torch.Tensor, n: int, block: int):
    _complex_exp_grad_kernel[(_get_vectorcore_num(),)](
        _flat(w).view(torch.float32),
        _flat(g).view(torch.float32),
        _flat(gz).view(torch.float32),
        n,
        block_size=block,
        index64=n >= _INT64_THRESHOLD,
    )


def _exp_forward_impl(z: torch.Tensor, block: int) -> torch.Tensor:
    n = z.numel()
    _check_numel(n)
    z = z.contiguous()
    # 用 torch.empty(shape, ...) 而非 empty_like：torch_npu 的 empty_like 会尝试沿用
    # 输入的 internal format，触发 "Cannot create tensor with interal format ..." 告警。
    out = torch.empty(z.shape, dtype=z.dtype, device=z.device)
    if n == 0:
        return out
    _launch_exp(z, out, n, block)
    return out


if _has_triton():

    class _ComplexExpFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, z, block):
            ctx.block = block
            out = _exp_forward_impl(z, block)
            ctx.save_for_backward(out)
            return out

        @staticmethod
        def backward(ctx, grad_out):
            (w,) = ctx.saved_tensors
            n = w.numel()
            # 反向处于 grad 模式 ⇔ 调用方传了 create_graph=True（或 gradcheck）。
            # kernel 只能产出一阶数值、结果 detached，无法再求导；此时改走等价
            # 的 torch 组合式 g * conj(w)：w 是前向输出、本身在 autograd 图上，
            # 于是二阶及更高阶导由 PyTorch 接管，h=0 处也正确。
            if torch.is_grad_enabled():
                return grad_out * w.conj(), None
            # 同前向：避免 empty_like 在 torch_npu 上触发 internal format 告警。
            gz = torch.empty(w.shape, dtype=w.dtype, device=w.device)
            if n > 0:
                _launch_grad(w, grad_out.contiguous(), gz, n, ctx.block)
            # trailing None: block 是非张量参数
            return gz, None


def complex_exp(
    z: torch.Tensor,
    *,
    block_size: int | None = None,
) -> torch.Tensor:
    """complex64 张量的指数 ``e^z``。

    NPU 上由单个 Triton kernel（Vector Core）完成，替代走 Ai_Cpu 的
    ``aclnnExp``（大张量慢三个数量级）；CPU 或无 triton 环境回退到
    ``torch.exp``（含 autograd）。

    :param z: complex64 张量，任意形状（非连续会先转连续）。
    :param block_size: 每 tile 复数元素数（2 的幂，钳制到 ``[16, 2048]``），
        默认 2048，可用 ``OIL_GAS_OPS_EXP_BLOCK`` 覆盖。
    """
    if z.dtype != torch.complex64 or not (_is_npu_tensor(z) and _has_triton()):
        return torch.exp(z)
    return _ComplexExpFunction.apply(z, _resolve_block(block_size))
