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

"""complex_addcmul 的 pure-PyTorch 组合实现。

计算 ``torch.addcmul(input, tensor1, tensor2, value=value)``：

    out = input + value * tensor1 * tensor2

为什么要单独实现：NPU 上 ``torch.addcmul`` 对 complex64 / complex128 直接报
``aclnnAddcmul failed, error code is 161002``（``DT_COMPLEX64`` 不在 aclnn 的
dtype 支持列表内），而 CPU 正常。实数 dtype 不受影响，调用方应直接用
``torch.addcmul``。

实现要点：

- 复数乘法走 ``complex_mul``：NPU 且 ``OIL_GAS_OPS_COMPLEX_MUL_IMPL`` 为
  ``ascendc`` / ``auto`` 时调用 AscendC ``ComplexMul``，否则用等价 torch 表达式。
  该算子只接受 float32，因此 ``complex128`` 始终走 torch 表达式。
- ``complex_mul`` 要求六路入参同 shape 且连续，先把三路广播到输出 shape。
- ``value`` 可为实数或复数标量（也接受 0 维张量），统一拆成实数对参与运算，
  避免在 NPU 上对复数标量做 ``torch.complex`` 之外的操作。
- 三路输入的 dtype 按 ``torch.promote_types`` 提升，与原生行为一致。

注意本实现是多个 torch 算子的组合，不是单个融合 kernel：它的目标是复数
``addcmul`` 的可用性，而非替代一个理想的三输入融合算子。
"""

from __future__ import annotations

import numbers

import torch
from torch.autograd import Function

from oil_gas_ops_prospect.complex_mul import complex_mul


def _real_imag(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """取复数的实部、虚部两个实数张量。

    先 ``resolve_conj`` 把惰性共轭位物化：NPU 上带 conj 位的张量参与后续运算
    会触发一次整张量拷贝。返回的是 stride 为 2 的视图，连续性由下游
    ``complex_mul`` 内部的 ``contiguous()`` 保证。
    """
    return torch.view_as_real(t.resolve_conj()).unbind(-1)


def _broadcast_parts(
    t: torch.Tensor, shape: torch.Size, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """把复数张量广播（可选类型提升）到 ``shape``，再取实部、虚部。

    返回的实/虚部保证连续，可直接喂给 ``complex_mul``。
    """
    t = t.to(dtype).expand(shape).contiguous()
    return _real_imag(t)


def _scalar_parts(value, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """把标量 ``value`` 拆成实数部分的两个 0 维张量 ``(vr, vi)``。

    ``torch.addcmul`` 的 ``value`` 可以是实数或复数；统一拆成实数对后，
    后续乘加全部落在实数域，避免在 NPU 上对 0 维复数张量做运算
    （``torch.complex`` 的实参不接受复数 dtype）。
    """
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise TypeError(f"value 必须是标量，收到 {tuple(value.shape)} 的张量")
        if value.requires_grad:
            raise TypeError(
                "value 不支持求导：本算子对 value 不返回梯度，"
                "需要对其求导时请把标量乘展开成显式乘法"
            )
        value = value.detach().reshape(()).to(dtype)
    elif not isinstance(value, numbers.Number):
        raise TypeError(f"value 必须是实数或复数标量，收到 {type(value).__name__}")
    else:
        value = torch.as_tensor(value, dtype=dtype)

    if value.is_complex():
        real_imag = torch.view_as_real(value)
        return real_imag[0], real_imag[1]
    return value, torch.zeros_like(value)


class _ComplexAddcmulFunction(Function):
    """``out = input + value * tensor1 * tensor2``，复数专用。"""

    @staticmethod
    def forward(ctx, input, tensor1, tensor2, value):
        ctx.save_for_backward(input, tensor1, tensor2)
        ctx.value = value

        shape = torch.broadcast_shapes(input.shape, tensor1.shape, tensor2.shape)
        dtype = torch.promote_types(
            torch.promote_types(input.dtype, tensor1.dtype), tensor2.dtype
        )

        ir, ii = _broadcast_parts(input, shape, dtype)
        ar, ai = _broadcast_parts(tensor1, shape, dtype)
        br, bi = _broadcast_parts(tensor2, shape, dtype)
        vr, vi = _scalar_parts(value, dtype)

        pr, pi = complex_mul(ar, ai, br, bi)
        wr = ir + vr * pr - vi * pi
        wi = ii + vr * pi + vi * pr
        return torch.complex(wr, wi).view(shape)

    @staticmethod
    def backward(ctx, grad_output):
        input, tensor1, tensor2 = ctx.saved_tensors
        value = ctx.value

        shape = torch.broadcast_shapes(input.shape, tensor1.shape, tensor2.shape)
        dtype = torch.promote_types(
            torch.promote_types(input.dtype, tensor1.dtype), tensor2.dtype
        )
        gr, gi = _broadcast_parts(grad_output, shape, dtype)
        ar, ai = _broadcast_parts(tensor1, shape, dtype)
        br, bi = _broadcast_parts(tensor2, shape, dtype)
        vr, vi = _scalar_parts(value, dtype)

        # PyTorch 复数 autograd 约定：dL/dt = conj(value) * grad_output * conj(t)。
        # value 也是复数时，原生 addcmul 的梯度用的是 conj(value)
        # （已用 CPU 原生算子实测核对）。conj 只作用于 t，所以先算
        # complex_mul(g, conj(t))，再左乘 conj(value)。
        t1r, t1i = complex_mul(gr, gi, br, -bi)
        t2r, t2i = complex_mul(gr, gi, ar, -ai)

        cvr = vr          # conj(value) 的实部
        cvi = -vi         # conj(value) 的虚部
        g1r = cvr * t1r - cvi * t1i
        g1i = cvr * t1i + cvi * t1r
        g2r = cvr * t2r - cvi * t2i
        g2i = cvr * t2i + cvi * t2r

        return (
            _unbroadcast(grad_output, input.shape),
            _unbroadcast(torch.complex(g1r, g1i), tensor1.shape),
            _unbroadcast(torch.complex(g2r, g2i), tensor2.shape),
            None,
        )


def _unbroadcast(grad: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """把广播后的梯度归约回原 shape（与 PyTorch 的 broadcast 反向规则一致）。"""
    if grad.shape == shape:
        return grad
    extra = grad.dim() - len(shape)
    if extra > 0:
        grad = grad.sum(dim=tuple(range(extra)))
    squeezed = [i for i, dim in enumerate(shape) if dim == 1 and grad.shape[i] != 1]
    if squeezed:
        grad = grad.sum(dim=squeezed, keepdim=True)
    return grad.reshape(shape)


def complex_addcmul(
    input: torch.Tensor,
    tensor1: torch.Tensor,
    tensor2: torch.Tensor,
    value=1.0,
) -> torch.Tensor:
    """复数三输入逐元素算子：``input + value * tensor1 * tensor2``。

    语义与 ``torch.addcmul`` 一致，用于绕过 NPU 上该算子不支持复数 dtype 的限制。

    Args:
        input: 被加项，复数张量（``complex64`` / ``complex128``）。
        tensor1: 乘数一，复数张量，支持广播。
        tensor2: 乘数二，复数张量，支持广播。
        value: 乘数标量，实数或复数，默认 ``1.0``。

    Returns:
        三个输入广播后的复数张量，dtype 为三者按 ``torch.promote_types`` 提升的结果。

    Raises:
        TypeError: 任一输入不是复数 dtype，或 ``value`` 不是标量。

    只处理复数：实数 dtype 请直接用 ``torch.addcmul``，本算子不做转发。
    """
    for name, t in (("input", input), ("tensor1", tensor1), ("tensor2", tensor2)):
        if not t.is_complex():
            raise TypeError(
                f"complex_addcmul 只接受复数张量，{name} 的 dtype 为 {t.dtype}；"
                "实数请直接用 torch.addcmul"
            )
    return _ComplexAddcmulFunction.apply(input, tensor1, tensor2, value)
