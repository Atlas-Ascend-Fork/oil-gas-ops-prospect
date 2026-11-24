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

"""complex_index_select 的 pure-PyTorch 组合实现。

计算 ``torch.index_select(input, dim, index)``，输入为复数张量：

    out[..., k, ...] = input[..., index[k], ...]（沿 ``dim`` 取行）

为什么要单独实现：NPU 上复数 ``index_select`` 的原生路径有两处缺陷（910B2 /
CANN 9.0.0 / torch_npu 2.7.1 实测）：

- **反向直接报错**：autograd 反向落在 ``aclnnIndexAdd``，其 dtype 支持列表不含
  ``DT_COMPLEX64`` / ``DT_COMPLEX128``，训练路径 ``backward()`` 抛
  ``RuntimeError``（error code 161002）。
- **前向派发到 AiCPU**：complex64 走 ``GatherV2``（AI_CPU 标量核），float32 走
  ``GatherV3``（AI_VECTOR_CORE）。同形状 kernel 耗时相差约两个数量级（中块形状
  9.66ms vs 47.3µs），大形状经公开 API 端到端慢 22~144 倍（随形状与派发开销
  占比变化，实测口径见算子 README 性能表）。

实现要点：

- **flat 路径**（输入连续且 ``dim < input.dim() - 1``）：complex64 在内存中是
  两个相邻 float32，``view(torch.float32)`` 后行宽翻倍、行数不变，沿 ``dim``
  gather 的字节访问模式与「复数感知 kernel」完全相同，因此能拿到与 float32
  基线一致的带宽；反向在 float 视图上做 ``index_add_``。``dim`` 为末轴时该
  视图会拆散实虚对，不能用。
- **real/imag 路径**（末轴 ``dim`` 或非连续输入）：实部、虚部各 gather 一次再
  ``torch.complex`` 重组。stride-2 访问比 flat 路径慢，但仍远快于原生 AiCPU
  路径；1-D 张量（``dim == 0`` 即末轴）也走这条路。
- 两条路径都是纯 torch 算子组合，CPU / NPU 行为一致；反向是 ``zeros`` +
  ``index_add_``（与原生 ``index_select`` 的反向同构）。二阶导不可用，与原生
  一致：反向公式只消费 ``grad_output``，不含可微的保存张量，``create_graph``
  无法在其上继续建图。
- 索引值越界 / 为负的行为跟随底层后端，与原生 ``torch.index_select`` 一致。

注意本实现的目标是复数 ``index_select`` 的可用性（反向从报错变为可用）与前向
性能，不是替代一个理想的融合 kernel；实数 dtype 请直接用 ``torch.index_select``，
``complex32`` 等实验性复数 dtype 不在支持范围内（仅 ``complex64`` / ``complex128``）。
"""

from __future__ import annotations

import torch
from torch.autograd import Function

__all__ = ["complex_index_select"]


def _real_dtype(dtype: torch.dtype) -> torch.dtype:
    return torch.float32 if dtype == torch.complex64 else torch.float64


def _flat_ok(z: torch.Tensor, dim: int) -> bool:
    """flat 路径是否适用：连续输入且 gather 轴不是末轴。

    末轴 gather 会拆散 ``view(float)`` 里的实虚对（包括 1-D 张量的 ``dim=0``），
    只能走 real/imag 路径。带 conj 位的张量 ``resolve_conj`` 物化后仍是原布局，
    连续性判断在物化前后一致。
    """
    return z.is_contiguous() and dim < z.dim() - 1


class _ComplexIndexSelectFunction(Function):
    """复数 ``index_select`` 的自定义前向 / 反向。"""

    @staticmethod
    def forward(ctx, z, dim, index):
        ctx.dim = dim
        ctx.src_shape = tuple(z.shape)
        ctx.src_dtype = z.dtype
        ctx.flat = _flat_ok(z, dim)
        ctx.save_for_backward(index)

        z = z.resolve_conj()
        if ctx.flat:
            out = torch.index_select(z.view(_real_dtype(z.dtype)), dim, index)
            return out.view(z.dtype)
        return torch.complex(
            z.real.index_select(dim, index), z.imag.index_select(dim, index)
        )

    @staticmethod
    def backward(ctx, grad_output):
        (index,) = ctx.saved_tensors
        dim = ctx.dim
        real_dtype = _real_dtype(ctx.src_dtype)
        kw = {"dtype": real_dtype, "device": grad_output.device}

        if ctx.flat:
            # 反向与 flat 前向同构：float 视图上 index_add_。zeros 直接按
            # float shape（末维翻倍）分配：NPU 上 complex dtype 的 fill 会落
            # 到 AiCPU 慢路径（910B2 实测 512MB 要 39ms，float fill 0.4ms）。
            # 下游 .conj() 回流的梯度带 conj 位：contiguous() 不消该位，
            # .view 会拒绝 conjugate view，先 resolve_conj 物化（无 conj 位
            # 时为 no-op）；real/imag 路径经 .real/.imag 自动物化，不受影响。
            gf = grad_output.resolve_conj().contiguous().view(real_dtype)
            f32_shape = ctx.src_shape[:-1] + (ctx.src_shape[-1] * 2,)
            gz = torch.zeros(f32_shape, dtype=real_dtype,
                             device=grad_output.device)
            gz.index_add_(dim, index, gf)
            return gz.view(ctx.src_dtype), None, None

        # real/imag 路径：实部、虚部各自 scatter，再重组复数。
        gz_r = torch.zeros(ctx.src_shape, **kw)
        gz_r.index_add_(dim, index, grad_output.real)
        gz_i = torch.zeros(ctx.src_shape, **kw)
        gz_i.index_add_(dim, index, grad_output.imag)
        return torch.complex(gz_r, gz_i), None, None


def complex_index_select(
    input: torch.Tensor, dim: int, index: torch.Tensor
) -> torch.Tensor:
    """复数版 ``torch.index_select``：沿 ``dim`` 按 ``index`` 抽取行。

    语义与 ``torch.index_select`` 一致，用于补齐 NPU 上复数路径的两处缺陷：
    原生反向（``aclnnIndexAdd``）不支持复数 dtype 直接报错；原生 complex64 前向
    派发到 AiCPU 标量核，大形状比 float32 路径慢两个数量级。

    Args:
        input: 复数张量（``complex64`` / ``complex128``），任意 rank、任意布局。
        dim: 抽取的轴，支持负数（与 torch 归一化规则一致）。
        index: 一维索引张量，``int32`` / ``int64``，可重复、可为空。

    Returns:
        与 ``torch.index_select(input, dim, index)`` 逐位一致的复数张量，
        dtype 与 ``input`` 相同。

    Raises:
        TypeError: ``input`` 不是复数张量（实数请直接用 ``torch.index_select``），
            或 ``input`` 的 dtype 不是 ``complex64`` / ``complex128``（如实验性
            的 ``complex32``），或 ``index`` 不是一维 int32 / int64 张量。
        IndexError: ``dim`` 越界。
        ValueError: 必须是一维张量，index.dim()不为1时抛该异常。

    实数 dtype 不做转发；索引值越界 / 为负的行为跟随底层后端，与原生一致。
    """
    if not isinstance(input, torch.Tensor) or not input.is_complex():
        got = getattr(input, "dtype", type(input).__name__)
        raise TypeError(
            f"complex_index_select 只接受复数张量，收到 dtype={got}；"
            "实数请直接用 torch.index_select"
        )
    if input.dtype not in (torch.complex64, torch.complex128):
        # complex32（实验性半精度复数）的 float 视图宽度换算与 c64/c128
        # 不同，反向会以无关的 shape 错误崩溃；显式拒绝优于误导性报错。
        raise TypeError(
            f"complex_index_select 只支持 complex64 / complex128，"
            f"收到 {input.dtype}"
        )
    if not isinstance(index, torch.Tensor):
        raise TypeError(
            f"index 必须是张量，收到 {type(index).__name__}"
        )
    if index.dim() != 1:
        raise ValueError(
            f"index 必须是一维张量，收到 {tuple(index.shape)}"
        )
    if index.dtype not in (torch.int32, torch.int64):
        raise TypeError(
            f"index 的 dtype 必须是 int32 或 int64，收到 {index.dtype}"
        )
    if not isinstance(dim, int) or isinstance(dim, bool):
        raise TypeError(f"dim 必须是 int，收到 {type(dim).__name__}")

    rank = input.dim()
    if rank == 0 or not -rank <= dim < rank:
        raise IndexError(
            f"Dimension out of range（期望 [-{rank}, {rank - 1}]，收到 {dim}）"
        )
    if dim < 0:
        dim += rank

    return _ComplexIndexSelectFunction.apply(input, dim, index)
