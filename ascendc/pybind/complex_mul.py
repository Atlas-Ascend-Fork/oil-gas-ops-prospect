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

"""ComplexMul / ComplexMulGrad：复数逐点乘融合算子（含反向）。

::

    wr = ur * vr - ui * vi
    wi = ur * vi + ui * vr

环境变量：
- ``OIL_GAS_OPS_COMPLEX_MUL_IMPL=torch|ascendc|auto``（默认 torch）
- ``OIL_GAS_OPS_REQUIRE_ASCENDC_COMPLEX_MUL=1`` 强制要求自定义算子
"""

from __future__ import annotations

import importlib
import os
from functools import lru_cache

import torch


@lru_cache(maxsize=1)
def _load_extension():
    return importlib.import_module("complex_mul_npu")


def _require_custom() -> bool:
    return os.environ.get("OIL_GAS_OPS_REQUIRE_ASCENDC_COMPLEX_MUL", "0") == "1"


def _impl() -> str:
    return os.environ.get("OIL_GAS_OPS_COMPLEX_MUL_IMPL", "torch").strip().lower()


def _is_npu(x: torch.Tensor) -> bool:
    return x.device.type == "npu"


def torch_complex_mul(
    ur: torch.Tensor, ui: torch.Tensor, vr: torch.Tensor, vi: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    wr = ur * vr - ui * vi
    wi = ur * vi + ui * vr
    return wr, wi


def torch_complex_mul_grad(
    gwr: torch.Tensor,
    gwi: torch.Tensor,
    ur: torch.Tensor,
    ui: torch.Tensor,
    vr: torch.Tensor,
    vi: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    gur = gwr * vr + gwi * vi
    gui = -gwr * vi + gwi * vr
    gvr = gwr * ur + gwi * ui
    gvi = -gwr * ui + gwi * ur
    return gur, gui, gvr, gvi


def _kernel_in(t: torch.Tensor) -> torch.Tensor:
    """转成 kernel 要求的连续张量。

    ``contiguous()`` 对已连续张量零拷贝，对非连续 / stride 0 张量会物化。
    不要在这里追加 ``clone()``：多出的整张量拷贝在 NPU 上退化为 AI_CPU 算子。
    """
    return t.detach().contiguous()


def _split_outputs(packed) -> tuple[torch.Tensor, ...]:
    """把 C++ 侧返回的 packed 输出切成相互独立的张量。

    kernel 受 TBE 限制，把 2 路（反向 4 路）输出写进同一块缓冲，返回的 view
    共享 storage。``autograd.Function`` 返回多个共享 storage 的 view 时，PyTorch
    会拒绝对其做原地写，因此**前向**返回前必须 clone。

    ``contiguous()`` 不能替代：packed 的每一路本来就连续，它是 no-op，别名仍在。
    """
    return tuple(t.clone() for t in packed)


def _ascendc_forward(ur, ui, vr, vi):
    ext = _load_extension()
    packed = ext.complex_mul_forward(
        _kernel_in(ur),
        _kernel_in(ui),
        _kernel_in(vr),
        _kernel_in(vi),
    )
    return _split_outputs(packed)


def _ascendc_backward(gwr, gwi, ur, ui, vr, vi):
    ext = _load_extension()
    # gwr / gwi 可能互为同一张量的 view（如 ``(wr + wi).sum()``，其中一个带
    # stride 0 广播布局）。``_kernel_in`` 的 contiguous() 已把它们物化成独立
    # 缓冲，不需要再 clone。
    gur, gui, gvr, gvi = ext.complex_mul_backward(
        _kernel_in(gwr),
        _kernel_in(gwi),
        _kernel_in(ur),
        _kernel_in(ui),
        _kernel_in(vr),
        _kernel_in(vi),
    )
    # 4 路梯度共享一块缓冲是允许的：autograd 直接累加 backward 返回值，不走
    # 「用户可见多 view」的原地写检查。
    return gur, gui, gvr, gvi


class _ComplexMulFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ur, ui, vr, vi, use_ascendc: bool):
        ctx.use_ascendc = bool(use_ascendc)
        ctx.save_for_backward(ur, ui, vr, vi)
        if use_ascendc:
            try:
                return _ascendc_forward(ur, ui, vr, vi)
            except Exception:
                if _require_custom():
                    raise
                return torch_complex_mul(ur, ui, vr, vi)
        return torch_complex_mul(ur, ui, vr, vi)

    @staticmethod
    def backward(ctx, gwr, gwi):
        ur, ui, vr, vi = ctx.saved_tensors
        if ctx.use_ascendc:
            try:
                gur, gui, gvr, gvi = _ascendc_backward(gwr, gwi, ur, ui, vr, vi)
                return gur, gui, gvr, gvi, None
            except Exception:
                if _require_custom():
                    raise
        gur, gui, gvr, gvi = torch_complex_mul_grad(gwr, gwi, ur, ui, vr, vi)
        return gur, gui, gvr, gvi, None


def complex_mul(
    ur: torch.Tensor,
    ui: torch.Tensor,
    vr: torch.Tensor,
    vi: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """复数逐点乘；在 NPU 且 ``OIL_GAS_OPS_COMPLEX_MUL_IMPL=ascendc`` 时走融合算子。"""
    impl = _impl()
    use_ascendc = impl in ("ascendc", "auto") and _is_npu(ur)
    if impl == "ascendc" and not _is_npu(ur) and _require_custom():
        raise RuntimeError("OIL_GAS_OPS_COMPLEX_MUL_IMPL=ascendc requires NPU tensors")
    if use_ascendc or impl == "ascendc":
        return _ComplexMulFn.apply(ur, ui, vr, vi, True)
    return _ComplexMulFn.apply(ur, ui, vr, vi, False)
