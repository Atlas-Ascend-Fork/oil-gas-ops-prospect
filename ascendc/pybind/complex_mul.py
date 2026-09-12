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


def _ascendc_forward(ur, ui, vr, vi):
    ext = _load_extension()
    wr, wi = ext.complex_mul_forward(
        ur.contiguous(), ui.contiguous(), vr.contiguous(), vi.contiguous()
    )
    return wr, wi


def _ascendc_backward(gwr, gwi, ur, ui, vr, vi):
    ext = _load_extension()
    return ext.complex_mul_backward(
        gwr.contiguous(),
        gwi.contiguous(),
        ur.contiguous(),
        ui.contiguous(),
        vr.contiguous(),
        vi.contiguous(),
    )


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
