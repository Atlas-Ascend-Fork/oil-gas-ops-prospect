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

"""CustomGRUV2：自研单层 GRU 融合算子（含 torch 回退与 autograd）。

::

    r = sigmoid(W_ir x + b_ir + W_hr h + b_hr)
    z = sigmoid(W_iz x + b_iz + W_hz h + b_hz)
    n = tanh(W_in x + b_in + r * (W_hn h + b_hn))
    h' = (1 - z) * n + z * h

环境变量：
- ``OIL_GAS_OPS_REQUIRE_ASCENDC_CUSTOM_GRU_V2=1`` 强制要求自定义算子（禁止静默回退）
"""

from __future__ import annotations

import importlib
import os
from functools import lru_cache

import torch
import torch.nn as nn


@lru_cache(maxsize=1)
def _load_extension():
    return importlib.import_module("custom_gru_v2_npu")


def _require_custom() -> bool:
    return os.environ.get("OIL_GAS_OPS_REQUIRE_ASCENDC_CUSTOM_GRU_V2", "0") == "1"


def _is_npu_tensor(x: torch.Tensor) -> bool:
    return x.device.type == "npu"


def _torch_gru_forward(
    x: torch.Tensor,
    weight_ih: torch.Tensor,
    weight_hh: torch.Tensor,
    bias_ih: torch.Tensor,
    bias_hh: torch.Tensor,
    hidden_size: int,
    bidirectional: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure-Python GRU reference using GRUCell loop (CPU/NPU fallback).

    Single-layer only.  For bidirectional=True the backward direction
    processes the time axis in reverse, matching the kernel.
    """
    if x.dim() == 2:
        x = x.unsqueeze(1)
        was_2d = True
    else:
        was_2d = False
    B, T, F_in = x.shape
    num_dirs = 2 if bidirectional else 1

    if weight_ih.dim() == 3:
        wi_list = [weight_ih[d] for d in range(num_dirs)]
        wh_list = [weight_hh[d] for d in range(num_dirs)]
        bi_list = [bias_ih[d] if bias_ih is not None else None for d in range(num_dirs)]
        bh_list = [bias_hh[d] if bias_hh is not None else None for d in range(num_dirs)]
    else:
        if bidirectional and weight_ih.shape[0] == num_dirs * 3 * hidden_size:
            half = 3 * hidden_size
            wi_list = [weight_ih[:half], weight_ih[half:]]
            wh_list = [weight_hh[:half], weight_hh[half:]]
            bi_list = ([bias_ih[:half], bias_ih[half:]] if bias_ih is not None
                       else [None, None])
            bh_list = ([bias_hh[:half], bias_hh[half:]] if bias_hh is not None
                       else [None, None])
        else:
            wi_list = [weight_ih] * num_dirs
            wh_list = [weight_hh] * num_dirs
            bi_list = [bias_ih] * num_dirs
            bh_list = [bias_hh] * num_dirs

    y = torch.empty(num_dirs, B, T, hidden_size, dtype=x.dtype, device=x.device)
    output_h = torch.empty(num_dirs, B, hidden_size, dtype=x.dtype, device=x.device)

    for d in range(num_dirs):
        h = torch.zeros(B, hidden_size, dtype=x.dtype, device=x.device)
        wi, wh = wi_list[d], wh_list[d]
        bi = bi_list[d] if bi_list[d] is not None else torch.zeros(3 * hidden_size, dtype=x.dtype, device=x.device)
        bh = bh_list[d] if bh_list[d] is not None else torch.zeros(3 * hidden_size, dtype=x.dtype, device=x.device)

        if d == 0:
            indices = range(T)
        else:
            indices = range(T - 1, -1, -1)

        for t in indices:
            xt = x[:, t, :]
            gi = torch.nn.functional.linear(xt, wi, bi)
            gh = torch.nn.functional.linear(h, wh, bh)
            r = torch.sigmoid(gi[:, :hidden_size] + gh[:, :hidden_size])
            z = torch.sigmoid(gi[:, hidden_size:2*hidden_size] + gh[:, hidden_size:2*hidden_size])
            n = torch.tanh(gi[:, 2*hidden_size:] + r * gh[:, 2*hidden_size:])
            h = (1 - z) * n + z * h
            y[d, :, t, :] = h

        output_h[d] = h

    if was_2d:
        y = y.squeeze(2)
        output_h = output_h.squeeze(1)

    return y, output_h


class _CustomGRUV2Function(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight_ih: torch.Tensor,
        weight_hh: torch.Tensor,
        bias_ih: torch.Tensor,
        bias_hh: torch.Tensor,
        hidden_size: int,
        bidirectional: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ctx.hidden_size = hidden_size
        ctx.bidirectional = bidirectional
        ctx.save_for_backward(x, weight_ih, weight_hh, bias_ih, bias_hh)

        ext = _load_extension()
        return ext.custom_gru_v2(x, weight_ih, weight_hh, bias_ih, bias_hh,
                                  hidden_size, bidirectional)

    @staticmethod
    def backward(ctx, grad_y: torch.Tensor, grad_h: torch.Tensor):
        x, weight_ih, weight_hh, bias_ih, bias_hh = ctx.saved_tensors
        with torch.enable_grad():
            x_req = x.detach().requires_grad_(True)
            wi_req = weight_ih.detach().requires_grad_(True)
            wh_req = weight_hh.detach().requires_grad_(True)
            bi_req = bias_ih.detach().requires_grad_(True) if bias_ih is not None and bias_ih.requires_grad else bias_ih
            bh_req = bias_hh.detach().requires_grad_(True) if bias_hh is not None and bias_hh.requires_grad else bias_hh
            y, h = _torch_gru_forward(
                x_req, wi_req, wh_req, bi_req, bh_req,
                ctx.hidden_size, ctx.bidirectional,
            )
        grad_tensors = [x_req, wi_req, wh_req]
        if bi_req is not None and bi_req.requires_grad:
            grad_tensors.append(bi_req)
        else:
            grad_tensors.append(None)
        if bh_req is not None and bh_req.requires_grad:
            grad_tensors.append(bh_req)
        else:
            grad_tensors.append(None)
        valid_tensors = [t for t in grad_tensors if t is not None]
        grads = torch.autograd.grad(
            (y, h), valid_tensors,
            (grad_y, grad_h), allow_unused=True,
        )
        grad_map = {}
        gi = 0
        for name, t in zip(["x", "wi", "wh", "bi", "bh"], grad_tensors):
            if t is not None:
                grad_map[name] = grads[gi]
                gi += 1
            else:
                grad_map[name] = None
        return (grad_map["x"], grad_map["wi"], grad_map["wh"],
                grad_map["bi"], grad_map["bh"], None, None)


def custom_gru_v2(
    x: torch.Tensor,
    weight_ih: torch.Tensor,
    weight_hh: torch.Tensor,
    bias_ih: torch.Tensor | None,
    bias_hh: torch.Tensor | None,
    hidden_size: int,
    bidirectional: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Custom GRU v2 fused kernel (single-layer).

    Args:
        x: Input tensor [B, T, F_in]
        weight_ih: Input-hidden weights [3*H, F_in] (or [num_dirs, 3*H, F_in])
        weight_hh: Hidden-hidden weights [3*H, H] (or [num_dirs, 3*H, H])
        bias_ih: Input-hidden bias [3*H] or None
        bias_hh: Hidden-hidden bias [3*H] or None
        hidden_size: Hidden dimension H
        bidirectional: Bidirectional flag (default False)

    Returns:
        y: Output [num_dirs, B, T, H]
        output_h: Final hidden state [num_dirs, B, H]
    """
    if bias_ih is None:
        if weight_ih.dim() == 3:
            bias_ih = torch.zeros(weight_ih.shape[0], 3 * hidden_size, dtype=x.dtype, device=x.device)
        else:
            bias_ih = torch.zeros(weight_ih.shape[0], dtype=x.dtype, device=x.device)
    if bias_hh is None:
        if weight_hh.dim() == 3:
            bias_hh = torch.zeros(weight_hh.shape[0], 3 * hidden_size, dtype=x.dtype, device=x.device)
        else:
            bias_hh = torch.zeros(weight_hh.shape[0], dtype=x.dtype, device=x.device)

    if not _is_npu_tensor(x):
        if _require_custom():
            raise RuntimeError("OIL_GAS_OPS_REQUIRE_ASCENDC_CUSTOM_GRU_V2=1 but input is not an NPU tensor.")
        return _torch_gru_forward(x, weight_ih, weight_hh, bias_ih, bias_hh,
                                   hidden_size, bidirectional)

    try:
        _load_extension()
    except ImportError as exc:
        if _require_custom():
            raise RuntimeError("custom_gru_v2_npu extension is required but could not be imported.") from exc
        return _torch_gru_forward(x, weight_ih, weight_hh, bias_ih, bias_hh,
                                   hidden_size, bidirectional)

    return _CustomGRUV2Function.apply(x, weight_ih, weight_hh, bias_ih, bias_hh,
                                       hidden_size, bidirectional)


class CustomGRU(nn.Module):
    """nn.Module wrapper for CustomGRU v2 fused kernel (single-layer)."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bidirectional: bool = False,
        bias: bool = True,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.num_dirs = 2 if bidirectional else 1

        self.weight_ih = nn.Parameter(torch.empty(3 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.empty(3 * hidden_size, hidden_size))
        if bias:
            self.bias_ih = nn.Parameter(torch.zeros(3 * hidden_size))
            self.bias_hh = nn.Parameter(torch.zeros(3 * hidden_size))
        else:
            self.register_parameter("bias_ih", None)
            self.register_parameter("bias_hh", None)

        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(p)
            elif "bias" in name:
                nn.init.zeros_(p)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bi = self.bias_ih if self.bias_ih is not None else torch.zeros(3 * self.hidden_size, device=x.device, dtype=x.dtype)
        bh = self.bias_hh if self.bias_hh is not None else torch.zeros(3 * self.hidden_size, device=x.device, dtype=x.dtype)
        return custom_gru_v2(x, self.weight_ih, self.weight_hh, bi, bh,
                             self.hidden_size, self.bidirectional)
