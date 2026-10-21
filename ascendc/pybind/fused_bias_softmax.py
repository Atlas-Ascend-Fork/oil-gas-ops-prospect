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

"""FusedBiasSoftmax / FusedSoftmaxGrad：openfold 注意力前反向融合算子。

::

    forward:  y = softmax(x + bias1(广播/全尺寸) + bias2, -1)
    backward: g = go @ V^T; probs ← probs ⊙ (g − rowsum(probs⊙g))
    attention_core: 上述两算子按 openfold attention_core 语义的组合封装

仅当调用者输入连续且成功进入 NPU AscendC 融合路径时，前向原地写回调用者的 ``x``，
反向原地覆写调用者的 ``probs``。非连续输入只会覆写 ``contiguous()`` 副本，参考回退非原地。

环境变量：

- ``OIL_GAS_OPS_FUSED_BIAS_SOFTMAX_IMPL=torch|ascendc|auto``（默认 torch）
- ``OIL_GAS_OPS_REQUIRE_ASCENDC_FUSED_BIAS_SOFTMAX=1`` 强制要求自定义算子
"""

from __future__ import annotations

import importlib
import os
from functools import lru_cache

import torch


@lru_cache(maxsize=1)
def _load_extension():
    return importlib.import_module("fused_bias_softmax_npu")


def _require_custom() -> bool:
    return os.environ.get("OIL_GAS_OPS_REQUIRE_ASCENDC_FUSED_BIAS_SOFTMAX", "0") == "1"


def _impl() -> str:
    return os.environ.get("OIL_GAS_OPS_FUSED_BIAS_SOFTMAX_IMPL", "torch").strip().lower()


def _is_npu(x: torch.Tensor) -> bool:
    return x.device.type == "npu"


def _use_ascendc(x: torch.Tensor) -> bool:
    """由环境开关与张量设备决定是否走融合算子路径。"""
    return _impl() in ("ascendc", "auto") and _is_npu(x)


# ── 纯 PyTorch 参考（CPU fallback / 对拍基准）─────────────────────────

def torch_fused_bias_softmax(x, bias1=None, bias2=None):
    """softmax(x + bias1 + bias2, -1)，非原地，fp32 域计算。"""
    logits = x.float().clone()
    if bias1 is not None:
        logits = logits + bias1.float()
    if bias2 is not None:
        logits = logits + bias2.float()
    return torch.softmax(logits, -1).to(x.dtype)


def torch_fused_softmax_grad(probs, grad_output, values):
    """g = go @ V^T 的 softmax JVP，非原地，fp32 域计算。"""
    pf = probs.float()
    g = torch.matmul(grad_output.float(), values.float().transpose(-1, -2))
    return (pf * (g - (g * pf).sum(-1, keepdim=True))).to(probs.dtype)


# ── AscendC 路径 ──────────────────────────────────────────────────────

def _ascendc_fused_bias_softmax(x, bias1, bias2):
    ext = _load_extension()
    return ext.fused_bias_softmax_forward(
        x.contiguous(),
        None if bias1 is None else bias1.contiguous(),
        None if bias2 is None else bias2.contiguous(),
    )


def _ascendc_fused_softmax_grad(probs, grad_output, values):
    ext = _load_extension()
    return ext.fused_softmax_grad_backward(
        probs.contiguous(), grad_output.contiguous(), values.contiguous()
    )


def fused_bias_softmax(x: torch.Tensor, bias1: torch.Tensor | None = None,
    bias2: torch.Tensor | None = None) -> torch.Tensor:
    """前向融合：softmax(x + bias1 + bias2, -1)。

    仅当调用者的 x 连续且成功进入 NPU AscendC 融合路径时，才会原地写回调用者的 x；
    非连续输入只会写回 contiguous() 副本，融合失败后的参考回退非原地。
    bias1: [B,H,1,S] 广播或 [B,H,Q,S] 全尺寸；bias2: [B,H,Q,S]；均可为 None。
    """
    impl = _impl()
    use_ascendc = impl in ("ascendc", "auto") and _is_npu(x)
    if impl == "ascendc" and not _is_npu(x) and _require_custom():
        raise RuntimeError("OIL_GAS_OPS_FUSED_BIAS_SOFTMAX_IMPL=ascendc requires NPU tensors")
    if use_ascendc:
        try:
            return _ascendc_fused_bias_softmax(x.contiguous(), bias1, bias2)
        except Exception:
            if _require_custom():
                raise
    return torch_fused_bias_softmax(x, bias1, bias2)


def fused_softmax_grad(probs: torch.Tensor, grad_output: torch.Tensor,
    values: torch.Tensor) -> torch.Tensor:
    """反向融合：g = go@V^T 融合 softmax JVP。

    仅当调用者的 probs 连续且成功进入 NPU AscendC 融合路径时，才会原地覆写调用者的 probs；
    非连续输入只会覆写 contiguous() 副本，融合失败后的参考回退非原地。
    probs: [B,H,Q,S]（Q≠S 合法）；grad_output: [B,H,Q,D]；values: [B,H,S,D]。
    """
    impl = _impl()
    use_ascendc = impl in ("ascendc", "auto") and _is_npu(probs)
    if impl == "ascendc" and not _is_npu(probs) and _require_custom():
        raise RuntimeError("OIL_GAS_OPS_FUSED_BIAS_SOFTMAX_IMPL=ascendc requires NPU tensors")
    if use_ascendc:
        try:
            return _ascendc_fused_softmax_grad(probs.contiguous(), grad_output.contiguous(),
                values.contiguous())
        except Exception:
            if _require_custom():
                raise
    return torch_fused_softmax_grad(probs, grad_output, values)


# ── openfold attention_core 语义的组合封装（autograd）────────────────

class _FusedAttentionCoreFn(torch.autograd.Function):
    """openfold AttentionCoreFunction 等价语义（bias 梯度含在 backward 内）。

    forward:  probs = softmax(qk^T + bias1 + bias2, -1); o = probs @ v
    backward: grad_v 先于 probs 覆写计算（与原版 CUDA 时序一致）；
              AscendC 融合分支原地覆写传入扩展的连续内部 probs，再求 bias/q/k 梯度；
              参考分支非原地。
    """

    @staticmethod
    def forward(ctx, q, k, v, bias1, bias2):
        use_ascendc = _use_ascendc(q)
        ctx.use_ascendc = use_ascendc
        q = q.contiguous()
        k = k.contiguous()
        logits = torch.matmul(q, k.transpose(-1, -2))
        if use_ascendc:
            logits = _ascendc_fused_bias_softmax(logits, bias1, bias2)
        else:
            logits = torch_fused_bias_softmax(logits, bias1, bias2)
        ctx.save_for_backward(q, k, v, logits)
        ctx.bias1_shape = bias1.shape if bias1 is not None else None
        ctx.bias2_shape = bias2.shape if bias2 is not None else None
        return torch.matmul(logits, v)

    @staticmethod
    def backward(ctx, grad_output):
        q, k, v, probs = ctx.saved_tensors
        grad_q = grad_k = grad_v = grad_bias1 = grad_bias2 = None

        grad_v = torch.matmul(probs.transpose(-1, -2), grad_output)
        go = grad_output.contiguous()
        if ctx.use_ascendc:
            g_logits = _ascendc_fused_softmax_grad(probs, go, v)
        else:
            g_logits = torch_fused_softmax_grad(probs, go, v)

        if ctx.bias1_shape is not None:
            dims = tuple(i for i, d in enumerate(ctx.bias1_shape) if d == 1)
            grad_bias1 = g_logits.clone() if not dims else torch.sum(g_logits, dim=dims, keepdim=True)
        if ctx.bias2_shape is not None:
            dims = tuple(i for i, d in enumerate(ctx.bias2_shape) if d == 1)
            grad_bias2 = g_logits.clone() if not dims else torch.sum(g_logits, dim=dims, keepdim=True)

        grad_q = torch.matmul(g_logits, k)
        grad_k = torch.matmul(q.transpose(-1, -2), g_logits).transpose(-1, -2)
        return grad_q, grad_k, grad_v, grad_bias1, grad_bias2


def attention_core(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    bias1: torch.Tensor | None = None, bias2: torch.Tensor | None = None) -> torch.Tensor:
    """openfold attention_core 等价调用：前反向均路由到融合算子（NPU + ascendc 模式）。

    用法与 openfold `AttentionCoreFunction.apply` 一致：
        o = attention_core(q, k, v, mask_bias, triangle_bias)
    """
    if _impl() == "ascendc" and not _is_npu(q) and _require_custom():
        raise RuntimeError("OIL_GAS_OPS_FUSED_BIAS_SOFTMAX_IMPL=ascendc requires NPU tensors")
    return _FusedAttentionCoreFn.apply(q, k, v, bias1, bias2)
