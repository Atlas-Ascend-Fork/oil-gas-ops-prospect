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

"""Conv3DGelu：Conv3d + tanh-GELU 的 cube+vector 融合算子。

前向只写出 GELU 结果。GELU 对齐 NPU 上 `nn.GELU` / `F.gelu` 实际执行的 tanh 近似
（也就是 CPU `F.gelu(approximate="tanh")`），不是 CPU `nn.GELU()` 的 erf 默认公式。
反向不保存卷积与 GELU 之间的预激活。输出不超过 256MB 时前向和反向一次算完；
更大的体积沿 D 按大约 48MB 一块切，避免私有格式缓冲和重算激活按整幅分配。

::

    y = gelu(conv3d(x, weight, bias), approximate="tanh")

环境变量：
- ``OIL_GAS_OPS_CONV3D_GELU_IMPL=torch|ascendc|auto``（默认 auto：NPU 上走融合算子）
- ``OIL_GAS_OPS_REQUIRE_ASCENDC_CONV3D_GELU=1`` 强制要求自定义算子
"""

from __future__ import annotations

import importlib
import os
import sys
import warnings
from functools import lru_cache
from typing import Sequence

# CANN 9 的 aclnn 执行器缓存会在 shape 切换后复用错误的输出地址。
# 必须在首次 import torch / 初始化 ACL 之前设为 0；setdefault 补不了已经初始化的进程。
if "torch" in sys.modules and os.environ.get("ACLNN_CACHE_LIMIT") != "0":
    warnings.warn(
        "ACLNN_CACHE_LIMIT 必须在 import torch 之前设为 0。"
        "当前进程已经导入 torch，conv3d_gelu 无法关闭 CANN 9 的执行器缓存，"
        "换 shape 后可能得到错误结果。请在启动 Python 前执行 ACLNN_CACHE_LIMIT=0。",
        RuntimeWarning,
        stacklevel=2,
    )
os.environ.setdefault("ACLNN_CACHE_LIMIT", "0")

import torch
import torch.nn.functional as F


def _triple(value, name: str) -> tuple[int, int, int]:
    if isinstance(value, bool) or not isinstance(value, (int, Sequence)):
        raise ValueError(f"{name} must be an int or a sequence of length 1 or 3, got {value!r}")
    if isinstance(value, int):
        return (value, value, value)
    seq = tuple(int(v) for v in value)
    if len(seq) == 1:
        return (seq[0], seq[0], seq[0])
    if len(seq) == 3:
        return seq  # type: ignore[return-value]
    raise ValueError(f"{name} must be an int or a sequence of length 1 or 3, got {value!r}")


def _pad6_from_conv3d(padding: Sequence[int]) -> tuple[int, int, int, int, int, int]:
    pd, ph, pw = _triple(padding, "padding")
    return (pw, pw, ph, ph, pd, pd)


@lru_cache(maxsize=1)
def _load_extension():
    return importlib.import_module("conv3d_gelu_npu")


# ACLNN_CACHE_LIMIT=0 只关掉 aclnn 的 executor / 二进制缓存，避免换 shape 后复用旧计划。
# 它不管「某个 shape 在本进程里第一次真正 launch」的数值。CANN 9.0.0 / 910B2 上测到：
# 新 shape 的前三次前向或反向和 CPU 标杆对不上，第四次才稳定。这不是缓存命中，
# 丢弃次数是在该版本上试出来的，换 CANN 后可能不够，也可能变成多余。
# 每个 key 只做一次这种丢弃，之后每次调用仍是单次 launch。
_WARMED_KEYS: set[tuple] = set()
# 切块反向还有第二层：每个窗口各自丢掉前三次之后，进程里第一次把这些窗口累加起来仍会偏。
# 整段再丢一次，用户看到的第一次切块反向才和标杆一致。同样只在 CANN 9.0.0 上核对过。
_CHUNKED_BWD_PRIMED = False


def _launch_key(x, weight, stride, padding, dilation, groups, bias, enable_hf32) -> tuple:
    return (
        tuple(int(v) for v in x.shape),
        tuple(int(v) for v in weight.shape),
        stride,
        padding,
        dilation,
        int(groups),
        bias is not None,
        bool(enable_hf32),
        str(x.dtype),
        str(x.device),
    )


def _forward_once(ext, x, weight, bias, stride, padding, dilation, groups, enable_hf32):
    key = _launch_key(x, weight, stride, padding, dilation, groups, bias, enable_hf32)
    # 见模块上方说明：丢掉前三次，返回第四次。次数绑在 CANN 9.0.0 的实测上。
    if key not in _WARMED_KEYS:
        _WARMED_KEYS.add(key)
        for _ in range(3):
            y = ext.conv3d_gelu_forward(x, weight, bias, stride, padding, dilation, groups, enable_hf32)
            torch.npu.synchronize()
            del y
    return ext.conv3d_gelu_forward(x, weight, bias, stride, padding, dilation, groups, enable_hf32)


def _out_dim(size: int, pad: int, kernel: int, dilation: int, stride: int) -> int:
    return (size + 2 * pad - dilation * (kernel - 1) - 1) // stride + 1


def _use_direct_1x1(x, weight, stride, padding, dilation, groups) -> bool:
    """与 Conv3dGeluUseDirect1x1 一致：这条路径没有整幅私有格式缓冲。"""
    if groups != 1 or tuple(int(v) for v in weight.shape[2:]) != (1, 1, 1):
        return False
    if stride != (1, 1, 1) or dilation != (1, 1, 1) or padding != (0, 0, 0):
        return False
    cin = int(x.shape[1])
    cout = int(weight.shape[0])
    return cin >= 8 and cout <= 64


# 输出不超过这个字节数时一次 launch。再切只会多付启动开销，省不下多少显存。
_ONE_LAUNCH_BYTES = 256 * 1024 * 1024
# 更大的输出按大约 48MB 一块切。160³ 前向约 20ms（整幅 14ms，一块一平面是 85ms），
# 新增显存约 1.4GB，整幅和拆开的卷积都是 3.0GB。
_CHUNK_BYTES = 48 * 1024 * 1024


def _auto_depth_planes(depth: int, plane_bytes: int) -> int:
    depth = max(depth, 1)
    output_bytes = depth * max(plane_bytes, 1)
    if output_bytes <= _ONE_LAUNCH_BYTES:
        return depth
    planes = max(8, _CHUNK_BYTES // max(plane_bytes, 1))
    return max(1, min(depth, int(planes)))


def _depth_planes(depth: int, plane_bytes: int) -> int:
    """沿输出深度切块。小体积一次算完；大体积每块大约 48MB。

    ``OIL_GAS_OPS_CONV3D_GELU_DEPTH_CHUNK<=0`` 恢复整幅，正数则固定平面数。
    若设置了 ``OIL_GAS_OPS_CONV3D_GELU_BWD_MB``，按该字节预算换算平面数。
    """
    if os.environ.get("OIL_GAS_OPS_CONV3D_GELU_BWD_MB"):
        return _output_depth_chunk(depth, plane_bytes, _bwd_budget_bytes())
    raw = os.environ.get("OIL_GAS_OPS_CONV3D_GELU_DEPTH_CHUNK")
    if raw is None:
        return _auto_depth_planes(depth, plane_bytes)
    try:
        value = int(raw)
    except ValueError:
        value = 0
    if value <= 0:
        return max(depth, 1)
    return max(1, min(depth, value))


def _iter_depth_windows(x, depth_out: int, kernel_d: int, stride, padding, dilation, planes: int):
    """产出与整幅 conv3d 等价的输出深度窗，以及对应的输入切片（D 维 pad 已补上）。"""
    sd = stride[0]
    pd = padding[0]
    dd = dilation[0]
    depth_in = int(x.shape[2])
    step = max(1, planes)
    for do0 in range(0, depth_out, step):
        do1 = min(depth_out, do0 + step)
        d0 = do0 * sd - pd
        d1 = (do1 - 1) * sd + (kernel_d - 1) * dd + 1 - pd
        src0 = max(d0, 0)
        src1 = min(d1, depth_in)
        pad_before = int(src0 - d0)
        pad_after = int(d1 - src1)
        x_slice = x[:, :, src0:src1]
        if pad_before or pad_after:
            x_slice = F.pad(x_slice, (0, 0, 0, 0, pad_before, pad_after))
        yield do0, do1, x_slice, pad_before, src0, src1


def _forward_memory_bound(ext, x, weight, bias, stride, padding, dilation, groups, enable_hf32):
    """1×1 直写一次 launch。大体积沿 D 分块，避免私有格式缓冲按整幅分配。"""
    if _use_direct_1x1(x, weight, stride, padding, dilation, groups):
        return _forward_once(ext, x, weight, bias, stride, padding, dilation, groups, enable_hf32)
    n = int(x.shape[0])
    cout = int(weight.shape[0])
    kd, kh, kw = (int(v) for v in weight.shape[2:])
    sd, sh, sw = stride
    pd, ph, pw = padding
    dd, dh, dw = dilation
    depth_out = _out_dim(int(x.shape[2]), pd, kd, dd, sd)
    ho = _out_dim(int(x.shape[3]), ph, kh, dh, sh)
    wo = _out_dim(int(x.shape[4]), pw, kw, dw, sw)
    plane_bytes = cout * ho * wo * x.element_size()
    planes = _depth_planes(depth_out, plane_bytes)
    if planes >= depth_out:
        return _forward_once(ext, x, weight, bias, stride, padding, dilation, groups, enable_hf32)
    y = torch.empty((n, cout, depth_out, ho, wo), device=x.device, dtype=x.dtype)
    for do0, do1, x_slice, _pad_before, _src0, _src1 in _iter_depth_windows(
        x, depth_out, kd, stride, padding, dilation, planes
    ):
        y_slice = _forward_once(
            ext, x_slice, weight, bias, stride, (0, ph, pw), dilation, groups, enable_hf32
        )
        y[:, :, do0:do1].copy_(y_slice)
        del y_slice
    return y


def _require_custom() -> bool:
    return os.environ.get("OIL_GAS_OPS_REQUIRE_ASCENDC_CONV3D_GELU", "0") == "1"


def _impl() -> str:
    return os.environ.get("OIL_GAS_OPS_CONV3D_GELU_IMPL", "auto").strip().lower()


def _is_npu_tensor(x: torch.Tensor) -> bool:
    return x.device.type == "npu"


def _bwd_budget_bytes() -> int:
    """反向重算的输出深度预算。越大越接近一次卷积，越小峰值越低。"""
    mb = os.environ.get("OIL_GAS_OPS_CONV3D_GELU_BWD_MB", "256")
    try:
        value = int(mb)
    except ValueError:
        value = 1024
    return max(1, value) * 1024 * 1024


def _output_depth_chunk(depth: int, plane_bytes: int, budget: int | None = None) -> int:
    """反向重算时沿 D 切块，避免整幅预激活和卷积 workspace 同时常驻。"""
    if budget is None:
        budget = _bwd_budget_bytes()
    if depth <= 1 or plane_bytes <= 0:
        return max(depth, 1)
    planes = max(1, budget // plane_bytes)
    return int(min(depth, planes))


def _call_backward(ext, ctx, grad_output, x, weight, bias, needs, padding):
    """重算 conv3d、做 GELU 反向，再做 conv3d 反向。每个形状预热一次。"""
    key = ("bwd",) + _launch_key(
        x, weight, ctx.stride, padding, ctx.dilation, ctx.groups, bias, ctx.enable_hf32
    )

    def launch():
        grad_x, grad_w, grad_b = ext.conv3d_gelu_backward(
            grad_output, x, weight, bias, ctx.stride, padding, ctx.dilation, int(ctx.groups),
            bool(needs[0]), bool(needs[1]), bool(needs[2]) and bias is not None, bool(ctx.enable_hf32),
        )
        return (
            grad_x if needs[0] else None,
            grad_w if needs[1] else None,
            grad_b if needs[2] and bias is not None else None,
        )

    # 新 shape 的前三次反向 launch 不可用，第四次才和标杆对得上。每个 key 只预热一次。
    if key not in _WARMED_KEYS:
        _WARMED_KEYS.add(key)
        for _ in range(3):
            launch()
            torch.npu.synchronize()
    return launch()


def _accumulate_chunked_backward(ext, ctx, grad_output, x, weight, bias, needs, planes):
    kd = int(weight.shape[2])
    depth_out = int(grad_output.shape[2])
    ph, pw = ctx.padding[1], ctx.padding[2]
    grad_x = torch.zeros_like(x) if needs[0] else None
    grad_w = torch.zeros_like(weight) if needs[1] else None
    grad_b = torch.zeros_like(bias) if (bias is not None and needs[2]) else None
    for do0, do1, x_slice, pad_before, src0, src1 in _iter_depth_windows(
        x, depth_out, kd, ctx.stride, ctx.padding, ctx.dilation, planes
    ):
        gx, gw, gb = _call_backward(
            ext, ctx, grad_output[:, :, do0:do1], x_slice, weight, bias, needs, (0, ph, pw)
        )
        if grad_x is not None and gx is not None:
            grad_x[:, :, src0:src1] += gx[:, :, pad_before:pad_before + (src1 - src0)]
        if grad_w is not None and gw is not None:
            grad_w += gw
        if grad_b is not None and gb is not None:
            grad_b += gb
        del gx, gw, gb, x_slice
    return grad_x, grad_w, grad_b


def _backward_fused(ctx, grad_output, x, weight, bias, needs):
    """NPU 反向。小体积一次算完；大体积按和前向相同的深度块累加梯度。"""
    global _CHUNKED_BWD_PRIMED
    ext = _load_extension()
    depth_out = int(grad_output.shape[2])
    ho, wo = int(grad_output.shape[3]), int(grad_output.shape[4])
    plane_bytes = int(grad_output.shape[1]) * ho * wo * grad_output.element_size()
    planes = _depth_planes(depth_out, plane_bytes)
    if planes >= depth_out:
        return _call_backward(ext, ctx, grad_output, x, weight, bias, needs, ctx.padding)
    if not _CHUNKED_BWD_PRIMED:
        _CHUNKED_BWD_PRIMED = True
        primed = _accumulate_chunked_backward(
            ext, ctx, grad_output, x, weight, bias, needs, planes
        )
        del primed
        torch.npu.synchronize()
    return _accumulate_chunked_backward(
        ext, ctx, grad_output, x, weight, bias, needs, planes
    )


def _backward_chunked(ctx, grad_output, x, weight, bias, needs):
    """按输出平面重算 conv+gelu。一块的预激活就是反向多出来的显存。"""
    stride = ctx.stride
    padding = ctx.padding
    dilation = ctx.dilation
    groups = ctx.groups
    ph, pw = padding[1], padding[2]
    kd = int(weight.shape[2])
    do = int(grad_output.shape[2])
    ho, wo = int(grad_output.shape[3]), int(grad_output.shape[4])
    plane_bytes = int(grad_output.shape[1]) * ho * wo * grad_output.element_size()
    planes = _depth_planes(do, plane_bytes)

    grad_x = torch.zeros_like(x) if needs[0] else None
    grad_w = torch.zeros_like(weight) if needs[1] else None
    grad_b = torch.zeros_like(bias) if (bias is not None and needs[2]) else None
    profile = os.environ.get("OIL_GAS_OPS_CONV3D_GELU_BWD_PROFILE", "0") == "1"
    if profile:
        import time as _time
        torch.npu.synchronize()
        _t0 = _time.perf_counter()

    for do0, do1, x_slice, pad_before, src0, src1 in _iter_depth_windows(
        x, do, kd, stride, padding, dilation, planes
    ):
        x_req = x_slice.detach().requires_grad_(bool(needs[0]))
        w_req = weight.detach().requires_grad_(bool(needs[1]))
        b_req = None if bias is None else bias.detach().requires_grad_(bool(needs[2]))
        with torch.enable_grad():
            y = torch_conv3d_gelu(
                x_req, w_req, b_req, stride, (0, ph, pw), dilation, groups,
            )
        go = grad_output[:, :, do0:do1]
        targets = []
        slots = []
        if needs[0]:
            targets.append(x_req)
            slots.append("x")
        if needs[1]:
            targets.append(w_req)
            slots.append("w")
        if b_req is not None and needs[2]:
            targets.append(b_req)
            slots.append("b")
        grads = torch.autograd.grad(y, targets, go, allow_unused=True)
        found = dict(zip(slots, grads))
        if grad_x is not None and found.get("x") is not None:
            gx = found["x"]
            valid = gx[:, :, pad_before:pad_before + (src1 - src0)]
            grad_x[:, :, src0:src1] += valid
        if grad_w is not None and found.get("w") is not None:
            grad_w += found["w"]
        if grad_b is not None and found.get("b") is not None:
            grad_b += found["b"]
        del y, x_slice, x_req, w_req, b_req, go, grads
    if profile:
        torch.npu.synchronize()
        nchunk = (do + planes - 1) // planes
        print(
            f"[bwd] x={tuple(x.shape)} w={tuple(weight.shape)} k={kd} "
            f"planes={planes} nchunk={nchunk} sec={_time.perf_counter()-_t0:.3f}",
            flush=True,
        )
    return grad_x, grad_w, grad_b


def torch_conv3d_gelu(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    stride: tuple[int, int, int],
    padding: tuple[int, int, int],
    dilation: tuple[int, int, int],
    groups: int,
) -> torch.Tensor:
    return F.gelu(
        F.conv3d(x, weight, bias, stride=stride, padding=padding, dilation=dilation, groups=groups),
        approximate="tanh",
    )


class _Conv3dGeluFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        stride: tuple[int, int, int],
        padding: tuple[int, int, int],
        dilation: tuple[int, int, int],
        groups: int,
        enable_hf32: bool,
    ) -> torch.Tensor:
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups
        ctx.has_bias = bias is not None
        ctx.enable_hf32 = bool(enable_hf32)
        # 只保存卷积输入和参数。GELU 的预激活在反向重算，不占用二者之间的激活显存。
        ctx.save_for_backward(x, weight, *(tuple() if bias is None else (bias,)))

        ext = _load_extension()
        return _forward_memory_bound(
            ext, x, weight, bias, stride, padding, dilation, groups, enable_hf32
        )

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        saved = ctx.saved_tensors
        x = saved[0]
        weight = saved[1]
        bias = saved[2] if ctx.has_bias else None

        needs = ctx.needs_input_grad
        if _is_npu_tensor(x):
            grad_x, grad_w, grad_b = _backward_fused(ctx, grad_output, x, weight, bias, needs)
        else:
            grad_x, grad_w, grad_b = _backward_chunked(ctx, grad_output, x, weight, bias, needs)
        return grad_x, grad_w, grad_b, None, None, None, None, None


def conv3d_gelu(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    stride=1,
    padding=0,
    dilation=1,
    groups: int = 1,
    *,
    padding_mode: str = "zeros",
    enable_hf32: bool = False,
) -> torch.Tensor:
    """y = gelu(conv3d(x, weight, bias))，NPU 上走 Conv3DGelu。"""
    if int(groups) != 1:
        raise ValueError(f"conv3d_gelu only supports groups=1, got {groups}")
    stride3 = _triple(stride, "stride")
    padding3 = _triple(padding, "padding")
    dilation3 = _triple(dilation, "dilation")

    if padding_mode != "zeros":
        x = F.pad(x, _pad6_from_conv3d(padding3), mode=padding_mode)
        padding3 = (0, 0, 0)

    impl = _impl()
    use_ascendc = impl in ("ascendc", "auto") and _is_npu_tensor(x)
    if impl == "ascendc" and not _is_npu_tensor(x) and _require_custom():
        raise RuntimeError("OIL_GAS_OPS_CONV3D_GELU_IMPL=ascendc requires an NPU tensor.")
    if not use_ascendc:
        return torch_conv3d_gelu(x, weight, bias, stride3, padding3, dilation3, groups)

    try:
        _load_extension()
    except ImportError as exc:
        if _require_custom() or impl == "ascendc":
            raise RuntimeError("conv3d_gelu_npu extension is required but could not be imported.") from exc
        return torch_conv3d_gelu(x, weight, bias, stride3, padding3, dilation3, groups)

    return _Conv3dGeluFunction.apply(
        x, weight, bias, stride3, padding3, dilation3, int(groups), bool(enable_hf32)
    )
