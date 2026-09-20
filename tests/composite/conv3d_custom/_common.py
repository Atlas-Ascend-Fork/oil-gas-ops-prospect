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

"""conv3d_custom 测试共用工具：误差对比、确定性随机数据与公共常量。

误差对比有两个层级：

- ``function_level_errors``：conv3d（函数式入口）vs F.conv3d + 原生 autograd；
- ``module_level_errors``：Conv3d 模块（强制 use_custom）vs nn.Conv3d，
  覆盖 padding_mode 分流（F.pad 后 conv_padding=0）等模块级路径。

两者都返回 ``{"fwd", "gi", "gw", "gb"}`` 四个误差；``metric="abs"`` 为最大绝对
误差（float64 数学等价口径），``metric="rel"`` 为 L2 相对误差（float32 口径，
``criterion_for`` 按 dtype 自动选择）。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from oil_gas_ops_prospect.conv3d_custom import Conv3d
from oil_gas_ops_prospect.conv3d_custom.functional import conv3d

TOL_F64 = 1e-12
TOL_F32 = 1e-5
SEED = 2026


def k3(kernel):
    """int 或 3 元组 kernel 统一为 (kd, kh, kw)。"""
    if isinstance(kernel, int):
        return (kernel, kernel, kernel)
    return tuple(kernel)


def randn(shape, gen, dtype=torch.float64):
    return torch.randn(*shape, generator=gen, dtype=dtype)


def rel_err(a, b):
    """L2 相对误差（与 complex_mul 微基准口径一致）。"""
    a = a.detach()
    b = b.detach()
    num = torch.linalg.norm((a - b).reshape(-1))
    den = torch.linalg.norm(b.reshape(-1)).clamp_min(1e-30)
    return float(num / den)


def tensor_diff(a, b, metric):
    if metric == "rel":
        return rel_err(a, b)
    return (a - b).abs().max().item()


def criterion_for(dtype):
    """float64 用绝对误差（数学等价口径）；float32 用 L2 相对误差（rtol 口径）。"""
    if dtype == torch.float64:
        return "abs", TOL_F64
    return "rel", TOL_F32


def assert_errors(errs, case_id, tol):
    for key, value in errs.items():
        assert value <= tol, f"{case_id}: {key} 误差 {value:.3e} 超过阈值 {tol:.0e}"


def function_level_errors(x, weight, bias, stride, padding, dilation, groups, metric="abs"):
    """conv3d 与 F.conv3d 原生 autograd 逐项对比，返回四个误差。"""
    out_ref = F.conv3d(x, weight, bias, stride, padding, dilation, groups)
    gout = randn(out_ref.shape, torch.Generator().manual_seed(SEED), x.dtype)

    custom_leaves = [x.detach().clone().requires_grad_(True), weight.detach().clone().requires_grad_(True)]
    if bias is not None:
        custom_leaves.append(bias.detach().clone().requires_grad_(True))
    out_c = conv3d(
        custom_leaves[0], custom_leaves[1],
        custom_leaves[2] if bias is not None else None,
        stride, padding, dilation, groups,
    )
    grads_c = torch.autograd.grad(out_c, custom_leaves, gout)

    native_leaves = [t.detach().clone().requires_grad_(True) for t in (x, weight)]
    if bias is not None:
        native_leaves.append(bias.detach().clone().requires_grad_(True))
    out_n = F.conv3d(
        native_leaves[0], native_leaves[1],
        native_leaves[2] if bias is not None else None,
        stride, padding, dilation, groups,
    )
    grads_n = torch.autograd.grad(out_n, native_leaves, gout)

    errs = {
        "fwd": tensor_diff(out_c, out_n, metric),
        "gi": tensor_diff(grads_c[0], grads_n[0], metric),
        "gw": tensor_diff(grads_c[1], grads_n[1], metric),
    }
    errs["gb"] = (
        tensor_diff(grads_c[2], grads_n[2], metric) if bias is not None else 0.0
    )
    return errs


def module_level_errors(
    in_shape,
    cout,
    kernel,
    *,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
    bias=True,
    padding_mode="zeros",
    dtype=torch.float64,
    metric="abs",
):
    """Conv3d（强制 use_custom）与 nn.Conv3d 全参数对比，返回四个误差。"""
    cin = in_shape[1]
    ref = nn.Conv3d(
        cin, cout, kernel, stride=stride, padding=padding, dilation=dilation,
        groups=groups, bias=bias, padding_mode=padding_mode, dtype=dtype,
    )
    cus = Conv3d(
        cin, cout, kernel, stride=stride, padding=padding, dilation=dilation,
        groups=groups, bias=bias, padding_mode=padding_mode,
    ).to(dtype)
    cus.load_state_dict(ref.state_dict())
    cus.use_custom = True  # CPU 上强制走自定义 Function 路径

    gen = torch.Generator().manual_seed(SEED)
    x_ref = randn(in_shape, gen, dtype).requires_grad_(True)
    x_cus = x_ref.detach().clone().requires_grad_(True)
    cus_params = [x_cus, cus.weight] + ([cus.bias] if bias else [])
    ref_params = [x_ref, ref.weight] + ([ref.bias] if bias else [])

    out_ref = ref(x_ref)
    gout = randn(out_ref.shape, torch.Generator().manual_seed(SEED), dtype)
    grads_ref = torch.autograd.grad(out_ref, ref_params, gout)

    out_cus = cus(x_cus)
    grads_cus = torch.autograd.grad(out_cus, cus_params, gout)

    return {
        "fwd": tensor_diff(out_cus, out_ref, metric),
        "gi": tensor_diff(grads_cus[0], grads_ref[0], metric),
        "gw": tensor_diff(grads_cus[1], grads_ref[1], metric),
        "gb": (
            tensor_diff(grads_cus[2], grads_ref[2], metric) if bias else 0.0
        ),
    }
