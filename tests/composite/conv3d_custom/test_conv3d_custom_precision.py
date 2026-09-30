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

"""conv3d_custom 精度测试（一）：主要取值场景组合，前向 + 反向。

用例分层：

1. 基础用例（PROOF_CASES）：stride=1 / dilation=1 的 9 组等价性用例；
   float64 最大绝对误差阈值 1e-12，float32 L2
   相对误差阈值 1e-5（绝对误差随数据幅度放大，rtol 口径见 _common.py）；
2. 扩展用例（EXTENDED_CASES）：stride>1 / dilation>1（含 input 不能被 stride
   整除的 output_padding 路径）、非立方 kernel、无 bias；
3. 组合扫描（COMBINATION_GRID）：kernel × padding × stride × dilation × groups
   × bias 的全组合（非立方空间 (9,8,10)，float64），覆盖维度整除性差异；
4. 模块级行为：padding_mode != 'zeros' 且 padding>0 的分流、state_dict 与
   nn.Conv3d 互通、use_custom 门控、CPU 回退路径。

实际训练提取的 11 种调用模式见 test_conv3d_custom_training_patterns.py；
NPU float32 实机尺寸的精度与性能见 bench_conv3d_custom_perf.py。
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path
from typing import NamedTuple

import pytest
import torch
import torch.nn.functional as F
from torch import nn

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from _common import (  # noqa: E402
    SEED,
    TOL_F64,
    assert_errors,
    criterion_for,
    function_level_errors,
    k3,
    module_level_errors,
    randn,
    tensor_diff,
)
from oil_gas_ops_prospect.conv3d_custom import Conv3d  # noqa: E402
from oil_gas_ops_prospect.conv3d_custom.functional import conv3d  # noqa: E402

DTYPES = [torch.float64, torch.float32]
DTYPE_IDS = ["f64", "f32"]

# 等价性基础用例 9 组（均为 stride=1、dilation=1）。
# 用例字段依次为：id、B、Cin、Cout、D、H、W、K、P、G。
PROOF_CASES = [
    ("proof1_same_k3", 2, 4, 8, 5, 5, 5, 3, 1, 1),
    ("proof2_valid_k3", 2, 4, 8, 5, 5, 5, 3, 0, 1),
    ("proof3_full_k3", 2, 4, 8, 5, 5, 5, 3, 2, 1),
    ("proof4_k1", 2, 4, 8, 5, 5, 5, 1, 0, 1),
    ("proof5_g3_same", 2, 6, 6, 5, 5, 5, 3, 1, 3),
    ("proof6_g3_valid", 2, 6, 12, 5, 5, 5, 3, 0, 3),
    ("proof7_g2_noncube_space", 2, 4, 4, 7, 6, 5, 3, 1, 2),
    ("proof8_conv0_like", 1, 16, 8, 4, 4, 4, 1, 0, 1),
    ("proof9_conv3_like", 1, 8, 8, 5, 5, 5, 3, 1, 1),
]

# 扩展路径：任意 stride / dilation（基础用例未覆盖的路径）。
# 用例字段依次为：id、B、Cin、Cout、D、H、W、K、P、G、S、DIL、bias。
EXTENDED_CASES = [
    ("ext_stride2_k3_pad1", 2, 4, 8, 6, 6, 6, 3, 1, 1, 2, 1, True),
    ("ext_stride2_uneven", 2, 4, 8, 8, 8, 8, 3, 0, 1, 2, 1, True),
    ("ext_stride2_g2", 2, 6, 6, 9, 9, 9, 3, 1, 2, 2, 1, True),
    ("ext_stride3_k5", 1, 4, 4, 10, 10, 10, 5, 2, 1, 3, 1, True),
    ("ext_dilation2_k3_pad2", 2, 4, 8, 7, 7, 7, 3, 2, 1, 1, 2, True),
    ("ext_stride2_dilation2", 2, 4, 8, 9, 9, 9, 3, 2, 1, 2, 2, True),
    ("ext_kernel_noncube", 2, 4, 4, 9, 8, 7, (3, 5, 7), 1, 1, 1, 1, True),
    ("ext_no_bias", 1, 4, 8, 5, 5, 5, 3, 1, 1, 1, 1, False),
]

# 组合扫描：kernel × padding × stride × dilation × groups × bias 全组合。
# 空间取非立方 (9, 8, 10)，使各维度的整除性 / output_padding 路径不同。
KERNELS = (1, 3, 5)
PADDINGS = (0, 1, 2)
STRIDES = (1, 2)
DILATIONS = (1, 2)
GROUPS = (1, 3)
BIAS_FLAGS = (True, False)
GRID_SPATIAL = (9, 8, 10)
GRID_SHAPE = (2, 6, 6, *GRID_SPATIAL)  # 字段依次为 B、Cin、Cout、D、H、W（Cout 可被 GROUPS 整除）


def _geometrically_valid(k, p, d):
    """有效核 d*(k-1)+1 必须不超过 空间维度 + 2*p，否则 F.conv3d 本身报错。"""
    eff = d * (k - 1) + 1
    return all(dim + 2 * p >= eff for dim in GRID_SPATIAL)


GRID_CASES = [
    pytest.param(
        *GRID_SHAPE, k, p, s, d, g, b,
        id=f"k{k}_p{p}_s{s}_d{d}_g{g}_{'bias' if b else 'nobias'}",
        marks=(
            []
            if _geometrically_valid(k, p, d)
            else [pytest.mark.skip(reason=f"有效核 {d}*({k}-1)+1 超过空间维度 {GRID_SPATIAL}")]
        ),
    )
    for k, p, s, d, g, b in itertools.product(
        KERNELS, PADDINGS, STRIDES, DILATIONS, GROUPS, BIAS_FLAGS
    )
]


class _CaseInputs(NamedTuple):
    """_case_inputs 的具名返回（G.FNM.05）：单测输入张量与卷积参数。"""

    x: torch.Tensor
    weight: torch.Tensor
    bias: torch.Tensor | None
    stride: int
    padding: int
    dilation: int
    groups: int


def _case_inputs(case, has_stride_dil, dtype):
    """按用例表生成 _CaseInputs（输入张量 + 卷积参数）。"""
    if has_stride_dil:
        _, b, cin, cout, d, h, w, k, p, g, s, dil, has_bias = case
    else:
        _, b, cin, cout, d, h, w, k, p, g = case
        s, dil, has_bias = 1, 1, True
    gen = torch.Generator().manual_seed(SEED)
    x = randn((b, cin, d, h, w), gen, dtype)
    weight = randn((cout, cin // g, *k3(k)), gen, dtype)
    bias = randn((cout,), gen, dtype) if has_bias else None
    return _CaseInputs(x, weight, bias, s, p, dil, g)


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
@pytest.mark.parametrize("case", PROOF_CASES, ids=[c[0] for c in PROOF_CASES])
def test_function_level_proof_cases(case, dtype):
    """基础用例：stride=1、dilation=1 下三梯度与原生等价。"""
    metric, tol = criterion_for(dtype)
    ci = _case_inputs(case, has_stride_dil=False, dtype=dtype)
    errs = function_level_errors(
        ci.x, ci.weight, ci.bias, ci.stride, ci.padding, ci.dilation, ci.groups,
        metric=metric,
    )
    assert_errors(errs, case[0], tol)


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
@pytest.mark.parametrize("case", EXTENDED_CASES, ids=[c[0] for c in EXTENDED_CASES])
def test_function_level_extended_paths(case, dtype):
    """扩展路径：stride>1 / dilation>1 / 非立方 kernel / 无 bias。"""
    metric, tol = criterion_for(dtype)
    ci = _case_inputs(case, has_stride_dil=True, dtype=dtype)
    errs = function_level_errors(
        ci.x, ci.weight, ci.bias, ci.stride, ci.padding, ci.dilation, ci.groups,
        metric=metric,
    )
    assert_errors(errs, case[0], tol)


@pytest.mark.parametrize(
    "b, cin, cout, d, h, w, k, p, s, dil, g, has_bias", GRID_CASES
)
def test_function_level_combination_grid(b, cin, cout, d, h, w, k, p, s, dil, g, has_bias):
    """主要取值组合扫描：144 组参数组合的前向 + 三梯度等价（float64）。"""
    case_id = f"k{k}_p{p}_s{s}_d{d}_g{g}_{'bias' if has_bias else 'nobias'}"
    spec = ("grid", b, cin, cout, d, h, w, k, p, g, s, dil, has_bias)
    ci = _case_inputs(spec, has_stride_dil=True, dtype=torch.float64)
    errs = function_level_errors(
        ci.x, ci.weight, ci.bias, ci.stride, ci.padding, ci.dilation, ci.groups
    )
    assert_errors(errs, case_id, TOL_F64)


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
@pytest.mark.parametrize("padding_mode", ["replicate", "reflect"])
@pytest.mark.parametrize("padding", [1, 2])
def test_module_padding_mode_nonzero(padding_mode, padding, dtype):
    """padding_mode != 'zeros' 且 padding>0：F.pad 分流后 conv_padding=0 的前/反向等价。"""
    metric, tol = criterion_for(dtype)
    errs = module_level_errors(
        (2, 4, 6, 6, 6), 8, 3, stride=1, padding=padding,
        padding_mode=padding_mode, dtype=dtype, metric=metric,
    )
    assert_errors(errs, f"{padding_mode}_p{padding}", tol)


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
def test_module_stride2_replicate(dtype):
    """模块级 stride=2 + replicate：分流与膨胀路径叠加。"""
    metric, tol = criterion_for(dtype)
    errs = module_level_errors(
        (2, 4, 9, 9, 9), 8, 3, stride=2, padding=1,
        padding_mode="replicate", dtype=dtype, metric=metric,
    )
    assert_errors(errs, "module_stride2_replicate", tol)


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
def test_state_dict_interop_with_torch_conv3d(dtype):
    """state_dict 与 nn.Conv3d 完全互通（strict 加载 + 双向 + 两条前向路径输出一致）。"""
    metric, tol = criterion_for(dtype)
    gen = torch.Generator().manual_seed(SEED)
    ref = nn.Conv3d(4, 8, 3, padding=1, dtype=dtype)
    x = randn((2, 4, 6, 6, 6), gen, dtype)

    cus = Conv3d(4, 8, 3, padding=1).to(dtype)
    cus.load_state_dict(ref.state_dict())

    out_ref = ref(x)
    assert tensor_diff(cus(x), out_ref, metric) <= tol  # CPU 原生路径

    cus.use_custom = True
    assert tensor_diff(cus(x), out_ref, metric) <= tol  # 自定义 Function 路径

    ref2 = nn.Conv3d(4, 8, 3, padding=1, dtype=dtype)
    ref2.load_state_dict(cus.state_dict())
    assert tensor_diff(ref2(x), out_ref, metric) <= tol


def test_use_custom_gating_follows_npu_availability():
    """use_custom 跟随 torch.npu 可用性（当前实现不再限制 stride/dilation）。"""
    has_npu = hasattr(torch, "npu") and hasattr(torch.npu, "empty_cache")
    assert Conv3d(4, 8, 3, stride=2).use_custom == has_npu
    assert Conv3d(4, 8, 3, dilation=2).use_custom == has_npu


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
def test_cpu_fallback_matches_native_for_stride4(dtype):
    """回退分支（F.conv3d）与原生一致（训练模式 7 的回退侧）。

    torch_npu 存在时 use_custom 默认为 True（即使输入在 CPU），
    显式置 False 以确定性地覆盖回退分支。
    """
    metric, tol = criterion_for(dtype)
    gen = torch.Generator().manual_seed(SEED)
    x = randn((1, 7, 12, 12, 12), gen, dtype)
    weight = randn((19, 7, 8, 8, 8), gen, dtype)

    m = Conv3d(7, 19, 8, stride=4, bias=False).to(dtype)
    with torch.no_grad():
        m.weight.copy_(weight)
    m.use_custom = False

    expected = F.conv3d(x, weight, None, 4, 0, 1, 1)
    assert tensor_diff(m(x), expected, metric) <= tol


@pytest.mark.parametrize("padding", ["valid", "same"])
@pytest.mark.parametrize("padding_mode", ["zeros", "reflect"])
def test_string_padding_matches_native(padding, padding_mode):
    """PyTorch's valid/same padding strings work through the custom module path."""
    gen = torch.Generator().manual_seed(SEED)
    x_ref = randn((1, 2, 5, 6, 7), gen, torch.float64).requires_grad_(True)
    x_custom = x_ref.detach().clone().requires_grad_(True)
    ref = nn.Conv3d(
        2, 3, (2, 3, 2), padding=padding, padding_mode=padding_mode, dtype=torch.float64
    )
    custom = Conv3d(
        2, 3, (2, 3, 2), padding=padding, padding_mode=padding_mode
    ).to(torch.float64)
    custom.load_state_dict(ref.state_dict())
    custom.use_custom = True

    out_ref = ref(x_ref)
    out_custom = custom(x_custom)
    assert torch.allclose(out_custom, out_ref, rtol=1e-12, atol=1e-12)

    out_functional = conv3d(x_ref.detach(), custom.weight, custom.bias, padding=padding)
    ref_functional = F.conv3d(x_ref.detach(), custom.weight, custom.bias, padding=padding)
    assert torch.allclose(out_functional, ref_functional, rtol=1e-12, atol=1e-12)

    grad = randn(out_ref.shape, torch.Generator().manual_seed(SEED + 1), torch.float64)
    grads_ref = torch.autograd.grad(out_ref, (x_ref, ref.weight, ref.bias), grad)
    grads_custom = torch.autograd.grad(out_custom, (x_custom, custom.weight, custom.bias), grad)
    for actual, expected in zip(grads_custom, grads_ref):
        assert torch.allclose(actual, expected, rtol=1e-12, atol=1e-12)
