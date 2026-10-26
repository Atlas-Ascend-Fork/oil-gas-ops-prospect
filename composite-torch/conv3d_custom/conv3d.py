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

"""Conv3d：支持参数域内兼容 torch.nn.Conv3d 的 nn.Module 形态。

- 检测到 ``torch.npu`` 接口时 ``use_custom`` 开启，前/反向走
  ``functional.conv3d``；否则回退 ``F.conv3d``。
- ``kernel_size``、``stride``、数值 ``padding``、``dilation`` 支持 int 或
  3 元组，``groups`` 为 int；``padding_mode`` 支持 zeros / reflect /
  replicate / circular。字符串 padding 不受支持。
- 构造器不提供原生模块的 ``device`` / ``dtype`` 关键字；weight/bias 为顶层
  参数，state_dict key 与 nn.Conv3d 互通。

用法::

    from oil_gas_ops_prospect.conv3d_custom import Conv3d

函数式入口、反向构造与精度验证说明见 ``functional.py``。
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .functional import _padding_for_string, conv3d


def _as_triple(value):
    """把 int 或 3 元组统一成 (d, h, w) 三元组。"""
    if isinstance(value, int):
        return (value, value, value)
    if isinstance(value, (tuple, list)):
        return tuple(value)
    raise ValueError(f"expected int or 3-tuple, got {type(value).__name__}: {value!r}")


def _fpad_order(padding):
    """把 (d, h, w) 的 padding 重排成 F.pad 要求的 (w,w, h,h, d,d) 顺序。"""
    d, h, w = _as_triple(padding)
    return (w, w, h, h, d, d)


def _validate_channel_groups(in_channels, out_channels, groups, padding_mode):
    if groups <= 0:
        raise ValueError("groups must be a positive integer")
    if in_channels % groups != 0:
        raise ValueError("in_channels must be divisible by groups")
    if out_channels % groups != 0:
        raise ValueError("out_channels must be divisible by groups")
    if padding_mode not in {"zeros", "reflect", "replicate", "circular"}:
        raise ValueError(f"unsupported padding_mode: {padding_mode!r}")


class Conv3d(nn.Module):
    """支持参数域内兼容 nn.Conv3d；检测到 torch.npu 时默认走自定义前/反向，state_dict key 互通。"""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode='zeros',
    ):
        super().__init__()
        _validate_channel_groups(in_channels, out_channels, groups, padding_mode)
        # 构造参数成对登记为实例属性，供 repr 与外部读取
        config = (
            ("in_channels", in_channels),
            ("out_channels", out_channels),
            ("kernel_size", kernel_size),
            ("stride", stride),
            ("padding", padding),
            ("dilation", dilation),
            ("groups", groups),
            ("padding_mode", padding_mode),
        )
        for attr, value in config:
            setattr(self, attr, value)

        # torch_npu 已导入即认为 NPU 可用：前/反向切换到自定义实现，
        # 否则保持原生 F.conv3d 路径
        self.use_custom = hasattr(torch, "npu") and hasattr(torch.npu, "empty_cache")

        kernel = _as_triple(kernel_size)
        in_per_group = in_channels // groups
        self.weight = nn.Parameter(torch.empty(out_channels, in_per_group, *kernel))
        self.register_parameter(
            "bias", nn.Parameter(torch.empty(out_channels)) if bias else None
        )

        self.reset_parameters()

    def reset_parameters(self):
        """初始化权重与偏置，分布与 nn.Conv3d 对齐（同 seed 逐位一致）。"""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            bound = self._bias_bound()
            nn.init.uniform_(self.bias, -bound, bound)

    def _bias_bound(self):
        """偏置均匀初始化的边界，即 fan_in 的 -1/2 次方。

        fan_in 按「每组输入通道数 × kernel 元素个数」直接计算，不经由
        nn.init 的受保护成员 _calculate_fan_in_and_fan_out。
        """
        per_group_in = self.weight.size(1)
        kernel_elems = self.weight[0, 0].numel()
        fan_in = per_group_in * kernel_elems
        return 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0

    def forward(self, x):
        # 非 zeros 模式先显式补边，卷积本身以 padding=0 执行；
        # 补边在计算图内，梯度经 F.pad 的 autograd 正确回传
        if self.padding_mode != "zeros":
            if isinstance(self.padding, str):
                pad = _padding_for_string(self.padding, self.kernel_size, self.stride, self.dilation)
            else:
                pad = _fpad_order(self.padding)
            if any(pad):
                x = F.pad(x, pad, mode=self.padding_mode)
            conv_padding = 0
        else:
            conv_padding = self.padding

        if not self.use_custom:
            return F.conv3d(
                x, self.weight, self.bias, self.stride, conv_padding,
                self.dilation, self.groups,
            )
        return conv3d(
            x, self.weight, self.bias, self.stride, conv_padding,
            self.dilation, self.groups,
        )

    def extra_repr(self):
        fields = [
            f"in_channels={self.in_channels}",
            f"out_channels={self.out_channels}",
            f"kernel_size={self.kernel_size}",
            f"stride={self.stride}",
        ]
        if self.padding != 0:
            fields.append(f"padding={self.padding}")
        if self.dilation != 1:
            fields.append(f"dilation={self.dilation}")
        if self.groups != 1:
            fields.append(f"groups={self.groups}")
        if self.bias is None:
            fields.append("bias=False")
        if self.padding_mode != "zeros":
            fields.append(f"padding_mode={self.padding_mode}")
        fields.append(f"use_custom={self.use_custom}")
        return ", ".join(fields)
