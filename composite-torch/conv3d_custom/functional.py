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

"""conv3d：torch.nn.functional.conv3d 的等价替换（函数式入口）。

始终经 ``_Conv3dFunction`` 组织前/反向：与原生数学等价（支持任意
stride / dilation / groups 与双阶反向），NPU 上可绕开原生 backward 对
大 kernel depthwise 的 L1 限制。nn.Module 形态（Conv3d，含 NPU 回退
逻辑与 state_dict 互通）见 ``conv3d.py``。

反向构造：grad_input 用 flip+转置 channel 后的 ``F.conv3d``（stride>1 时先对
grad_output 做零膨胀 + output_padding 补齐；padding 为负时裁剪），
grad_weight 调 ``torch.nn.grad.conv3d_weight``，grad_bias 对 grad_output 求和。

精度验证：float64 下三梯度与原生 autograd 数学等价（阈值 1e-12 内）；
NPU float32 实际训练尺寸全模式对比通过（grad_weight 因内核累加顺序差异在
float32 正常范围内）；固化用例见 tests/composite/conv3d_custom/。
"""

import torch
import torch.nn.functional as F
from torch.autograd import Function
from torch.nn.modules.utils import _triple


class _Conv3dFunction(Function):
    """ 对标 nn.F.conv3d """

    @staticmethod
    def forward(ctx, x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        # 保存参数用于反向传播
        ctx.stride = _triple(stride)
        ctx.padding = _triple(padding)
        ctx.dilation = _triple(dilation)
        ctx.groups = groups

        # 保存输入、权重和偏置
        ctx.save_for_backward(x, weight, bias)

        # 前向传播：使用PyTorch原生卷积
        output = F.conv3d(x, weight, bias, ctx.stride, ctx.padding, ctx.dilation, ctx.groups)

        return output

    # torch.autograd.Function 协议要求 backward 对每个前向输入各返回一个梯度
    # （7 个输入对应 7 个返回值），不可封装为具名对象，故豁免 G.FNM.05。
    @staticmethod
    def backward(ctx, grad_output):
        # 获取保存的输入、权重和偏置
        x, weight, bias = ctx.saved_tensors
        stride = ctx.stride
        padding = ctx.padding
        dilation = ctx.dilation
        groups = ctx.groups

        grad_input = grad_weight = grad_bias = None

        # 1. 计算输入梯度 — 转置卷积 (支持任意 stride/dilation)
        #
        # 数学原理:
        #   forward: out[j] = Σ_k W[k] * in[j*s + d*k - p]   (cross-correlation)
        #   backward: grad_in[m] = Σ_k W_flip[k] * grad_out_dilated[m + d*k - pad]
        #   其中:
        #     - W_flip = flip(W) 后转置 in/out channel
        #     - pad = dilation*(K-1) - padding          (dilation>1 时不同于 K-1-p)
        #     - grad_out_dilated = grad_output 元素间插入 stride-1 个零
        #       + output_padding 补齐到 input 尺寸
        #     - 卷积 stride=1 (stride 由膨胀处理), dilation=dilation
        if ctx.needs_input_grad[0]:
            kd, kh, kw = weight.shape[2], weight.shape[3], weight.shape[4]
            k = (kd, kh, kw)

            # 翻转 weight 的空间维度 (cross-correlation -> convolution)
            weight_flip = torch.flip(weight, dims=[2, 3, 4])

            # 转置 in/out channel 维度
            if groups == 1:
                weight_flip = weight_flip.permute(1, 0, 2, 3, 4)
            else:
                g = groups
                out_ch, in_per_g = weight.shape[0], weight.shape[1]
                weight_flip = weight_flip.view(g, out_ch // g, in_per_g, kd, kh, kw)
                weight_flip = weight_flip.permute(0, 2, 1, 3, 4, 5).reshape(
                    g * in_per_g, out_ch // g, kd, kh, kw)

            # stride>1: 在 grad_output 元素间插入 stride-1 个零, 并用 output_padding
            # 补偿 forward 时因 input 尺寸不能被 stride 整除而丢弃的边缘像素
            grad_output_dilated = _Conv3dFunction._dilate_grad_output(
                grad_output, stride, x.shape[2:], padding, dilation, k)

            # 转置卷积 padding: dilation*(K-1) - p
            # dilation=1 时该式退化为 K-1-p，与原始实现一致
            pad = [dilation[d] * (k[d] - 1) - padding[d] for d in range(3)]

            # padding 为负时需裁剪 grad_output_dilated 两侧 (等价于负 padding)
            grad_output_dilated = _Conv3dFunction._crop_negative_padding(
                grad_output_dilated, pad)
            pad = tuple(max(0, p) for p in pad)

            grad_input = F.conv3d(
                grad_output_dilated, weight_flip, bias=None,
                stride=(1, 1, 1), padding=pad, dilation=dilation, groups=groups,
            )

        # 2. 计算权重梯度
        if ctx.needs_input_grad[1]:
            # 使用torch.nn.grad.conv3d_weight来计算权重梯度
            grad_weight = torch.nn.grad.conv3d_weight(
                x, weight.shape, grad_output, stride, padding, dilation, groups
            )

        # 3. 计算偏置梯度
        if bias is not None and ctx.needs_input_grad[2]:
            # 对除通道维度外的所有维度求和
            grad_bias = grad_output.sum(dim=[0, 2, 3, 4])

        return (grad_input, grad_weight, grad_bias, None, None, None, None)

    @staticmethod
    def _dilate_grad_output(grad_output, stride, input_shape, padding, dilation, kernel):
        """在 grad_output 的空间维度元素间插入 stride-1 个零，并用 output_padding
        补齐到原始 input 形状。

        转置卷积中 stride>1 的标准处理：将 grad_output 按 stride 膨胀后再以
        stride=1 做正向卷积。output_padding 补偿 forward 时因 input 尺寸不能
        被 stride 整除而丢弃的边缘像素。

        参数:
            grad_output  : [B, C, D_out, H_out, W_out]
            stride       : (s_d, s_h, s_w)
            input_shape  : (D_in, H_in, W_in) — 原始 input 的空间尺寸
            padding      : (p_d, p_h, p_w)
            dilation     : (d_d, d_h, d_w)
            kernel       : (k_d, k_h, k_w)
        返回:
            膨胀后的 grad_output, 形状 [B, C, new_D, new_H, new_W]
        """
        if all(s == 1 for s in stride):
            return grad_output

        batch, chan, d_out, h_out, w_out = grad_output.shape
        s_d, s_h, s_w = stride

        # 各维 output_padding 为 (input + 2*p - d*(K-1) - 1) 对 s 取模，
        # 即 forward 时因 input 尺寸不能被 stride 整除而丢弃的边缘像素数
        op_d = (input_shape[0] + 2 * padding[0] - dilation[0] * (kernel[0] - 1) - 1) % s_d
        op_h = (input_shape[1] + 2 * padding[1] - dilation[1] * (kernel[1] - 1) - 1) % s_h
        op_w = (input_shape[2] + 2 * padding[2] - dilation[2] * (kernel[2] - 1) - 1) % s_w

        new_d = (d_out - 1) * s_d + 1 + op_d
        new_h = (h_out - 1) * s_h + 1 + op_h
        new_w = (w_out - 1) * s_w + 1 + op_w

        result = torch.zeros(batch, chan, new_d, new_h, new_w,
                             dtype=grad_output.dtype, device=grad_output.device)
        result[:, :, ::s_d, ::s_h, ::s_w] = grad_output
        return result

    @staticmethod
    def _crop_negative_padding(grad_output_dilated, pad):
        """转置卷积 padding 为负时，裁剪膨胀张量两侧（等价于负 padding 的卷积）。"""
        crop = [max(0, -p) for p in pad]
        if not any(c > 0 for c in crop):
            return grad_output_dilated
        slices = [slice(None), slice(None)]
        for c in crop:
            slices.append(slice(c, -c) if c > 0 else slice(None))
        return grad_output_dilated[tuple(slices)]


def _padding_for_string(padding, kernel_size, stride, dilation):
    """Return F.pad's six-value order for the string padding accepted by conv3d."""
    if padding == "valid":
        return (0, 0, 0, 0, 0, 0)
    if padding != "same":
        raise ValueError(f"expected padding to be 'valid' or 'same', got {padding!r}")

    def triple(value):
        if isinstance(value, int):
            return (value, value, value)
        result = tuple(value)
        if len(result) != 3:
            raise ValueError(f"expected a 3-tuple, got {value!r}")
        return result

    if any(s != 1 for s in triple(stride)):
        raise ValueError("padding='same' is not supported for strided convolutions")
    kernel = triple(kernel_size)
    dilation = triple(dilation)
    totals = tuple(d * (k - 1) for k, d in zip(kernel, dilation))
    before = tuple(total // 2 for total in totals)
    after = tuple(total - left for total, left in zip(totals, before))
    d0, h0, w0 = before
    d1, h1, w1 = after
    return (w0, w1, h0, h1, d0, d1)


def conv3d(x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    """torch.nn.functional.conv3d 的等价替换（函数式入口，仅引用的包不同）。

    始终经 _Conv3dFunction 组织前/反向：与原生数学等价（支持任意
    stride / dilation 与双阶反向），NPU 上可绕开原生 backward 对大 kernel
    depthwise 的 L1 限制。nn.Module 形态与 NPU 回退逻辑见 Conv3d。
    """
    if isinstance(padding, str):
        explicit_padding = _padding_for_string(padding, weight.shape[2:], stride, dilation)
        if any(explicit_padding):
            x = F.pad(x, explicit_padding)
        padding = 0
    return _Conv3dFunction.apply(x, weight, bias, stride, padding, dilation, groups)
