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

"""ROIAlignRotated：detectron2 旋转 ROIAlign 的 NPU 版本（含反向）。

语义对齐 detectron2 `detectron2/layers/csrc/ROIAlignRotated/ROIAlignRotated_cuda.cu`
（Apache-2.0）：

    input [N,C,H,W] + rois [K,6] -> out [K,C,PH,PW]
    rois 每行 = (batch_idx, x_ctr, y_ctr, width, height, angle_deg)

环境变量：
- ``OIL_GAS_OPS_ROI_ALIGN_ROTATED_IMPL=torch|ascendc|auto``（默认 auto）
- ``OIL_GAS_OPS_REQUIRE_ASCENDC_ROI_ALIGN_ROTATED=1`` 强制要求自定义算子
"""

from __future__ import annotations

import importlib
import math
import os
from functools import lru_cache

import torch
from torch import nn


@lru_cache(maxsize=1)
def _load_extension():
    return importlib.import_module("roi_align_rotated_npu")


def _require_custom() -> bool:
    return os.environ.get("OIL_GAS_OPS_REQUIRE_ASCENDC_ROI_ALIGN_ROTATED", "0") == "1"


def _impl() -> str:
    return os.environ.get("OIL_GAS_OPS_ROI_ALIGN_ROTATED_IMPL", "auto").strip().lower()


def _is_npu(x: torch.Tensor) -> bool:
    return x.device.type == "npu"


def _pair(value):
    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise ValueError(f"output_size must be an int or a 2-element sequence, got {value!r}")
        return int(value[0]), int(value[1])
    return int(value), int(value)


def _bilinear(plane: torch.Tensor, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """plane [C,H,W] 上的双线性插值；y/x 为 0 维 tensor。返回 [C]。"""
    channels, height, width = plane.shape
    if height == 0 or width == 0:
        return plane.new_zeros((channels,))

    valid = (y >= -1.0) & (y <= float(height)) & (x >= -1.0) & (x <= float(width))
    yc = y.clamp(min=0.0, max=float(height - 1))
    xc = x.clamp(min=0.0, max=float(width - 1))
    y_low = yc.floor().long()
    x_low = xc.floor().long()
    y_high = (y_low + 1).clamp(max=height - 1)
    x_high = (x_low + 1).clamp(max=width - 1)

    ly = yc - y_low.to(yc.dtype)
    lx = xc - x_low.to(xc.dtype)
    hy = 1.0 - ly
    hx = 1.0 - lx

    v1 = plane[:, y_low, x_low]
    v2 = plane[:, y_low, x_high]
    v3 = plane[:, y_high, x_low]
    v4 = plane[:, y_high, x_high]
    value = hy * hx * v1 + hy * lx * v2 + ly * hx * v3 + ly * lx * v4
    return torch.where(valid, value, torch.zeros_like(value))


def torch_roi_align_rotated(
    input: torch.Tensor,
    rois: torch.Tensor,
    output_size,
    spatial_scale: float = 1.0,
    sampling_ratio: int = 0,
) -> torch.Tensor:
    """纯 PyTorch 参考实现，逐行对齐 detectron2 ROIAlignRotated CUDA 语义。"""
    if input.dim() != 4:
        raise ValueError(f"input must be 4D, got {tuple(input.shape)}")
    if rois.dim() != 2 or rois.shape[1] != 6:
        raise ValueError(f"rois must be [K,6], got {tuple(rois.shape)}")
    if sampling_ratio < 0:
        raise ValueError(f"sampling_ratio must be >= 0, got {sampling_ratio}")

    pooled_h, pooled_w = _pair(output_size)
    if pooled_h <= 0 or pooled_w <= 0:
        raise ValueError(f"output_size must be positive, got {output_size!r}")

    n, channels, _, _ = input.shape
    num_rois = rois.shape[0]
    if num_rois == 0:
        return input.new_zeros((0, channels, pooled_h, pooled_w))

    rois_d = rois.detach().to(device=input.device, dtype=input.dtype)
    roi_outputs = []
    for ki in range(num_rois):
        batch_idx = int(rois_d[ki, 0].item())
        if batch_idx < 0 or batch_idx >= n:
            roi_outputs.append(input.new_zeros((channels, pooled_h, pooled_w)))
            continue

        center_x = rois_d[ki, 1]
        center_y = rois_d[ki, 2]
        roi_w = rois_d[ki, 3] * spatial_scale
        roi_h = rois_d[ki, 4] * spatial_scale
        theta = rois_d[ki, 5] * math.pi / 180.0
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)

        roi_center_w = center_x * spatial_scale - 0.5
        roi_center_h = center_y * spatial_scale - 0.5
        if float(roi_w) < 0.0 or float(roi_h) < 0.0:
            roi_outputs.append(input.new_zeros((channels, pooled_h, pooled_w)))
            continue

        bin_w = roi_w / pooled_w
        bin_h = roi_h / pooled_h
        if sampling_ratio > 0:
            grid_h = int(sampling_ratio)
            grid_w = int(sampling_ratio)
        else:
            grid_h = int(math.ceil(float(roi_h) / pooled_h))
            grid_w = int(math.ceil(float(roi_w) / pooled_w))
        if grid_h <= 0 or grid_w <= 0:
            roi_outputs.append(input.new_zeros((channels, pooled_h, pooled_w)))
            continue

        count = max(grid_h * grid_w, 1)
        roi_start_h = -roi_h / 2.0
        roi_start_w = -roi_w / 2.0
        plane = input[batch_idx]

        rows = []
        for ph in range(pooled_h):
            bins = []
            for pw in range(pooled_w):
                acc = input.new_zeros((channels,))
                for iy in range(grid_h):
                    yy = roi_start_h + ph * bin_h + (iy + 0.5) * bin_h / grid_h
                    for ix in range(grid_w):
                        xx = roi_start_w + pw * bin_w + (ix + 0.5) * bin_w / grid_w
                        y = yy * cos_t - xx * sin_t + roi_center_h
                        x = yy * sin_t + xx * cos_t + roi_center_w
                        acc = acc + _bilinear(plane, y, x)
                bins.append(acc / count)
            rows.append(torch.stack(bins, dim=1))  # [C, PW]
        roi_outputs.append(torch.stack(rows, dim=1))  # [C, PH, PW]

    return torch.stack(roi_outputs, dim=0)


def _ascendc_forward(input, rois, pooled_h, pooled_w, spatial_scale, sampling_ratio):
    ext = _load_extension()
    return ext.roi_align_rotated_forward(
        input.contiguous(),
        rois.contiguous(),
        pooled_h,
        pooled_w,
        float(spatial_scale),
        int(sampling_ratio),
    )


def _ascendc_backward(grad_output, rois, n, c, h, w, pooled_h, pooled_w, spatial_scale, sampling_ratio):
    ext = _load_extension()
    return ext.roi_align_rotated_backward(
        grad_output.contiguous(),
        rois.contiguous(),
        n,
        c,
        h,
        w,
        pooled_h,
        pooled_w,
        float(spatial_scale),
        int(sampling_ratio),
    )


class _ROIAlignRotatedFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, rois, pooled_h, pooled_w, spatial_scale, sampling_ratio, use_ascendc: bool):
        ctx.pooled_h = pooled_h
        ctx.pooled_w = pooled_w
        ctx.spatial_scale = spatial_scale
        ctx.sampling_ratio = sampling_ratio
        ctx.input_shape = tuple(input.shape)
        ctx.use_ascendc = bool(use_ascendc)
        ctx.save_for_backward(input, rois)
        if use_ascendc:
            try:
                return _ascendc_forward(input, rois, pooled_h, pooled_w, spatial_scale, sampling_ratio)
            except Exception:
                if _require_custom():
                    raise
        return torch_roi_align_rotated(input, rois, (pooled_h, pooled_w), spatial_scale, sampling_ratio)

    @staticmethod
    def backward(ctx, grad_output):
        input_ref, rois = ctx.saved_tensors
        n, c, h, w = ctx.input_shape
        if ctx.use_ascendc:
            try:
                grad_input = _ascendc_backward(
                    grad_output,
                    rois,
                    n,
                    c,
                    h,
                    w,
                    ctx.pooled_h,
                    ctx.pooled_w,
                    ctx.spatial_scale,
                    ctx.sampling_ratio,
                )
                return grad_input, None, None, None, None, None, None
            except Exception:
                if _require_custom():
                    raise

        # torch 回退：对参考实现重新求导。
        inp = input_ref.detach().requires_grad_(True)
        out = torch_roi_align_rotated(
            inp, rois, (ctx.pooled_h, ctx.pooled_w), ctx.spatial_scale, ctx.sampling_ratio
        )
        (grad_input,) = torch.autograd.grad(out, inp, grad_output)
        return grad_input, None, None, None, None, None, None


def roi_align_rotated(
    input: torch.Tensor,
    rois: torch.Tensor,
    output_size,
    spatial_scale: float = 1.0,
    sampling_ratio: int = 0,
) -> torch.Tensor:
    """旋转 ROIAlign。NPU + auto/ascendc 时走 AscendC 融合算子，否则走纯 torch 参考。"""
    if input.dim() != 4:
        raise ValueError(f"input must be 4D, got {tuple(input.shape)}")
    if rois.dim() != 2 or rois.shape[1] != 6:
        raise ValueError(f"rois must be [K,6], got {tuple(rois.shape)}")
    if sampling_ratio < 0:
        raise ValueError(f"sampling_ratio must be >= 0, got {sampling_ratio}")

    pooled_h, pooled_w = _pair(output_size)
    orig_dtype = input.dtype

    # detectron2：fp16 输入先转 float；AscendC kernel 只支持 float32。
    if orig_dtype == torch.float16:
        input = input.float()
        rois = rois.float()
    if rois.dtype != input.dtype:
        rois = rois.to(input.dtype)

    impl = _impl()
    use_ascendc = impl in ("ascendc", "auto") and _is_npu(input) and input.dtype == torch.float32
    if impl == "ascendc" and not _is_npu(input) and _require_custom():
        raise RuntimeError("OIL_GAS_OPS_ROI_ALIGN_ROTATED_IMPL=ascendc requires NPU tensors")

    out = _ROIAlignRotatedFn.apply(
        input, rois, pooled_h, pooled_w, float(spatial_scale), int(sampling_ratio), use_ascendc
    )
    if out.dtype != orig_dtype:
        out = out.to(orig_dtype)
    return out


class ROIAlignRotated(nn.Module):
    """detectron2 `ROIAlignRotated` 的等价模块。"""

    def __init__(self, output_size, spatial_scale, sampling_ratio):
        super().__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale
        self.sampling_ratio = sampling_ratio

    def forward(self, input: torch.Tensor, rois: torch.Tensor) -> torch.Tensor:
        return roi_align_rotated(input, rois, self.output_size, self.spatial_scale, self.sampling_ratio)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(output_size={self.output_size}, "
            f"spatial_scale={self.spatial_scale}, sampling_ratio={self.sampling_ratio})"
        )
