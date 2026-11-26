/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * 仅用于测试的 pybind 包装：把 detectron2 的 ROIAlignRotated CPU C++ 实现
 * （ROIAlignRotated_cpu.cpp）编成独立 golden 扩展。
 *
 * 该文件不进入 whl，也不参与算子构建；由 verify_detectron2_cpu_golden.py
 * 通过 torch.utils.cpp_extension.load 临时编译。
 */

#include <torch/extension.h>

#include "ROIAlignRotated.h"

namespace {

at::Tensor ForwardCpu(const at::Tensor& input, const at::Tensor& rois, double spatialScale, int64_t pooledHeight,
                      int64_t pooledWidth, int64_t samplingRatio)
{
    return detectron2::ROIAlignRotated_forward_cpu(input, rois, static_cast<float>(spatialScale),
                                                   static_cast<int>(pooledHeight), static_cast<int>(pooledWidth),
                                                   static_cast<int>(samplingRatio));
}

at::Tensor BackwardCpu(const at::Tensor& grad, const at::Tensor& rois, double spatialScale, int64_t pooledHeight,
                       int64_t pooledWidth, int64_t batchSize, int64_t channels, int64_t height, int64_t width,
                       int64_t samplingRatio)
{
    return detectron2::ROIAlignRotated_backward_cpu(
        grad, rois, static_cast<float>(spatialScale), static_cast<int>(pooledHeight), static_cast<int>(pooledWidth),
        static_cast<int>(batchSize), static_cast<int>(channels), static_cast<int>(height), static_cast<int>(width),
        static_cast<int>(samplingRatio));
}

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("roi_align_rotated_forward_cpu", &ForwardCpu, "detectron2 ROIAlignRotated CPU forward");
    m.def("roi_align_rotated_backward_cpu", &BackwardCpu, "detectron2 ROIAlignRotated CPU backward");
}
