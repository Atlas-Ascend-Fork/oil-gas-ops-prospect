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
 * ROIAlignRotated / ROIAlignRotatedGrad 的 pybind 直调绑定。
 *
 * 前向: aclnnROIAlignRotated(input, rois, spatial_scale, PH, PW, sampling_ratio) -> out
 * 反向: aclnnROIAlignRotatedGrad(grad_output, rois, spatial_scale, PH, PW, sampling_ratio, N,C,H,W) -> grad_input
 */

#include <cstdint>
#include <vector>

#include <torch/extension.h>
#include <ATen/ATen.h>

#include "acl/acl.h"
#include "aclnn/acl_meta.h"
#include "acl_tensor_utils.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

extern "C" {
aclnnStatus aclnnROIAlignRotatedGetWorkspaceSize(const aclTensor* input, const aclTensor* rois, double spatialScale,
                                                 int64_t pooledHeight, int64_t pooledWidth, int64_t samplingRatio,
                                                 const aclTensor* out, uint64_t* workspaceSize,
                                                 aclOpExecutor** executor);
aclnnStatus aclnnROIAlignRotated(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

aclnnStatus aclnnROIAlignRotatedGradGetWorkspaceSize(const aclTensor* gradOutput, const aclTensor* rois,
                                                     double spatialScale, int64_t pooledHeight, int64_t pooledWidth,
                                                     int64_t samplingRatio, int64_t batchSize, int64_t channels,
                                                     int64_t height, int64_t width, const aclTensor* out,
                                                     uint64_t* workspaceSize, aclOpExecutor** executor);
aclnnStatus aclnnROIAlignRotatedGrad(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                     aclrtStream stream);
}

namespace {

using OgPybind::BindInputAddr;
using OgPybind::BindOutputAddr;
using OgPybind::Destroy;
using OgPybind::MakeExecutorRepeatable;
using OgPybind::ShapeOf;
using OgPybind::StableWorkspace;

std::vector<int64_t> ContiguousStrides(const std::vector<int64_t>& shape)
{
    std::vector<int64_t> strides(shape.size(), 1);
    for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i) {
        strides[static_cast<size_t>(i)] = strides[static_cast<size_t>(i + 1)] * shape[static_cast<size_t>(i + 1)];
    }
    return strides;
}

aclTensor* CreateNdTensor(const std::vector<int64_t>& shape, void* deviceData)
{
    auto strides = ContiguousStrides(shape);
    return aclCreateTensor(shape.data(), shape.size(), ACL_FLOAT, strides.data(), 0, ACL_FORMAT_ND, shape.data(),
                           shape.size(), deviceData);
}

void CheckFloatNd(const at::Tensor& t, const char* name)
{
    TORCH_CHECK(t.device().is_privateuseone(), name, " must be an NPU tensor");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(t.scalar_type() == at::kFloat, name, " must be float32");
}

} // namespace

at::Tensor roi_align_rotated_forward(const at::Tensor& input, const at::Tensor& rois, int64_t pooledHeight,
                                     int64_t pooledWidth, double spatialScale, int64_t samplingRatio)
{
    CheckFloatNd(input, "input");
    CheckFloatNd(rois, "rois");
    TORCH_CHECK(input.dim() == 4, "input must be 4D");
    TORCH_CHECK(rois.dim() == 2 && rois.size(1) == 6, "rois must be [K,6]");
    TORCH_CHECK(pooledHeight > 0 && pooledWidth > 0, "pooled size must be positive");
    TORCH_CHECK(samplingRatio >= 0, "sampling_ratio must be >= 0");

    const int64_t numRois = rois.size(0);
    const int64_t channels = input.size(1);
    auto out = at::empty({numRois, channels, pooledHeight, pooledWidth}, input.options().requires_grad(false));
    if (out.numel() == 0) {
        return out;
    }

    const auto inputShape = ShapeOf(input);
    const auto roisShape = ShapeOf(rois);
    const auto outShape = ShapeOf(out);
    aclTensor* inputT = CreateNdTensor(inputShape, input.data_ptr());
    aclTensor* roisT = CreateNdTensor(roisShape, rois.data_ptr());
    aclTensor* outT = CreateNdTensor(outShape, out.data_ptr());

    aclrtStream stream = c10_npu::getCurrentNPUStream().stream(true);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnROIAlignRotatedGetWorkspaceSize(inputT, roisT, spatialScale, pooledHeight, pooledWidth,
                                                    samplingRatio, outT, &workspaceSize, &executor);
    TORCH_CHECK(ret == ACL_SUCCESS, "aclnnROIAlignRotatedGetWorkspaceSize failed, ret=", static_cast<int>(ret));
    MakeExecutorRepeatable(executor);
    BindInputAddr(executor, 0, inputT, input.data_ptr());
    BindInputAddr(executor, 1, roisT, rois.data_ptr());
    BindOutputAddr(executor, 0, outT, out.data_ptr());

    void* workspace = nullptr;
    at::Tensor& workspaceTensor = StableWorkspace(false, workspaceSize, input);
    if (workspaceSize > 0) {
        workspace = workspaceTensor.data_ptr();
    }
    ret = aclnnROIAlignRotated(workspace, workspaceSize, executor, stream);
    TORCH_CHECK(ret == ACL_SUCCESS, "aclnnROIAlignRotated failed, ret=", static_cast<int>(ret));

    Destroy(inputT);
    Destroy(roisT);
    Destroy(outT);
    return out;
}

at::Tensor roi_align_rotated_backward(const at::Tensor& gradOutput, const at::Tensor& rois, int64_t batchSize,
                                      int64_t channels, int64_t height, int64_t width, int64_t pooledHeight,
                                      int64_t pooledWidth, double spatialScale, int64_t samplingRatio)
{
    CheckFloatNd(gradOutput, "grad_output");
    CheckFloatNd(rois, "rois");
    TORCH_CHECK(gradOutput.dim() == 4, "grad_output must be 4D");
    TORCH_CHECK(rois.dim() == 2 && rois.size(1) == 6, "rois must be [K,6]");
    TORCH_CHECK(batchSize >= 0 && channels >= 0 && height >= 0 && width >= 0, "invalid input shape");

    auto gradInput = at::empty({batchSize, channels, height, width}, gradOutput.options().requires_grad(false));
    if (gradInput.numel() == 0) {
        return gradInput;
    }

    const auto gradShape = ShapeOf(gradOutput);
    const auto roisShape = ShapeOf(rois);
    const auto outShape = ShapeOf(gradInput);
    aclTensor* gradT = CreateNdTensor(gradShape, gradOutput.data_ptr());
    aclTensor* roisT = CreateNdTensor(roisShape, rois.data_ptr());
    aclTensor* outT = CreateNdTensor(outShape, gradInput.data_ptr());

    aclrtStream stream = c10_npu::getCurrentNPUStream().stream(true);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnROIAlignRotatedGradGetWorkspaceSize(gradT, roisT, spatialScale, pooledHeight, pooledWidth,
                                                        samplingRatio, batchSize, channels, height, width, outT,
                                                        &workspaceSize, &executor);
    TORCH_CHECK(ret == ACL_SUCCESS, "aclnnROIAlignRotatedGradGetWorkspaceSize failed, ret=", static_cast<int>(ret));
    MakeExecutorRepeatable(executor);
    BindInputAddr(executor, 0, gradT, gradOutput.data_ptr());
    BindInputAddr(executor, 1, roisT, rois.data_ptr());
    BindOutputAddr(executor, 0, outT, gradInput.data_ptr());

    void* workspace = nullptr;
    at::Tensor& workspaceTensor = StableWorkspace(true, workspaceSize, gradOutput);
    if (workspaceSize > 0) {
        workspace = workspaceTensor.data_ptr();
    }
    ret = aclnnROIAlignRotatedGrad(workspace, workspaceSize, executor, stream);
    TORCH_CHECK(ret == ACL_SUCCESS, "aclnnROIAlignRotatedGrad failed, ret=", static_cast<int>(ret));

    Destroy(gradT);
    Destroy(roisT);
    Destroy(outT);
    return gradInput;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("roi_align_rotated_forward", &roi_align_rotated_forward, "ROIAlignRotated forward");
    m.def("roi_align_rotated_backward", &roi_align_rotated_backward, "ROIAlignRotated backward");
}
