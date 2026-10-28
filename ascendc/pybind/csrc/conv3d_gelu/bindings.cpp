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

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <tuple>
#include <vector>

#include <torch/extension.h>
#include <ATen/ATen.h>
#include <c10/util/Optional.h>

#include "acl/acl.h"
#include "aclnn/acl_meta.h"
#include "aclnnop/aclnn_convolution.h"
#include "aclnnop/aclnn_convolution.h"
#include "aclnnop/aclnn_convolution_backward.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

// Conv3DGelu 现在提供了对齐 native conv3d 的自定义 L2 aclnn 接口
// aclnnConv3dGeluCustom：**直接接收 NCDHW 的 x/weight/y**，内部用 l0op::TransData
// 在同一 executor 内完成 NCDHW<->私有格式（NDC1HWC0 / FRACTAL_Z_3D）的转换并 launch
// cube conv + vector gelu。所以 binding 不再做任何 host 侧 transdata，直接把 torch 的
// NCDHW 张量喂进去即可（与 torch_npu 调 conv3d 的体验一致）。
//
// 该函数从 cust_opapi.so 导出（visibility=default），此处直接声明其原型。
extern "C" {
aclnnStatus aclnnConv3dGeluCustomGetWorkspaceSize(const aclTensor* x, const aclTensor* weight,
                                                  const aclTensor* biasOptional, const aclIntArray* stride,
                                                  const aclIntArray* padding, const aclIntArray* dilation,
                                                  int64_t groups, aclTensor* out, uint64_t* workspaceSize,
                                                  aclOpExecutor** executor);
aclnnStatus aclnnConv3dGeluCustom(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);
aclnnStatus aclnnConv3dGeluGradGetWorkspaceSize(const aclTensor* z, const aclTensor* gradY, aclTensor* gradZ,
                                                uint64_t* workspaceSize, aclOpExecutor** executor);
aclnnStatus aclnnConv3dGeluGrad(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);
}

namespace {

std::vector<int64_t> ContiguousStrides(const std::vector<int64_t>& shape)
{
    std::vector<int64_t> strides(shape.size(), 1);
    for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i) {
        strides[static_cast<size_t>(i)] = strides[static_cast<size_t>(i + 1)] * shape[static_cast<size_t>(i + 1)];
    }
    return strides;
}

std::vector<int64_t> Normalize3(c10::IntArrayRef values, int64_t fallback)
{
    if (values.size() == 1) {
        return {values[0], values[0], values[0]};
    }
    if (values.size() == 3) {
        return {values[0], values[1], values[2]};
    }
    return {fallback, fallback, fallback};
}

std::array<int64_t, 6> NormalizePads(c10::IntArrayRef padding)
{
    std::array<int64_t, 6> pads = {0, 0, 0, 0, 0, 0};
    if (padding.size() == 1) {
        for (int i = 0; i < 3; ++i) {
            pads[i * 2] = padding[0];
            pads[i * 2 + 1] = padding[0];
        }
    } else if (padding.size() == 3) {
        for (int i = 0; i < 3; ++i) {
            pads[i * 2] = padding[i];
            pads[i * 2 + 1] = padding[i];
        }
    } else if (padding.size() == 6) {
        for (int i = 0; i < 6; ++i) {
            pads[i] = padding[i];
        }
    }
    return pads;
}

int64_t ConvOutDim(int64_t in, int64_t padBefore, int64_t padAfter, int64_t kernel, int64_t dilation, int64_t stride)
{
    return (in + padBefore + padAfter - dilation * (kernel - 1) - 1) / stride + 1;
}

// 逻辑/存储 shape 一致的 NCDHW(或 ND) aclTensor，device buffer 取自 torch 张量。
aclTensor* CreateNcdhwTensor(const std::vector<int64_t>& shape, aclFormat format, void* deviceData)
{
    auto strides = ContiguousStrides(shape);
    return aclCreateTensor(shape.data(), shape.size(), ACL_FLOAT, strides.data(), 0, format, shape.data(), shape.size(),
                           deviceData);
}

using Conv3dGeluCustomV2GetWorkspaceSizeFn = aclnnStatus (*)(const aclTensor*, const aclTensor*, const aclTensor*,
                                                             const aclIntArray*, const aclIntArray*, const aclIntArray*,
                                                             int64_t, bool, aclTensor*, uint64_t*, aclOpExecutor**);

aclnnStatus GetWorkspaceSize(const aclTensor* xTensor, const aclTensor* wTensor, const aclTensor* biasTensor,
                             const aclIntArray* strideArr, const aclIntArray* padArr, const aclIntArray* dilArr,
                             int64_t groups, bool enableHf32, aclTensor* yTensor, uint64_t* workspaceSize,
                             aclOpExecutor** executor)
{
    if (!enableHf32) {
        return aclnnConv3dGeluCustomGetWorkspaceSize(xTensor, wTensor, biasTensor, strideArr, padArr, dilArr, groups,
                                                     yTensor, workspaceSize, executor);
    }

    const char* symbolName = "aclnnConv3dGeluCustomV2GetWorkspaceSize";
    auto* symbol = dlsym(RTLD_DEFAULT, symbolName);
    if (symbol == nullptr) {
        void* handle = dlopen("libcust_opapi.so", RTLD_NOW | RTLD_LOCAL);
        TORCH_CHECK(
            handle != nullptr,
            "enable_hf32=True requires libcust_opapi.so from the rebuilt conv3d_gelu OPP package. dlopen failed: ",
            dlerror());
        symbol = dlsym(handle, symbolName);
    }
    TORCH_CHECK(symbol != nullptr, "enable_hf32=True requires rebuilt and reinstalled conv3d_gelu OPP package with ",
                symbolName, ". dlsym failed: ", dlerror());
    auto fn = reinterpret_cast<Conv3dGeluCustomV2GetWorkspaceSizeFn>(symbol);
    return fn(xTensor, wTensor, biasTensor, strideArr, padArr, dilArr, groups, enableHf32, yTensor, workspaceSize,
              executor);
}

} // namespace

// Conv3D + GELU 融合算子前向：y = gelu(conv3d(x, weight, bias))（tanh 近似），全程 NCDHW。
at::Tensor conv3d_gelu_forward(const at::Tensor& x, const at::Tensor& weight, const c10::optional<at::Tensor>& bias,
                               c10::IntArrayRef stride, c10::IntArrayRef padding, c10::IntArrayRef dilation,
                               int64_t groups, bool enableHf32)
{
    TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1, "x must be on NPU");
    TORCH_CHECK(x.scalar_type() == at::kFloat && weight.scalar_type() == at::kFloat,
                "Conv3DGelu only supports float32");
    TORCH_CHECK(x.dim() == 5 && weight.dim() == 5, "x/weight must be 5D NCDHW");

    auto stride3 = Normalize3(stride, 1);
    auto dilation3 = Normalize3(dilation, 1);
    auto pads = NormalizePads(padding);

    int64_t n = x.size(0);
    int64_t di = x.size(2);
    int64_t hi = x.size(3);
    int64_t wi = x.size(4);
    int64_t cout = weight.size(0);
    int64_t kd = weight.size(2);
    int64_t kh = weight.size(3);
    int64_t kw = weight.size(4);
    int64_t dout = ConvOutDim(di, pads[0], pads[1], kd, dilation3[0], stride3[0]);
    int64_t hout = ConvOutDim(hi, pads[2], pads[3], kh, dilation3[1], stride3[1]);
    int64_t wout = ConvOutDim(wi, pads[4], pads[5], kw, dilation3[2], stride3[2]);

    // 直接传 NCDHW 连续张量，输出也分配为 NCDHW（无任何 host transdata）。
    at::Tensor xc = x.contiguous();
    at::Tensor wc = weight.contiguous();
    at::Tensor out = at::empty({n, cout, dout, hout, wout}, x.options());

    std::vector<int64_t> xShape = {n, x.size(1), di, hi, wi};
    std::vector<int64_t> wShape = {cout, weight.size(1), kd, kh, kw};
    std::vector<int64_t> yShape = {n, cout, dout, hout, wout};

    aclTensor* xTensor = CreateNcdhwTensor(xShape, ACL_FORMAT_NCDHW, xc.data_ptr());
    aclTensor* wTensor = CreateNcdhwTensor(wShape, ACL_FORMAT_NCDHW, wc.data_ptr());
    aclTensor* yTensor = CreateNcdhwTensor(yShape, ACL_FORMAT_NCDHW, out.data_ptr());

    at::Tensor biasC;
    aclTensor* biasTensor = nullptr;
    if (bias.has_value() && bias->defined()) {
        biasC = bias->contiguous();
        std::vector<int64_t> bShape = {cout};
        auto bStrides = ContiguousStrides(bShape);
        biasTensor = aclCreateTensor(bShape.data(), bShape.size(), ACL_FLOAT, bStrides.data(), 0, ACL_FORMAT_ND,
                                     bShape.data(), bShape.size(), biasC.data_ptr());
    }

    std::vector<int64_t> strideV = {stride3[0], stride3[1], stride3[2]};
    std::vector<int64_t> padV(pads.begin(), pads.end());
    std::vector<int64_t> dilV = {dilation3[0], dilation3[1], dilation3[2]};
    aclIntArray* strideArr = aclCreateIntArray(strideV.data(), strideV.size());
    aclIntArray* padArr = aclCreateIntArray(padV.data(), padV.size());
    aclIntArray* dilArr = aclCreateIntArray(dilV.data(), dilV.size());

    aclrtStream stream = c10_npu::getCurrentNPUStream().stream(false);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = GetWorkspaceSize(xTensor, wTensor, biasTensor, strideArr, padArr, dilArr, groups, enableHf32, yTensor,
                                &workspaceSize, &executor);
    TORCH_CHECK(ret == ACL_SUCCESS, "aclnnConv3dGeluCustomGetWorkspaceSize failed, ret=", static_cast<int>(ret));

    if (std::getenv("CONV3D_GELU_DEBUG") != nullptr) {
        fprintf(stderr, "[conv3d_gelu] workspaceSize = %.2f MB (out=%.2f MB)\n", workspaceSize / (1024.0 * 1024.0),
                (out.numel() * 4.0) / (1024.0 * 1024.0));
    }

    at::Tensor workspace;
    void* workspacePtr = nullptr;
    if (workspaceSize > 0) {
        workspace = at::empty({static_cast<int64_t>(workspaceSize)}, x.options().dtype(at::kByte));
        workspacePtr = workspace.data_ptr();
    }
    ret = aclnnConv3dGeluCustom(workspacePtr, workspaceSize, executor, stream);
    TORCH_CHECK(ret == ACL_SUCCESS, "aclnnConv3dGeluCustom failed, ret=", static_cast<int>(ret));
    // 不在此处同步。workspace / contiguous 临时张量由 torch_npu 缓存分配器按当前流回收，
    // 与原生 conv3d 一样异步返回，避免每次调用都把流水线打断。

    (void)aclDestroyTensor(xTensor);
    (void)aclDestroyTensor(wTensor);
    (void)aclDestroyTensor(yTensor);
    if (biasTensor != nullptr) {
        (void)aclDestroyTensor(biasTensor);
    }
    (void)aclDestroyIntArray(strideArr);
    (void)aclDestroyIntArray(padArr);
    (void)aclDestroyIntArray(dilArr);

    return out;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> conv3d_gelu_backward(
    const at::Tensor& gradOutput, const at::Tensor& x, const at::Tensor& weight, const c10::optional<at::Tensor>& bias,
    c10::IntArrayRef stride, c10::IntArrayRef padding, c10::IntArrayRef dilation, int64_t groups, bool needInput,
    bool needWeight, bool needBias, bool enableHf32)
{
    TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1, "x must be on NPU");
    TORCH_CHECK(
        gradOutput.scalar_type() == at::kFloat && x.scalar_type() == at::kFloat && weight.scalar_type() == at::kFloat,
        "Conv3DGeluGrad only supports float32");
    TORCH_CHECK(needInput || needWeight || needBias, "conv3d_gelu backward needs at least one gradient");

    auto stride3 = Normalize3(stride, 1);
    auto dilation3 = Normalize3(dilation, 1);
    auto pads = NormalizePads(padding);
    at::Tensor gradOutC = gradOutput.contiguous();
    at::Tensor xc = x.contiguous();
    at::Tensor wc = weight.contiguous();

    std::vector<int64_t> xShape(xc.sizes().begin(), xc.sizes().end());
    std::vector<int64_t> wShape(wc.sizes().begin(), wc.sizes().end());
    std::vector<int64_t> dyShape(gradOutC.sizes().begin(), gradOutC.sizes().end());
    at::Tensor dz = at::empty_like(gradOutC);
    at::Tensor gradX = needInput ? at::empty_like(xc) : at::Tensor();
    at::Tensor gradW = needWeight ? at::empty_like(wc) : at::Tensor();
    at::Tensor gradB;

    aclTensor* xTensor = CreateNcdhwTensor(xShape, ACL_FORMAT_NCDHW, xc.data_ptr());
    aclTensor* wTensor = CreateNcdhwTensor(wShape, ACL_FORMAT_NCDHW, wc.data_ptr());
    aclTensor* dyTensor = CreateNcdhwTensor(dyShape, ACL_FORMAT_NCDHW, gradOutC.data_ptr());
    aclTensor* dzTensor = CreateNcdhwTensor(dyShape, ACL_FORMAT_NCDHW, dz.data_ptr());
    aclTensor* gradXTensor = needInput ? CreateNcdhwTensor(xShape, ACL_FORMAT_NCDHW, gradX.data_ptr()) : nullptr;
    aclTensor* gradWTensor = needWeight ? CreateNcdhwTensor(wShape, ACL_FORMAT_NCDHW, gradW.data_ptr()) : nullptr;
    aclTensor* gradBTensor = nullptr;
    aclTensor* biasTensor = nullptr;
    at::Tensor biasC;
    std::vector<int64_t> biasSizes;
    if (bias.has_value() && bias->defined()) {
        biasC = bias->contiguous();
        biasSizes = {biasC.size(0)};
        auto bStrides = ContiguousStrides(biasSizes);
        biasTensor = aclCreateTensor(biasSizes.data(), biasSizes.size(), ACL_FLOAT, bStrides.data(), 0, ACL_FORMAT_ND,
                                     biasSizes.data(), biasSizes.size(), biasC.data_ptr());
        if (needBias) {
            gradB = at::empty_like(biasC);
            gradBTensor = aclCreateTensor(biasSizes.data(), biasSizes.size(), ACL_FLOAT, bStrides.data(), 0,
                                          ACL_FORMAT_ND, biasSizes.data(), biasSizes.size(), gradB.data_ptr());
        }
    }

    std::vector<int64_t> strideV = {stride3[0], stride3[1], stride3[2]};
    std::vector<int64_t> pad3 = {pads[0], pads[2], pads[4]};
    std::vector<int64_t> dilV = {dilation3[0], dilation3[1], dilation3[2]};
    std::vector<int64_t> outPad = {0, 0, 0};
    bool maskValue[3] = {needInput, needWeight, needBias && biasTensor != nullptr};
    aclIntArray* strideArr = aclCreateIntArray(strideV.data(), strideV.size());
    aclIntArray* padArr = aclCreateIntArray(pad3.data(), pad3.size());
    aclIntArray* dilArr = aclCreateIntArray(dilV.data(), dilV.size());
    aclIntArray* outPadArr = aclCreateIntArray(outPad.data(), outPad.size());
    aclIntArray* biasSizeArr = biasSizes.empty() ? nullptr : aclCreateIntArray(biasSizes.data(), biasSizes.size());
    aclBoolArray* mask = aclCreateBoolArray(maskValue, 3);

    aclrtStream stream = c10_npu::getCurrentNPUStream().stream(false);
    auto run = [&](uint64_t workspaceSize, aclOpExecutor* executor, auto launch) {
        at::Tensor workspace;
        void* workspacePtr = nullptr;
        if (workspaceSize > 0) {
            workspace = at::empty({static_cast<int64_t>(workspaceSize)}, x.options().dtype(at::kByte));
            workspacePtr = workspace.data_ptr();
        }
        launch(workspacePtr, workspaceSize, executor, stream);
    };

    at::Tensor z = at::empty_like(gradOutC);
    aclTensor* zTensor = CreateNcdhwTensor(dyShape, ACL_FORMAT_NCDHW, z.data_ptr());
    constexpr int8_t kCubeKeepDtype = 0;
    const int8_t cubeMath = enableHf32 ? static_cast<int8_t>(3) : kCubeKeepDtype;
    uint64_t convFwdWs = 0;
    aclOpExecutor* convFwdExec = nullptr;
    auto ret = aclnnConvolutionGetWorkspaceSize(xTensor, wTensor, biasTensor, strideArr, padArr, dilArr, false,
                                                outPadArr, groups, zTensor, cubeMath, &convFwdWs, &convFwdExec);
    TORCH_CHECK(ret == ACL_SUCCESS, "aclnnConvolutionGetWorkspaceSize failed, ret=", static_cast<int>(ret));
    run(convFwdWs, convFwdExec, aclnnConvolution);

    uint64_t geluWs = 0;
    aclOpExecutor* geluExec = nullptr;
    ret = aclnnConv3dGeluGradGetWorkspaceSize(zTensor, dyTensor, dzTensor, &geluWs, &geluExec);
    TORCH_CHECK(ret == ACL_SUCCESS, "aclnnConv3dGeluGradGetWorkspaceSize failed, ret=", static_cast<int>(ret));
    run(geluWs, geluExec, aclnnConv3dGeluGrad);
    uint64_t convWs = 0;
    aclOpExecutor* convExec = nullptr;
    ret = aclnnConvolutionBackwardGetWorkspaceSize(dzTensor, xTensor, wTensor, biasSizeArr, strideArr, padArr, dilArr,
                                                   false, outPadArr, static_cast<int>(groups), mask,
                                                   enableHf32 ? static_cast<int8_t>(3) : kCubeKeepDtype, gradXTensor,
                                                   gradWTensor, gradBTensor, &convWs, &convExec);
    TORCH_CHECK(ret == ACL_SUCCESS, "aclnnConvolutionBackwardGetWorkspaceSize failed, ret=", static_cast<int>(ret));
    run(convWs, convExec, aclnnConvolutionBackward);

    (void)aclDestroyTensor(xTensor);
    (void)aclDestroyTensor(wTensor);
    (void)aclDestroyTensor(dyTensor);
    (void)aclDestroyTensor(dzTensor);
    (void)aclDestroyTensor(zTensor);
    if (biasTensor != nullptr) {
        (void)aclDestroyTensor(biasTensor);
    }
    if (gradXTensor != nullptr) {
        (void)aclDestroyTensor(gradXTensor);
    }
    if (gradWTensor != nullptr) {
        (void)aclDestroyTensor(gradWTensor);
    }
    if (gradBTensor != nullptr) {
        (void)aclDestroyTensor(gradBTensor);
    }
    (void)aclDestroyIntArray(strideArr);
    (void)aclDestroyIntArray(padArr);
    (void)aclDestroyIntArray(dilArr);
    (void)aclDestroyIntArray(outPadArr);
    if (biasSizeArr != nullptr) {
        (void)aclDestroyIntArray(biasSizeArr);
    }
    (void)aclDestroyBoolArray(mask);
    return {gradX, gradW, gradB};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("conv3d_gelu_forward", &conv3d_gelu_forward, "Conv3D+GELU fused forward (cube conv + vector gelu)",
          pybind11::arg("x"), pybind11::arg("weight"), pybind11::arg("bias"), pybind11::arg("stride"),
          pybind11::arg("padding"), pybind11::arg("dilation"), pybind11::arg("groups"),
          pybind11::arg("enable_hf32") = false);
    m.def("conv3d_gelu_backward", &conv3d_gelu_backward,
          "Conv3D+GELU fused backward (recompute conv, gelu grad, conv grad)", pybind11::arg("grad_output"),
          pybind11::arg("x"), pybind11::arg("weight"), pybind11::arg("bias"), pybind11::arg("stride"),
          pybind11::arg("padding"), pybind11::arg("dilation"), pybind11::arg("groups"), pybind11::arg("need_input"),
          pybind11::arg("need_weight"), pybind11::arg("need_bias"), pybind11::arg("enable_hf32") = false);
}
