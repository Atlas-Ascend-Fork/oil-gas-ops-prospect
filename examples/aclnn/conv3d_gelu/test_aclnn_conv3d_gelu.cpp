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
 * 两段式 aclnnConv3dGeluCustom 小样例：1×1、stride=1、无 padding 的 NCDHW conv + tanh GELU。
 */

#include <cmath>
#include <vector>

#include "aclnn_example.h"
#include "og_aclnn_decls.h"

namespace {

constexpr int64_t kN = 1;
constexpr int64_t kCin = 4;
constexpr int64_t kCout = 2;
constexpr int64_t kSpatial = 2;
constexpr size_t kOutNumel = static_cast<size_t>(kN * kCout * kSpatial * kSpatial * kSpatial);
constexpr float kGeluScale = 1.5957691216057308f;
constexpr float kGeluCube = 0.044715f;

const std::vector<int64_t> kXShape = {kN, kCin, kSpatial, kSpatial, kSpatial};
const std::vector<int64_t> kWShape = {kCout, kCin, 1, 1, 1};
const std::vector<int64_t> kYShape = {kN, kCout, kSpatial, kSpatial, kSpatial};
const std::vector<int64_t> kBShape = {kCout};

std::vector<int64_t> StridesOf(const std::vector<int64_t>& shape) { return ContiguousStrides(shape); }

aclTensor* MakeNcdhw(const std::vector<int64_t>& shape, void* deviceAddr)
{
    const auto strides = StridesOf(shape);
    return aclCreateTensor(shape.data(), shape.size(), ACL_FLOAT, strides.data(), 0, ACL_FORMAT_NCDHW, shape.data(),
                           shape.size(), deviceAddr);
}

struct Tensors {
    std::vector<float> x;
    std::vector<float> weight;
    std::vector<float> bias;
    void* xDev = nullptr;
    void* wDev = nullptr;
    void* bDev = nullptr;
    void* yDev = nullptr;
    aclTensor* xT = nullptr;
    aclTensor* wT = nullptr;
    aclTensor* bT = nullptr;
    aclTensor* yT = nullptr;
    aclIntArray* stride = nullptr;
    aclIntArray* pad = nullptr;
    aclIntArray* dil = nullptr;

    Tensors()
        : x(static_cast<size_t>(kN * kCin * kSpatial * kSpatial * kSpatial), 0.5f),
          weight(static_cast<size_t>(kCout * kCin), 0.25f),
          bias(static_cast<size_t>(kCout), 0.1f)
    {}
};

int SetupTensors(Tensors& t)
{
    auto ret = CreateAclTensor(t.x, kXShape, &t.xDev, aclDataType::ACL_FLOAT, &t.xT);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(t.weight, kWShape, &t.wDev, aclDataType::ACL_FLOAT, &t.wT);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(t.bias, kBShape, &t.bDev, aclDataType::ACL_FLOAT, &t.bT);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // CreateAclTensor 建的是 ND。conv 入口要 NCDHW，销毁后按 NCDHW 重建 view。
    DestroyTensors(&t.xT, 1);
    DestroyTensors(&t.wT, 1);
    t.xT = MakeNcdhw(kXShape, t.xDev);
    t.wT = MakeNcdhw(kWShape, t.wDev);
    CHECK_RET(t.xT != nullptr && t.wT != nullptr, return 1);
    std::vector<float> y(kOutNumel, 0.0f);
    ret = CreateAclTensor(y, kYShape, &t.yDev, aclDataType::ACL_FLOAT, &t.yT);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    DestroyTensors(&t.yT, 1);
    t.yT = MakeNcdhw(kYShape, t.yDev);
    const int64_t ones[3] = {1, 1, 1};
    const int64_t zeros[3] = {0, 0, 0};
    t.stride = aclCreateIntArray(ones, 3);
    t.pad = aclCreateIntArray(zeros, 3);
    t.dil = aclCreateIntArray(ones, 3);
    CHECK_RET(t.yT != nullptr && t.stride != nullptr && t.pad != nullptr && t.dil != nullptr, return 1);
    return ACL_SUCCESS;
}

int LaunchOp(const Tensors& t, aclrtStream stream, void** workspaceAddr)
{
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnConv3dGeluCustomGetWorkspaceSize(t.xT, t.wT, t.bT, t.stride, t.pad, t.dil, 1, t.yT, &workspaceSize,
                                                     &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnConv3dGeluCustomGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    ret = MallocWorkspace(workspaceSize, workspaceAddr);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = aclnnConv3dGeluCustom(*workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnConv3dGeluCustom failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    return ACL_SUCCESS;
}

float GeluTanh(float v)
{
    const float cube = v;
    const float inner = kGeluScale * (cube + kGeluCube * cube * cube * cube);
    return cube / (1.0f + std::exp(-inner));
}

std::vector<float> Expected(const Tensors&)
{
    // 每个输出是 Cin 个 0.5*0.25 再加上 bias 0.1。
    const float conv = static_cast<float>(kCin) * 0.5f * 0.25f + 0.1f;
    return std::vector<float>(kOutNumel, GeluTanh(conv));
}

void Release(Tensors& t)
{
    DestroyTensors(&t.xT, 1);
    DestroyTensors(&t.wT, 1);
    DestroyTensors(&t.bT, 1);
    DestroyTensors(&t.yT, 1);
    if (t.stride != nullptr) {
        aclDestroyIntArray(t.stride);
    }
    if (t.pad != nullptr) {
        aclDestroyIntArray(t.pad);
    }
    if (t.dil != nullptr) {
        aclDestroyIntArray(t.dil);
    }
    void* bufs[] = {t.xDev, t.wDev, t.bDev, t.yDev};
    FreeDeviceBuffers(bufs, 4);
}

} // namespace

int main()
{
    const AclnnExampleOps<Tensors> ops{SetupTensors, LaunchOp, &Tensors::yDev, Expected, Release};
    return RunAclnnExample(ops, kOutNumel, 1e-4f, 1e-4f);
}
