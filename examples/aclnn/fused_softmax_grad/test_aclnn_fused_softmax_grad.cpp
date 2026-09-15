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
 * 两段式 aclnnFusedSoftmaxGrad 小样例：probs [1,1,2,2]、go [1,1,2,2]、values [1,1,2,2]（Q=S=D=2），
 * 结果原地覆写 probs。
 */

#include <cmath>

#include "aclnn_example.h"
#include "og_aclnn_decls.h"

namespace {

constexpr size_t kNumel = 4;
constexpr size_t kCols = 2;

const std::vector<int64_t> kAttnShape = {1, 1, 2, 2};
const std::vector<int64_t> kGoShape = {1, 1, 2, 2};
const std::vector<int64_t> kValueShape = {1, 1, 2, 2};
const std::vector<float> kLogits = {0.0f, 1.0f, 1.0f, 0.0f};
const std::vector<float> kGo = {0.1f, -0.2f, 0.3f, 0.4f};
const std::vector<float> kValues = {1.0f, 0.0f, 0.0f, 1.0f}; // 2x2 单位阵，故 g = go @ V^T = go

std::vector<float> SoftmaxRows()
{
    std::vector<float> probs(kNumel, 0.0f);
    for (size_t row = 0; row < kNumel / kCols; ++row) {
        const size_t base = row * kCols;
        const float a = kLogits[base];
        const float b = kLogits[base + 1];
        const float m = a > b ? a : b;
        const float ea = std::exp(a - m);
        const float eb = std::exp(b - m);
        const float s = ea + eb;
        probs[base] = ea / s;
        probs[base + 1] = eb / s;
    }
    return probs;
}

struct Tensors {
    std::vector<float> probs = SoftmaxRows();
    void* pDev = nullptr;
    void* goDev = nullptr;
    void* vDev = nullptr;
    aclTensor* pT = nullptr;
    aclTensor* goT = nullptr;
    aclTensor* vT = nullptr;
    aclTensor* outT = nullptr;
};

int SetupTensors(Tensors& t)
{
    auto ret = CreateAclTensor(t.probs, kAttnShape, &t.pDev, aclDataType::ACL_FLOAT, &t.pT);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(kGo, kGoShape, &t.goDev, aclDataType::ACL_FLOAT, &t.goT);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(kValues, kValueShape, &t.vDev, aclDataType::ACL_FLOAT, &t.vT);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 原地覆写：out 复用 probs 的 device 缓冲
    return CreateAclTensorView(t.pDev, kAttnShape, aclDataType::ACL_FLOAT, &t.outT);
}

int LaunchOp(const Tensors& t, aclrtStream stream, void** workspaceAddr)
{
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnFusedSoftmaxGradGetWorkspaceSize(t.pT, t.goT, t.vT, t.outT, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFusedSoftmaxGradGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    ret = MallocWorkspace(workspaceSize, workspaceAddr);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = aclnnFusedSoftmaxGrad(*workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFusedSoftmaxGrad failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    return ACL_SUCCESS;
}

std::vector<float> Expected(const Tensors& t)
{
    const std::vector<float>& probs = t.probs;
    std::vector<float> expected(kNumel, 0.0f);
    for (size_t row = 0; row < kNumel / kCols; ++row) {
        const size_t base = row * kCols;
        float dot = 0.0f;
        for (size_t c = 0; c < kCols; ++c) {
            dot += kGo[base + c] * probs[base + c];
        }
        for (size_t c = 0; c < kCols; ++c) {
            expected[base + c] = probs[base + c] * (kGo[base + c] - dot);
        }
    }
    return expected;
}

void Release(Tensors& t)
{
    DestroyTensors(&t.pT, 1);
    DestroyTensors(&t.goT, 1);
    DestroyTensors(&t.vT, 1);
    DestroyTensors(&t.outT, 1);
    FreeDeviceBuffers(&t.pDev, 1);
    FreeDeviceBuffers(&t.goDev, 1);
    FreeDeviceBuffers(&t.vDev, 1);
}

} // namespace

int main()
{
    const AclnnExampleOps<Tensors> ops{SetupTensors, LaunchOp, &Tensors::pDev, Expected, Release};
    return RunAclnnExample(ops, kNumel);
}
