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
 * 内联小 tensor 两段式 aclnnROIAlignRotatedGrad。
 *
 * 与前向样例同一 ROI：grad_output 全 1 的 4x4，输入 5x5，box [1,1,3,3] 旋转 0 度。
 * 期望 grad_input 由仓内纯 torch 参考实现反向计算得到。
 */

#include "aclnn_example.h"
#include "og_aclnn_decls.h"

namespace {

constexpr int64_t kOutNumel = 25;
const std::vector<int64_t> kGradShape = {1, 1, 4, 4};
const std::vector<int64_t> kRoisShape = {1, 6};
const std::vector<int64_t> kOutShape = {1, 1, 5, 5};

const std::vector<float> kGrad = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
const std::vector<float> kRois = {0, 2, 2, 2, 2, 0};
const std::vector<float> kExpected = {
    0.0625f, 0.4375f, 0.4375f, 0.0625f, 0.0f,    0.4375f, 3.0625f, 3.0625f, 0.4375f, 0.0f, 0.4375f, 3.0625f, 3.0625f,
    0.4375f, 0.0f,    0.0625f, 0.4375f, 0.4375f, 0.0625f, 0.0f,    0.0f,    0.0f,    0.0f, 0.0f,    0.0f,
};

struct Tensors {
    void* gradDev = nullptr;
    void* roisDev = nullptr;
    void* outDev = nullptr;
    aclTensor* gradT = nullptr;
    aclTensor* roisT = nullptr;
    aclTensor* outT = nullptr;
};

int SetupTensors(Tensors& t)
{
    auto ret = CreateAclTensor(kGrad, kGradShape, &t.gradDev, aclDataType::ACL_FLOAT, &t.gradT);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(kRois, kRoisShape, &t.roisDev, aclDataType::ACL_FLOAT, &t.roisT);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    const std::vector<float> zeros(kOutNumel, 0.0f);
    ret = CreateAclTensor(zeros, kOutShape, &t.outDev, aclDataType::ACL_FLOAT, &t.outT);
    return ret;
}

int LaunchOp(const Tensors& t, aclrtStream stream, void** workspaceAddr)
{
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnROIAlignRotatedGradGetWorkspaceSize(t.gradT, t.roisT, 1.0, 4, 4, 0, 1, 1, 5, 5, t.outT,
                                                        &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnROIAlignRotatedGradGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    ret = MallocWorkspace(workspaceSize, workspaceAddr);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = aclnnROIAlignRotatedGrad(*workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnROIAlignRotatedGrad failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    return ACL_SUCCESS;
}

std::vector<float> Expected(const Tensors&) { return kExpected; }

void Release(Tensors& t)
{
    DestroyTensors(&t.gradT, 1);
    DestroyTensors(&t.roisT, 1);
    DestroyTensors(&t.outT, 1);
    FreeDeviceBuffers(&t.gradDev, 1);
    FreeDeviceBuffers(&t.roisDev, 1);
    FreeDeviceBuffers(&t.outDev, 1);
}

} // namespace

int main()
{
    const AclnnExampleOps<Tensors> ops{SetupTensors, LaunchOp, &Tensors::outDev, Expected, Release};
    return RunAclnnExample(ops, kOutNumel);
}
