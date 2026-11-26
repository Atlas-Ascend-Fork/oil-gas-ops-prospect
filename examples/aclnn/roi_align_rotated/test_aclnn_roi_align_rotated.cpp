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
 * 内联小 tensor 两段式 aclnnROIAlignRotated。
 *
 * 用例来自 detectron2 官方单测：5x5 arange 输入，box [1,1,3,3] -> 旋转框中心(2,2)、
 * 宽高 2、角度 0，output_size=4x4，sampling_ratio=0。期望值即 detectron2 单测的
 * 4x4 双线性上采样结果。
 */

#include "aclnn_example.h"
#include "og_aclnn_decls.h"

namespace {

constexpr int64_t kOutNumel = 16;
const std::vector<int64_t> kInputShape = {1, 1, 5, 5};
const std::vector<int64_t> kRoisShape = {1, 6};
const std::vector<int64_t> kOutShape = {1, 1, 4, 4};

const std::vector<float> kInput = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                                   13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
const std::vector<float> kRois = {0, 2, 2, 2, 2, 0};
const std::vector<float> kExpected = {4.5f, 5.0f,  5.5f,  6.0f,  7.0f,  7.5f,  8.0f,  8.5f,
                                      9.5f, 10.0f, 10.5f, 11.0f, 12.0f, 12.5f, 13.0f, 13.5f};

struct Tensors {
    void* inputDev = nullptr;
    void* roisDev = nullptr;
    void* outDev = nullptr;
    aclTensor* inputT = nullptr;
    aclTensor* roisT = nullptr;
    aclTensor* outT = nullptr;
};

int SetupTensors(Tensors& t)
{
    auto ret = CreateAclTensor(kInput, kInputShape, &t.inputDev, aclDataType::ACL_FLOAT, &t.inputT);
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
    auto ret = aclnnROIAlignRotatedGetWorkspaceSize(t.inputT, t.roisT, 1.0, 4, 4, 0, t.outT, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnROIAlignRotatedGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    ret = MallocWorkspace(workspaceSize, workspaceAddr);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = aclnnROIAlignRotated(*workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnROIAlignRotated failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    return ACL_SUCCESS;
}

std::vector<float> Expected(const Tensors&) { return kExpected; }

void Release(Tensors& t)
{
    DestroyTensors(&t.inputT, 1);
    DestroyTensors(&t.roisT, 1);
    DestroyTensors(&t.outT, 1);
    FreeDeviceBuffers(&t.inputDev, 1);
    FreeDeviceBuffers(&t.roisDev, 1);
    FreeDeviceBuffers(&t.outDev, 1);
}

} // namespace

int main()
{
    const AclnnExampleOps<Tensors> ops{SetupTensors, LaunchOp, &Tensors::outDev, Expected, Release};
    return RunAclnnExample(ops, kOutNumel);
}
