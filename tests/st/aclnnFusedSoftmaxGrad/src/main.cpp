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

#include <vector>

#include "common.h"
#include "og_aclnn_decls.h"

namespace {

constexpr int kDeviceId = 0;
constexpr size_t kDimsRank = 5; // B H Q S D

int deviceId = kDeviceId;
aclrtStream stream = nullptr;
bool gIsDevice = false;
void* hostP = nullptr;
void* hostGo = nullptr;
void* hostV = nullptr;
void* devP = nullptr;
void* devGo = nullptr;
void* devV = nullptr;
aclTensor* pT = nullptr;
aclTensor* goT = nullptr;
aclTensor* vT = nullptr;
aclTensor* outT = nullptr;
size_t attnBytes = 0;
size_t goBytes = 0;
size_t valueBytes = 0;
std::vector<int64_t> gDims{2, 4, 64, 64, 64};

void ReleaseTensors()
{
    DestroyTensors(&pT, 1);
    DestroyTensors(&goT, 1);
    DestroyTensors(&vT, 1);
    DestroyTensors(&outT, 1);
    FreeHostDevice(hostP, devP, gIsDevice);
    FreeHostDevice(hostGo, devGo, gIsDevice);
    FreeHostDevice(hostV, devV, gIsDevice);
    hostP = hostGo = hostV = nullptr;
    devP = devGo = devV = nullptr;
}

/** shape.txt 给 4 维时按 D == S 补齐第 5 维。 */
bool ReadDims()
{
    if (!ReadShapeFile("../input/shape.txt", gDims)) {
        return false;
    }
    if (gDims.size() == kDimsRank - 1) {
        gDims.push_back(gDims[3]);
    }
    if (gDims.size() != kDimsRank) {
        ERROR_LOG("fused_softmax_grad expects B H Q S [D]");
        return false;
    }
    return true;
}

bool ProcessInput()
{
    if (!ReadDims()) {
        return false;
    }
    const std::vector<int64_t> attnShape{gDims[0], gDims[1], gDims[2], gDims[3]};
    const std::vector<int64_t> goShape{gDims[0], gDims[1], gDims[2], gDims[4]};
    const std::vector<int64_t> valueShape{gDims[0], gDims[1], gDims[3], gDims[4]};
    attnBytes = TensorBytes(attnShape, ACL_FLOAT);
    goBytes = TensorBytes(goShape, ACL_FLOAT);
    valueBytes = TensorBytes(valueShape, ACL_FLOAT);
    if (!MallocHostDevice(attnBytes, gIsDevice, &hostP, &devP) ||
        !MallocHostDevice(goBytes, gIsDevice, &hostGo, &devGo) ||
        !MallocHostDevice(valueBytes, gIsDevice, &hostV, &devV)) {
        return false;
    }
    if (!ReadFile("../input/input_probs.bin", hostP, attnBytes) ||
        !ReadFile("../input/input_go.bin", hostGo, goBytes) || !ReadFile("../input/input_v.bin", hostV, valueBytes)) {
        return false;
    }
    if (!HostToDevice(devP, hostP, attnBytes, gIsDevice) || !HostToDevice(devGo, hostGo, goBytes, gIsDevice) ||
        !HostToDevice(devV, hostV, valueBytes, gIsDevice)) {
        return false;
    }
    pT = CreateNdTensor(attnShape, ACL_FLOAT, devP);
    goT = CreateNdTensor(goShape, ACL_FLOAT, devGo);
    vT = CreateNdTensor(valueShape, ACL_FLOAT, devV);
    outT = CreateNdTensor(attnShape, ACL_FLOAT, devP); // 原地覆写 probs
    INFO_LOG("Set input success.");
    return pT != nullptr && goT != nullptr && vT != nullptr && outT != nullptr;
}

bool ExecuteOp()
{
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnFusedSoftmaxGradGetWorkspaceSize(pT, goT, vT, outT, &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("aclnnFusedSoftmaxGradGetWorkspaceSize failed. error code is %d", static_cast<int32_t>(ret));
        return false;
    }
    INFO_LOG("GetWorkspaceSize success, workspace size %lu", static_cast<unsigned long>(workspaceSize));
    return RunAclnnOp(aclnnFusedSoftmaxGrad, workspaceSize, executor, stream, "aclnnFusedSoftmaxGrad");
}

bool GetResult()
{
    if (!DeviceToHost(hostP, devP, attnBytes, gIsDevice)) {
        return false;
    }
    return WriteFile("../output/output_out.bin", hostP, attnBytes);
}

} // namespace

int main() { return RunStMain(deviceId, &stream, &gIsDevice, {ProcessInput, ExecuteOp, GetResult, ReleaseTensors}); }
