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
constexpr size_t kXRank = 4;

int deviceId = kDeviceId;
aclrtStream stream = nullptr;
bool gIsDevice = false;
void* hostX = nullptr;
void* devX = nullptr;
void* hostB1 = nullptr;
void* devB1 = nullptr;
aclTensor* xT = nullptr;
aclTensor* yT = nullptr;
aclTensor* b1T = nullptr;
size_t bytes = 0;
size_t biasBytes = 0;
std::vector<int64_t> gShape{2, 4, 64, 64};
std::vector<int64_t> gBiasShape;

void ReleaseTensors()
{
    DestroyTensors(&xT, 1);
    DestroyTensors(&yT, 1);
    DestroyTensors(&b1T, 1);
    FreeHostDevice(hostX, devX, gIsDevice);
    FreeHostDevice(hostB1, devB1, gIsDevice);
    hostX = hostB1 = nullptr;
    devX = devB1 = nullptr;
}

/** bias1 的 numel 只接受两种：按 (B,H,1,S) 广播，或与 x 同尺寸。 */
bool ResolveBiasShape(int64_t numel)
{
    if (numel == gShape[0] * gShape[1] * gShape[3]) {
        gBiasShape = {gShape[0], gShape[1], 1, gShape[3]};
        return true;
    }
    if (numel == ShapeNumel(gShape)) {
        gBiasShape = gShape;
        return true;
    }
    ERROR_LOG("bias1 numel %ld matches neither broadcast nor full x", static_cast<long>(numel));
    return false;
}

bool LoadOptionalBias1()
{
    const char* path = "../input/input_bias1.bin";
    if (!FileExists(path)) {
        return true;
    }
    if (gShape.size() != kXRank) {
        ERROR_LOG("bias1 requires 4D x shape");
        return false;
    }
    size_t fileBytes = 0;
    if (!GetFileSize(path, &fileBytes) || fileBytes == 0 || fileBytes % sizeof(float) != 0) {
        ERROR_LOG("invalid bias1 file");
        return false;
    }
    if (!ResolveBiasShape(static_cast<int64_t>(fileBytes / sizeof(float)))) {
        return false;
    }
    biasBytes = fileBytes;
    if (!MallocHostDevice(biasBytes, gIsDevice, &hostB1, &devB1)) {
        return false;
    }
    if (!ReadFile(path, hostB1, biasBytes) || !HostToDevice(devB1, hostB1, biasBytes, gIsDevice)) {
        return false;
    }
    b1T = CreateNdTensor(gBiasShape, ACL_FLOAT, devB1);
    return b1T != nullptr;
}

bool ProcessInput()
{
    if (!ReadShapeFile("../input/shape.txt", gShape) || gShape.size() != kXRank) {
        ERROR_LOG("fused_bias_softmax expects 4D shape");
        return false;
    }
    bytes = TensorBytes(gShape, ACL_FLOAT);
    if (!MallocHostDevice(bytes, gIsDevice, &hostX, &devX)) {
        return false;
    }
    if (!ReadFile("../input/input_x.bin", hostX, bytes) || !HostToDevice(devX, hostX, bytes, gIsDevice)) {
        return false;
    }
    // 原地写回：y 与 x 共用同一块 device 缓冲
    xT = CreateNdTensor(gShape, ACL_FLOAT, devX);
    yT = CreateNdTensor(gShape, ACL_FLOAT, devX);
    if (!LoadOptionalBias1()) {
        return false;
    }
    INFO_LOG("Set input success.");
    return xT != nullptr && yT != nullptr;
}

bool ExecuteOp()
{
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnFusedBiasSoftmaxGetWorkspaceSize(xT, b1T, nullptr, yT, &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("aclnnFusedBiasSoftmaxGetWorkspaceSize failed. error code is %d", static_cast<int32_t>(ret));
        return false;
    }
    INFO_LOG("GetWorkspaceSize success, workspace size %lu", static_cast<unsigned long>(workspaceSize));
    return RunAclnnOp(aclnnFusedBiasSoftmax, workspaceSize, executor, stream, "aclnnFusedBiasSoftmax");
}

bool GetResult()
{
    if (!DeviceToHost(hostX, devX, bytes, gIsDevice)) {
        return false;
    }
    return WriteFile("../output/output_y.bin", hostX, bytes);
}

} // namespace

int main() { return RunStMain(deviceId, &stream, &gIsDevice, {ProcessInput, ExecuteOp, GetResult, ReleaseTensors}); }
