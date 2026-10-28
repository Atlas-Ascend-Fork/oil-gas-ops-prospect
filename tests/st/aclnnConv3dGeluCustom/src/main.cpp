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

#include <fstream>
#include <vector>

#include "common.h"
#include "og_aclnn_decls.h"

namespace {

constexpr int kDeviceId = 0;
constexpr size_t kRank = 5;
constexpr size_t kAttrCount = 10;

int deviceId = kDeviceId;
aclrtStream stream = nullptr;
bool gIsDevice = false;
void* hostX = nullptr;
void* devX = nullptr;
void* hostW = nullptr;
void* devW = nullptr;
void* hostB = nullptr;
void* devB = nullptr;
void* hostY = nullptr;
void* devY = nullptr;
aclTensor* xT = nullptr;
aclTensor* wT = nullptr;
aclTensor* bT = nullptr;
aclTensor* yT = nullptr;
aclIntArray* strideArr = nullptr;
aclIntArray* padArr = nullptr;
aclIntArray* dilArr = nullptr;
size_t xBytes = 0;
size_t wBytes = 0;
size_t yBytes = 0;
std::vector<int64_t> gX;
std::vector<int64_t> gW;
std::vector<int64_t> gY;
int64_t gStride[3] = {1, 1, 1};
int64_t gPad[3] = {0, 0, 0};
int64_t gDil[3] = {1, 1, 1};
int64_t gBias = 0;

std::vector<int64_t> ContiguousStrides(const std::vector<int64_t>& shape)
{
    std::vector<int64_t> strides(shape.size(), 1);
    for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i) {
        strides[static_cast<size_t>(i)] = strides[static_cast<size_t>(i + 1)] * shape[static_cast<size_t>(i + 1)];
    }
    return strides;
}

aclTensor* CreateNcdhw(const std::vector<int64_t>& shape, void* deviceData)
{
    const auto strides = ContiguousStrides(shape);
    return aclCreateTensor(shape.data(), shape.size(), ACL_FLOAT, strides.data(), 0, ACL_FORMAT_NCDHW, shape.data(),
                           shape.size(), deviceData);
}

void ReleaseTensors()
{
    DestroyTensors(&xT, 1);
    DestroyTensors(&wT, 1);
    DestroyTensors(&bT, 1);
    DestroyTensors(&yT, 1);
    if (strideArr != nullptr) {
        aclDestroyIntArray(strideArr);
        strideArr = nullptr;
    }
    if (padArr != nullptr) {
        aclDestroyIntArray(padArr);
        padArr = nullptr;
    }
    if (dilArr != nullptr) {
        aclDestroyIntArray(dilArr);
        dilArr = nullptr;
    }
    FreeHostDevice(hostX, devX, gIsDevice);
    FreeHostDevice(hostW, devW, gIsDevice);
    FreeHostDevice(hostB, devB, gIsDevice);
    FreeHostDevice(hostY, devY, gIsDevice);
    hostX = hostW = hostB = hostY = nullptr;
    devX = devW = devB = devY = nullptr;
}

bool LoadTensor(const char* path, const std::vector<int64_t>& shape, void** host, void** device, size_t* bytes,
                aclTensor** tensor, bool ncdhw)
{
    *bytes = TensorBytes(shape, ACL_FLOAT);
    if (!MallocHostDevice(*bytes, gIsDevice, host, device)) {
        return false;
    }
    if (!ReadFile(path, *host, *bytes) || !HostToDevice(*device, *host, *bytes, gIsDevice)) {
        return false;
    }
    if (ncdhw) {
        *tensor = CreateNcdhw(shape, *device);
    } else {
        *tensor = CreateNdTensor(shape, ACL_FLOAT, *device);
    }
    return *tensor != nullptr;
}

bool ReadAttr()
{
    std::ifstream file("../input/conv_attr.txt");
    if (!file.is_open()) {
        ERROR_LOG("open conv_attr.txt failed");
        return false;
    }
    int64_t attr[kAttrCount] = {};
    for (size_t i = 0; i < kAttrCount; ++i) {
        if (!(file >> attr[i])) {
            ERROR_LOG("conv_attr.txt needs %zu ints", kAttrCount);
            return false;
        }
    }
    for (int i = 0; i < 3; ++i) {
        gStride[i] = attr[i];
        gPad[i] = attr[3 + i];
        gDil[i] = attr[6 + i];
    }
    gBias = attr[9];
    return true;
}

bool ProcessInput()
{
    if (!ReadShapeFile("../input/shape_x.txt", gX) || gX.size() != kRank) {
        ERROR_LOG("shape_x.txt must be 5D");
        return false;
    }
    if (!ReadShapeFile("../input/shape_w.txt", gW) || gW.size() != kRank) {
        ERROR_LOG("shape_w.txt must be 5D");
        return false;
    }
    if (!ReadShapeFile("../input/shape_y.txt", gY) || gY.size() != kRank) {
        ERROR_LOG("shape_y.txt must be 5D");
        return false;
    }
    if (!ReadAttr()) {
        return false;
    }
    if (!LoadTensor("../input/input_x.bin", gX, &hostX, &devX, &xBytes, &xT, true)) {
        return false;
    }
    if (!LoadTensor("../input/input_weight.bin", gW, &hostW, &devW, &wBytes, &wT, true)) {
        return false;
    }
    if (gBias != 0) {
        std::vector<int64_t> bShape = {gW[0]};
        size_t bBytes = 0;
        if (!LoadTensor("../input/input_bias.bin", bShape, &hostB, &devB, &bBytes, &bT, false)) {
            return false;
        }
    }
    yBytes = TensorBytes(gY, ACL_FLOAT);
    if (!MallocHostDevice(yBytes, gIsDevice, &hostY, &devY)) {
        return false;
    }
    yT = CreateNcdhw(gY, devY);
    strideArr = aclCreateIntArray(gStride, 3);
    padArr = aclCreateIntArray(gPad, 3);
    dilArr = aclCreateIntArray(gDil, 3);
    INFO_LOG("Set input success.");
    return yT != nullptr && strideArr != nullptr && padArr != nullptr && dilArr != nullptr;
}

bool ExecuteOp()
{
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnConv3dGeluCustomGetWorkspaceSize(xT, wT, bT, strideArr, padArr, dilArr, 1, yT, &workspaceSize,
                                                     &executor);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("aclnnConv3dGeluCustomGetWorkspaceSize failed. error code is %d", static_cast<int32_t>(ret));
        return false;
    }
    INFO_LOG("GetWorkspaceSize success, workspace size %lu", static_cast<unsigned long>(workspaceSize));
    return RunAclnnOp(aclnnConv3dGeluCustom, workspaceSize, executor, stream, "aclnnConv3dGeluCustom");
}

bool GetResult()
{
    if (!DeviceToHost(hostY, devY, yBytes, gIsDevice)) {
        return false;
    }
    return WriteFile("../output/output_y.bin", hostY, yBytes);
}

} // namespace

int main() { return RunStMain(deviceId, &stream, &gIsDevice, {ProcessInput, ExecuteOp, GetResult, ReleaseTensors}); }
