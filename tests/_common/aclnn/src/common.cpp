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

#include "common.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <fstream>
#include <sstream>
#include <utility>

namespace {

std::vector<int64_t> ContiguousStrides(const std::vector<int64_t>& shape)
{
    std::vector<int64_t> strides(shape.size(), 1);
    for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i) {
        strides[static_cast<size_t>(i)] = strides[static_cast<size_t>(i + 1)] * shape[static_cast<size_t>(i + 1)];
    }
    return strides;
}

} // namespace

bool ReadFile(const std::string& filePath, void* buffer, size_t bufferSize)
{
    struct stat sBuf {};
    if (stat(filePath.data(), &sBuf) == -1 || S_ISREG(sBuf.st_mode) == 0) {
        ERROR_LOG("failed to get file %s", filePath.c_str());
        return false;
    }
    std::ifstream file(filePath, std::ios::binary);
    if (!file.is_open()) {
        ERROR_LOG("Open file failed. path = %s", filePath.c_str());
        return false;
    }
    file.seekg(0, std::ios::end);
    const std::streamoff size = file.tellg();
    if (size <= 0 || static_cast<size_t>(size) > bufferSize) {
        ERROR_LOG("invalid file size for %s", filePath.c_str());
        return false;
    }
    file.seekg(0, std::ios::beg);
    file.read(static_cast<char*>(buffer), static_cast<std::streamsize>(size));
    return static_cast<bool>(file);
}

bool FileExists(const std::string& filePath)
{
    struct stat sBuf {};
    return stat(filePath.c_str(), &sBuf) == 0 && S_ISREG(sBuf.st_mode) != 0;
}

bool GetFileSize(const std::string& filePath, size_t* size)
{
    if (size == nullptr) {
        return false;
    }
    struct stat sBuf {};
    if (stat(filePath.c_str(), &sBuf) == -1 || S_ISREG(sBuf.st_mode) == 0 || sBuf.st_size < 0) {
        return false;
    }
    *size = static_cast<size_t>(sBuf.st_size);
    return true;
}

bool ReadShapeFile(const std::string& filePath, std::vector<int64_t>& shape)
{
    if (!FileExists(filePath)) {
        return true;
    }
    std::ifstream file(filePath);
    if (!file.is_open()) {
        ERROR_LOG("Open shape file failed. path = %s", filePath.c_str());
        return false;
    }
    std::vector<int64_t> parsed;
    std::string token;
    while (file >> token) {
        for (char& c : token) {
            if (c == ',' || c == 'x' || c == 'X') {
                c = ' ';
            }
        }
        std::istringstream inner(token);
        int64_t dim = 0;
        while (inner >> dim) {
            if (dim <= 0) {
                ERROR_LOG("invalid dim in %s", filePath.c_str());
                return false;
            }
            parsed.push_back(dim);
        }
    }
    if (parsed.empty()) {
        ERROR_LOG("empty shape file %s", filePath.c_str());
        return false;
    }
    shape = std::move(parsed);
    INFO_LOG("Read shape file %s, rank=%zu", filePath.c_str(), shape.size());
    return true;
}

bool WriteFile(const std::string& filePath, const void* buffer, size_t size)
{
    if (buffer == nullptr) {
        ERROR_LOG("Write file failed. buffer is nullptr");
        return false;
    }
    const int fd = open(filePath.c_str(), O_RDWR | O_CREAT | O_TRUNC, S_IRUSR | S_IWRITE);
    if (fd < 0) {
        ERROR_LOG("Open file failed. path = %s", filePath.c_str());
        return false;
    }
    const ssize_t writeSize = write(fd, buffer, size);
    (void)close(fd);
    if (static_cast<size_t>(writeSize) != size) {
        ERROR_LOG("Write file Failed.");
        return false;
    }
    return true;
}

int64_t ShapeNumel(const std::vector<int64_t>& shape)
{
    int64_t numel = 1;
    for (int64_t dim : shape) {
        numel *= dim;
    }
    return numel;
}

size_t TensorBytes(const std::vector<int64_t>& shape, aclDataType dtype)
{
    return static_cast<size_t>(ShapeNumel(shape)) * aclDataTypeSize(dtype);
}

bool InitResource(int deviceId, aclrtStream* stream, bool* isDevice)
{
    const aclError initRet = aclInit(nullptr);
    if (initRet != ACL_SUCCESS) {
        ERROR_LOG("acl init failed, ret=%d", static_cast<int>(initRet));
        return false;
    }
    if (aclrtSetDevice(deviceId) != ACL_SUCCESS) {
        ERROR_LOG("Set device failed. deviceId is %d", deviceId);
        (void)aclFinalize();
        return false;
    }
    if (aclrtCreateStream(stream) != ACL_SUCCESS) {
        ERROR_LOG("Create stream failed");
        (void)aclrtResetDevice(deviceId);
        (void)aclFinalize();
        return false;
    }
    aclrtRunMode runMode;
    if (aclrtGetRunMode(&runMode) != ACL_SUCCESS) {
        ERROR_LOG("Get run mode failed");
        DestroyResource(deviceId, *stream);
        return false;
    }
    *isDevice = (runMode == ACL_DEVICE);
    INFO_LOG("Init resource success, device=%d, runMode=%d", deviceId, static_cast<int>(runMode));
    return true;
}

void DestroyResource(int deviceId, aclrtStream stream)
{
    if (stream != nullptr && aclrtDestroyStream(stream) != ACL_SUCCESS) {
        ERROR_LOG("Destroy stream failed");
    }
    if (aclrtResetDevice(deviceId) != ACL_SUCCESS) {
        ERROR_LOG("Reset device %d failed", deviceId);
    }
    if (aclFinalize() != ACL_SUCCESS) {
        ERROR_LOG("Finalize acl failed");
    }
}

bool MallocHostDevice(size_t bytes, bool isDevice, void** host, void** device)
{
    if (isDevice) {
        if (aclrtMalloc(host, bytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
            ERROR_LOG("Malloc host-side device memory failed");
            return false;
        }
    } else if (aclrtMallocHost(host, bytes) != ACL_SUCCESS) {
        ERROR_LOG("Malloc host memory failed");
        return false;
    }
    if (aclrtMalloc(device, bytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
        ERROR_LOG("Malloc device memory failed");
        FreeHostDevice(*host, nullptr, isDevice);
        *host = nullptr;
        return false;
    }
    return true;
}

void FreeHostDevice(void* host, void* device, bool isDevice)
{
    if (device != nullptr) {
        (void)aclrtFree(device);
    }
    if (host == nullptr) {
        return;
    }
    if (isDevice) {
        (void)aclrtFree(host);
    } else {
        (void)aclrtFreeHost(host);
    }
}

aclTensor* CreateNdTensor(const std::vector<int64_t>& shape, aclDataType dtype, void* deviceData, int64_t storageNumel)
{
    const auto strides = ContiguousStrides(shape);
    const int64_t numel = storageNumel > 0 ? storageNumel : ShapeNumel(shape);
    std::vector<int64_t> storageDims = {numel};
    return aclCreateTensor(shape.data(), shape.size(), dtype, strides.data(), 0, ACL_FORMAT_ND, storageDims.data(), 1,
                           deviceData);
}

void DestroyTensors(aclTensor** tensors, size_t count)
{
    if (tensors == nullptr) {
        return;
    }
    for (size_t i = 0; i < count; ++i) {
        if (tensors[i] != nullptr) {
            (void)aclDestroyTensor(tensors[i]);
            tensors[i] = nullptr;
        }
    }
}

bool HostToDevice(void* device, const void* host, size_t bytes, bool isDevice)
{
    const aclrtMemcpyKind kind = isDevice ? ACL_MEMCPY_DEVICE_TO_DEVICE : ACL_MEMCPY_HOST_TO_DEVICE;
    if (aclrtMemcpy(device, bytes, host, bytes, kind) != ACL_SUCCESS) {
        ERROR_LOG("Copy host to device failed");
        return false;
    }
    return true;
}

bool DeviceToHost(void* host, const void* device, size_t bytes, bool isDevice)
{
    const aclrtMemcpyKind kind = isDevice ? ACL_MEMCPY_DEVICE_TO_DEVICE : ACL_MEMCPY_DEVICE_TO_HOST;
    if (aclrtMemcpy(host, bytes, device, bytes, kind) != ACL_SUCCESS) {
        ERROR_LOG("Copy device to host failed");
        return false;
    }
    return true;
}

bool PrepareStInputs(const char* const* files, size_t count, std::vector<int64_t>& shape, size_t* oneBytes,
                     bool isDevice, void** hosts, void** devices, aclTensor** tensors, size_t packFactor,
                     void** packHost, void** packDevice)
{
    if (!ReadShapeFile("../input/shape.txt", shape) || shape.empty()) {
        return false;
    }
    const size_t bytes = TensorBytes(shape, ACL_FLOAT);
    *oneBytes = bytes;
    for (size_t i = 0; i < count; ++i) {
        if (!MallocHostDevice(bytes, isDevice, &hosts[i], &devices[i])) {
            return false;
        }
        if (!ReadFile(files[i], hosts[i], bytes) || !HostToDevice(devices[i], hosts[i], bytes, isDevice)) {
            return false;
        }
        tensors[i] = CreateNdTensor(shape, ACL_FLOAT, devices[i]);
        if (tensors[i] == nullptr) {
            ERROR_LOG("Create input tensor %zu failed", i);
            return false;
        }
    }
    return MallocHostDevice(bytes * packFactor, isDevice, packHost, packDevice);
}

void ReleaseStBuffers(void** hosts, void** devices, size_t count, void** packHost, void** packDevice, bool isDevice)
{
    for (size_t i = 0; i < count; ++i) {
        FreeHostDevice(hosts[i], devices[i], isDevice);
        hosts[i] = nullptr;
        devices[i] = nullptr;
    }
    FreeHostDevice(*packHost, *packDevice, isDevice);
    *packHost = nullptr;
    *packDevice = nullptr;
}

int RunStMain(int deviceId, aclrtStream* stream, bool* isDevice, const StStages& stages)
{
    if (!InitResource(deviceId, stream, isDevice)) {
        return FAILED;
    }
    int rc = SUCCESS;
    if (!stages.processInput() || !stages.executeOp() || !stages.getResult()) {
        rc = FAILED;
    } else {
        INFO_LOG("Write output success.");
    }
    stages.release();
    DestroyResource(deviceId, *stream);
    return rc;
}

bool RunAclnnOp(aclnnStatus (*run)(void*, uint64_t, aclOpExecutor*, aclrtStream), uint64_t workspaceSize,
                aclOpExecutor* executor, aclrtStream stream, const char* opName)
{
    void* workspace = nullptr;
    if (workspaceSize != 0 && aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
        ERROR_LOG("Malloc workspace failed");
        return false;
    }
    const auto ret = run(workspace, workspaceSize, executor, stream);
    if (workspace != nullptr) {
        (void)aclrtFree(workspace);
    }
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("%s execute failed. error code is %d", opName, static_cast<int32_t>(ret));
        return false;
    }
    if (aclrtSynchronizeStreamWithTimeout(stream, 5000) != ACL_SUCCESS) {
        ERROR_LOG("Synchronize stream failed");
        return false;
    }
    INFO_LOG("%s execute success", opName);
    return true;
}
