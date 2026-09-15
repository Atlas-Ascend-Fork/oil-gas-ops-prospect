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

#ifndef OG_ACLNN_TEST_COMMON_H
#define OG_ACLNN_TEST_COMMON_H

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "aclnn/acl_meta.h"

#define SUCCESS 0
#define FAILED 1

#define INFO_LOG(fmt, args...) fprintf(stdout, "[INFO]  " fmt "\n", ##args)
#define ERROR_LOG(fmt, args...) fprintf(stderr, "[ERROR]  " fmt "\n", ##args)

bool ReadFile(const std::string& filePath, void* buffer, size_t bufferSize);
bool WriteFile(const std::string& filePath, const void* buffer, size_t size);
bool FileExists(const std::string& filePath);
bool GetFileSize(const std::string& filePath, size_t* size);
/** 读空格/逗号分隔的 shape 文件。文件不存在时保留 in/out 的 shape 并返回 true。 */
bool ReadShapeFile(const std::string& filePath, std::vector<int64_t>& shape);

int64_t ShapeNumel(const std::vector<int64_t>& shape);
size_t TensorBytes(const std::vector<int64_t>& shape, aclDataType dtype);

bool InitResource(int deviceId, aclrtStream* stream, bool* isDevice);
void DestroyResource(int deviceId, aclrtStream stream);

bool MallocHostDevice(size_t bytes, bool isDevice, void** host, void** device);
void FreeHostDevice(void* host, void* device, bool isDevice);

aclTensor* CreateNdTensor(const std::vector<int64_t>& shape, aclDataType dtype, void* deviceData,
                          int64_t storageNumel = -1);

/** 销毁并置空 tensors[0, count)，重复调用安全。 */
void DestroyTensors(aclTensor** tensors, size_t count);

bool HostToDevice(void* device, const void* host, size_t bytes, bool isDevice);
bool DeviceToHost(void* host, const void* device, size_t bytes, bool isDevice);

/**
 * 输出打包型算子的通用前处理：读 ../input/shape.txt 到 shape，按 files 顺序载入 count 个同 shape 的
 * float32 输入（分配 host/device、读文件、拷到 device、建 ND tensor），再分配 packFactor 份单张量大小的
 * 打包输出缓冲。*oneBytes 回填单张量字节数。任一步失败返回 false，已分配的缓冲交给 ReleaseStBuffers 回收。
 */
bool PrepareStInputs(const char* const* files, size_t count, std::vector<int64_t>& shape, size_t* oneBytes,
                     bool isDevice, void** hosts, void** devices, aclTensor** tensors, size_t packFactor,
                     void** packHost, void** packDevice);

/** 与 PrepareStInputs 配对，释放并置空全部输入缓冲与打包输出缓冲。 */
void ReleaseStBuffers(void** hosts, void** devices, size_t count, void** packHost, void** packDevice, bool isDevice);

bool RunAclnnOp(aclnnStatus (*run)(void*, uint64_t, aclOpExecutor*, aclrtStream), uint64_t workspaceSize,
                aclOpExecutor* executor, aclrtStream stream, const char* opName);

/** 各算子 ST 的三段式主流程，由 RunStMain 依次调用。 */
struct StStages {
    bool (*processInput)();
    bool (*executeOp)();
    bool (*getResult)();
    void (*release)();
};

/** ST main 的固定骨架：起资源 -> 跑三段 -> 释放 tensor/缓冲 -> 收资源。 */
int RunStMain(int deviceId, aclrtStream* stream, bool* isDevice, const StStages& stages);

#endif
