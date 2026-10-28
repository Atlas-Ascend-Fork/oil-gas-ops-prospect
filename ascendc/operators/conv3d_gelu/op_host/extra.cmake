# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# -----------------------------------------------------------------------------------------------------------

# [oil_gas_ops] conv3d_gelu 自定义 L2 aclnn 接入。
# 目标：对齐 native conv3d 的格式处理——提供一个直接接收 NCDHW 的 aclnn 接口
# (aclnnConv3dGeluCustom)，内部用 l0op::TransData 在同一 executor 内完成
# NCDHW<->私有格式 的转换并 launch Conv3DGelu。源码原位编译进 cust_opapi。
# 本文件由 merge 追加到 op_host/CMakeLists.txt 末尾（host 库目标已定义）。

get_filename_component(REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../.." ABSOLUTE)
set(CG_OPAPI_DIR "${REPO_ROOT}/ascendc/operators/conv3d_gelu/op_host/op_api")

# opdev / aclnn_kernels（l0op::TransData/Contiguous/ViewCopy、executor 宏）头文件根。
set(CG_ACLNN_INC
    ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/include
    ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/include/aclnn
)

set(CG_OPAPI_SRCS
    ${CG_OPAPI_DIR}/conv3d_gelu_l0.cpp
    ${CG_OPAPI_DIR}/aclnn_conv3d_gelu_custom.cpp
)

if(TARGET cust_opapi)
    target_sources(cust_opapi PRIVATE ${CG_OPAPI_SRCS})
    target_include_directories(cust_opapi BEFORE PRIVATE ${CG_OPAPI_DIR} ${CG_ACLNN_INC})

    # l0op::TransData/Contiguous/ViewCopy 在 libopapi.so；InferShape/executor 在 libnnopbase.so。
    # 显式链接，确保 cust_opapi 记录 DT_NEEDED，运行时能解析这些符号（不依赖调用方）。
    find_library(CG_OPAPI_LIB NAMES opapi
        PATHS ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/lib64
              ${ASCEND_CANN_PACKAGE_PATH}/opp/built-in/op_impl/ai_core/tbe/op_api/lib/linux/${CMAKE_SYSTEM_PROCESSOR}
        NO_DEFAULT_PATH)
    find_library(CG_NNOPBASE_LIB NAMES nnopbase
        PATHS ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/lib64
              ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/devlib
        NO_DEFAULT_PATH)
    if(CG_OPAPI_LIB)
        target_link_libraries(cust_opapi PRIVATE ${CG_OPAPI_LIB})
    endif()
    if(CG_NNOPBASE_LIB)
        target_link_libraries(cust_opapi PRIVATE ${CG_NNOPBASE_LIB})
    endif()
endif()
