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

# Conv3DGeluGrad 的 L2 aclnn 接进 cust_opapi。
# 本文件由 merge 追加到 op_host/CMakeLists.txt 末尾。

get_filename_component(REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../.." ABSOLUTE)
set(CGG_OPAPI_DIR "${REPO_ROOT}/ascendc/operators/conv3d_gelu_grad/op_host/op_api")
set(CGG_ACLNN_INC
    ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/include
    ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/include/aclnn
)
set(CGG_OPAPI_SRCS
    ${CGG_OPAPI_DIR}/conv3d_gelu_grad_l0.cpp
    ${CGG_OPAPI_DIR}/aclnn_conv3d_gelu_grad.cpp
)

if(TARGET cust_opapi)
    target_sources(cust_opapi PRIVATE ${CGG_OPAPI_SRCS})
    target_include_directories(cust_opapi BEFORE PRIVATE ${CGG_OPAPI_DIR} ${CGG_ACLNN_INC})
    find_library(CGG_OPAPI_LIB NAMES opapi
        PATHS ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/lib64
              ${ASCEND_CANN_PACKAGE_PATH}/opp/built-in/op_impl/ai_core/tbe/op_api/lib/linux/${CMAKE_SYSTEM_PROCESSOR}
        NO_DEFAULT_PATH)
    find_library(CGG_NNOPBASE_LIB NAMES nnopbase
        PATHS ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/lib64
              ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/devlib
        NO_DEFAULT_PATH)
    if(CGG_OPAPI_LIB)
        target_link_libraries(cust_opapi PRIVATE ${CGG_OPAPI_LIB})
    endif()
    if(CGG_NNOPBASE_LIB)
        target_link_libraries(cust_opapi PRIVATE ${CGG_NNOPBASE_LIB})
    endif()
endif()
