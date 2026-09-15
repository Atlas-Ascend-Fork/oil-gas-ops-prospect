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
#
# 对齐 ops-nn <op>/examples/test_aclnn_*.cpp 的独立编译。
# 调用方须先 project() 并设置 ACLNN_TARGET、ACLNN_EXAMPLE_SRC。

if(NOT ACLNN_TARGET)
    message(FATAL_ERROR "ACLNN_TARGET must be set before including aclnn_example.cmake")
endif()
if(NOT ACLNN_EXAMPLE_SRC)
    message(FATAL_ERROR "ACLNN_EXAMPLE_SRC must be set before including aclnn_example.cmake")
endif()

add_compile_options(-std=c++17)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/output")

set(INC_PATH "$ENV{DDK_PATH}")
if(INC_PATH STREQUAL "")
    if(DEFINED ENV{ASCEND_HOME_PATH} AND NOT "$ENV{ASCEND_HOME_PATH}" STREQUAL "")
        set(INC_PATH "$ENV{ASCEND_HOME_PATH}")
    else()
        set(INC_PATH "/usr/local/Ascend/ascend-toolkit/latest")
    endif()
    message(STATUS "set default INC_PATH: ${INC_PATH}")
else()
    message(STATUS "env INC_PATH: ${INC_PATH}")
endif()

set(CUST_PKG_PATH "${INC_PATH}/opp/vendors/customize/op_api")
set(LIB_PATH "$ENV{NPU_HOST_LIB}")
if(LIB_PATH STREQUAL "")
    string(TOLOWER "${CMAKE_SYSTEM_NAME}" SYSTEM_NAME_LOWER)
    set(LIB_PATH "${INC_PATH}/${CMAKE_SYSTEM_PROCESSOR}-${SYSTEM_NAME_LOWER}/lib64")
    message(STATUS "set default LIB_PATH: ${LIB_PATH}")
else()
    message(STATUS "env LIB_PATH: ${LIB_PATH}")
endif()

get_filename_component(ACLNN_EXAMPLE_DIR "${CMAKE_CURRENT_LIST_DIR}" ABSOLUTE)
get_filename_component(OG_REPO_ROOT "${ACLNN_EXAMPLE_DIR}/../.." ABSOLUTE)

include_directories(
    ${ACLNN_EXAMPLE_DIR}
    ${OG_REPO_ROOT}/ascendc/operators/_common/op_api
    ${INC_PATH}/include
    ${CUST_PKG_PATH}/include
)

link_directories(
    ${LIB_PATH}
    ${CUST_PKG_PATH}/lib
)

add_executable(${ACLNN_TARGET} ${ACLNN_EXAMPLE_SRC})

target_link_libraries(${ACLNN_TARGET}
    ascendcl
    cust_opapi
    acl_op_compiler
    nnopbase
    stdc++
)

install(TARGETS ${ACLNN_TARGET} DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
