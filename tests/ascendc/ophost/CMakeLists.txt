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

# ophost 单元测试（googletest）汇总入口。
#
# 入口: bash build.sh -u  或  bash build.sh -u --ophost  或  cd ascendc && bash build.sh -u --ophost
#
# 用例源码在仓库根 tests/ut/op_host/<op>/test_*_tiling.cpp。
# 本文件只做 include / 链接 / ctest 注册，无卡可跑。
#
# 不覆盖 <Op>TilingFunc / InferShape 本身：它们的签名依赖 gert::TilingContext，
# CANN 二进制包未提供 gert faker。算子把切分算法抽到 *_tiling_core.h 后，
# UT 直接驱动核心函数。
#
# 新增算子：在 tests/ut/op_host/<op>/ 放下 test_*.cpp，并保证 operators/<op>/op_host 存在。

cmake_minimum_required(VERSION 3.16)

get_filename_component(REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(oil_gas_ophost_ut CXX)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    if(NOT CMAKE_BUILD_TYPE)
        set(CMAKE_BUILD_TYPE Release)
    endif()
    list(APPEND CMAKE_MODULE_PATH "${REPO_ROOT}/cmake")
    enable_testing()
endif()

set(OPERATORS_DIR "${REPO_ROOT}/ascendc/operators")

include(third_party/googletest)
include(security_flags)

set(OPHOST_UT_INCLUDE_DIRS "${OPERATORS_DIR}/_common/op_host")
file(GLOB _op_dirs LIST_DIRECTORIES true "${OPERATORS_DIR}/*")
foreach(_dir ${_op_dirs})
    if(IS_DIRECTORY "${_dir}/op_host")
        list(APPEND OPHOST_UT_INCLUDE_DIRS "${_dir}/op_host")
    endif()
endforeach()
foreach(_dir ${OPHOST_UT_INCLUDE_DIRS})
    if(NOT IS_DIRECTORY "${_dir}")
        message(FATAL_ERROR "missing operator op_host dir: ${_dir}")
    endif()
endforeach()

file(GLOB OPHOST_UT_SOURCES CONFIGURE_DEPENDS
    "${REPO_ROOT}/tests/ut/op_host/*/test_*.cpp")
if(NOT OPHOST_UT_SOURCES)
    message(FATAL_ERROR "no ophost UT sources under ${REPO_ROOT}/tests/ut/op_host/")
endif()

add_executable(ophost_ut ${OPHOST_UT_SOURCES})
target_include_directories(ophost_ut PRIVATE ${OPHOST_UT_INCLUDE_DIRS})
target_link_libraries(ophost_ut PRIVATE ${OIL_GAS_GTEST_TARGETS})
oil_gas_apply_security_flags(ophost_ut)

add_test(NAME ophost_ut COMMAND ophost_ut)
