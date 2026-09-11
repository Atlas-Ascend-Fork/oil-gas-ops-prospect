# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# ophost 单元测试（googletest）。
#
# 入口: cd ascendc && bash build.sh -u --ophost
#
# 只编译 host 侧、与 gert / CANN 运行时无关的切分算法，因此**无卡环境即可跑**：
#   operators/<op>/op_host/*_tiling_core.h  —  平台无关的 tiling 算法（header-only）
#
# 不覆盖 <Op>TilingFunc / InferShape 本身：它们的签名依赖 gert::TilingContext，而
# CANN 二进制包未提供 gert faker（该基建在 metadef 源码仓）。算子把切分算法抽到
# *_tiling_core.h 后，UT 直接驱动核心函数，覆盖的是真实算法。
#
# 新增算子时：把它的 *_tiling_core.h 所在 op_host 目录加进 OPHOST_UT_INCLUDE_DIRS，
# 并把测试源加进 ophost_ut 的 sources。

cmake_minimum_required(VERSION 3.16)
project(oil_gas_ophost_ut CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

get_filename_component(ASCENDC_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
set(OPERATORS_DIR "${ASCENDC_ROOT}/operators")

# ---- googletest ----
# 优先用环境里已装的 gtest（容器内为 /opt/conda）；找不到再回落到 FetchContent。
find_package(GTest QUIET)
if(GTest_FOUND)
    set(_GTEST_TARGETS GTest::gtest GTest::gtest_main)
    message(STATUS "googletest: system (find_package)")
else()
    find_path(GTEST_INCLUDE_DIR gtest/gtest.h)
    find_library(GTEST_LIB NAMES gtest)
    find_library(GTEST_MAIN_LIB NAMES gtest_main)
    if(GTEST_INCLUDE_DIR AND GTEST_LIB AND GTEST_MAIN_LIB)
        add_library(ut_gtest INTERFACE)
        target_include_directories(ut_gtest INTERFACE ${GTEST_INCLUDE_DIR})
        target_link_libraries(ut_gtest INTERFACE ${GTEST_LIB} ${GTEST_MAIN_LIB} pthread)
        set(_GTEST_TARGETS ut_gtest)
        message(STATUS "googletest: ${GTEST_LIB}")
    else()
        message(STATUS "googletest: not found locally, fetching")
        include(FetchContent)
        FetchContent_Declare(googletest
            GIT_REPOSITORY https://github.com/google/googletest.git
            GIT_TAG v1.14.0)
        set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
        FetchContent_MakeAvailable(googletest)
        set(_GTEST_TARGETS gtest gtest_main)
    endif()
endif()

# ---- 被测算子的 op_host 目录（tiling core 头所在处）----
# _common/op_host 是各算子共用的切分算法（merge 时平铺进 msopgen 工程的 op_host/），
# 这里按源码位置加进 include 路径。
set(OPHOST_UT_INCLUDE_DIRS
    "${OPERATORS_DIR}/_common/op_host"
    "${OPERATORS_DIR}/complex_mul/op_host"
    "${OPERATORS_DIR}/complex_mul_grad/op_host"
    "${OPERATORS_DIR}/fused_bias_softmax/op_host"
    "${OPERATORS_DIR}/fused_softmax_grad/op_host"
)
foreach(_dir ${OPHOST_UT_INCLUDE_DIRS})
    if(NOT IS_DIRECTORY "${_dir}")
        message(FATAL_ERROR "missing operator op_host dir: ${_dir}")
    endif()
endforeach()

# ---- 测试目标 ----
enable_testing()

add_executable(ophost_ut
    test_complex_mul_tiling.cpp
    test_fused_bias_softmax_tiling.cpp
    test_fused_softmax_grad_tiling.cpp
)
target_include_directories(ophost_ut PRIVATE ${OPHOST_UT_INCLUDE_DIRS})
target_link_libraries(ophost_ut PRIVATE ${_GTEST_TARGETS})
target_compile_options(ophost_ut PRIVATE -Wall -Wextra)

add_test(NAME ophost_ut COMMAND ophost_ut)
