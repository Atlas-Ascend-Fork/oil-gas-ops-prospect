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

# ophost 单元测试（googletest）。
#
# 入口: bash build.sh -u  或  bash build.sh -u --ophost  或  cd ascendc && bash build.sh -u --ophost
#
# 只编译 host 侧、与 gert / CANN 运行时无关的切分算法，因此**无卡环境即可跑**：
#   ascendc/operators/<op>/op_host/*_tiling_core.h  —  平台无关的 tiling 算法（header-only）
#
# 不覆盖 <Op>TilingFunc / InferShape 本身：它们的签名依赖 gert::TilingContext，而
# CANN 二进制包未提供 gert faker（该基建在 metadef 源码仓）。算子把切分算法抽到
# *_tiling_core.h 后，UT 直接驱动核心函数，覆盖的是真实算法。
#
# 新增算子时：把它的 *_tiling_core.h 所在 op_host 目录加进 OPHOST_UT_INCLUDE_DIRS，
# 并把测试源加进 ophost_ut 的 sources。

cmake_minimum_required(VERSION 3.16)

# 两种配置方式都支持：
#   - 仓库顶层 `cmake -S . -B build` 通过 add_subdirectory 引入；
#   - 独立配置 `cmake -S tests/ascendc/ophost -B <dir>`，`ascendc/build.py ut-ophost` 走这条路径。
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

# 被测源码在 ascendc/ 下，测试用例集中在 tests/ 下，两者通过仓库根目录关联。
set(OPERATORS_DIR "${REPO_ROOT}/ascendc/operators")

# ---- googletest ----
# 优先用环境里已装的 gtest（容器内为 /opt/conda）；找不到再从 gitcode 三方件镜像拉取。
include(third_party/googletest)
include(security_flags)

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
add_executable(ophost_ut
    test_complex_mul_tiling.cpp
    test_fused_bias_softmax_tiling.cpp
    test_fused_softmax_grad_tiling.cpp
)
target_include_directories(ophost_ut PRIVATE ${OPHOST_UT_INCLUDE_DIRS})
target_link_libraries(ophost_ut PRIVATE ${OIL_GAS_GTEST_TARGETS})
oil_gas_apply_security_flags(ophost_ut)

add_test(NAME ophost_ut COMMAND ophost_ut)
