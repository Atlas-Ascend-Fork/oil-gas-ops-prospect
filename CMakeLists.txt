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

# 仓库顶层 CMake 工程，覆盖**无卡即可构建**的 C++ 目标：ophost 侧 tiling 单元测试。
#
#   cmake -S . -B build && cmake --build build -j && ctest --test-dir build --output-on-failure
#
# AscendC 算子本身（op_host / op_kernel）不由本文件编译：CANN 自定义算子必须由
# msopgen 依据当前 CANN 与 SOC 生成配套工程后再交给 opbuild / bisheng 编译，入口是
# `bash build.sh --pkg` 与 `bash ascendc/build.sh --pkg`，详见 ascendc/README.md。

cmake_minimum_required(VERSION 3.16)

include("${CMAKE_CURRENT_SOURCE_DIR}/version.cmake")

project(oil_gas_ops_prospect
    VERSION ${OIL_GAS_OPS_PROSPECT_VERSION}
    DESCRIPTION "AscendC and Triton-Ascend custom operators for oil & gas exploration"
    LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(BUILD_OPHOST_UT "Build the device-free ophost tiling unit tests" ON)

message(STATUS "oil-gas-ops-prospect ${OIL_GAS_OPS_PROSPECT_VERSION} "
               "(CANN ${OIL_GAS_OPS_PROSPECT_CANN_VERSION}, SOC ${OIL_GAS_OPS_PROSPECT_SOC_VERSIONS})")

if(BUILD_OPHOST_UT)
    enable_testing()
    add_subdirectory(tests/ascendc/ophost)
endif()
