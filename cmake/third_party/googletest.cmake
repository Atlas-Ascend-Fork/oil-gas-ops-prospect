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

# googletest 依赖解析，结果放在 OIL_GAS_GTEST_TARGETS。
#
# 解析顺序：环境里已安装的 gtest（容器内一般在 /opt/conda）→ 裸库文件 → 从 gitcode
# 三方件镜像下载源码构建。镜像地址已在 SECURITY.md 的公网地址声明表中登记，版本与
# Third_Party_Open_Source_Software_List.yaml 保持一致。

if(DEFINED OIL_GAS_GTEST_TARGETS)
    return()
endif()

set(OIL_GAS_GTEST_VERSION "1.14.0" CACHE STRING "googletest version used by the ophost UT")
set(OIL_GAS_GTEST_URL
    "https://gitcode.com/cann-src-third-party/googletest/releases/download/v${OIL_GAS_GTEST_VERSION}/googletest-${OIL_GAS_GTEST_VERSION}.tar.gz"
    CACHE STRING "googletest source archive (gitcode third-party mirror)")

find_package(GTest QUIET)
if(GTest_FOUND)
    set(OIL_GAS_GTEST_TARGETS GTest::gtest GTest::gtest_main)
    message(STATUS "googletest: system (find_package)")
    return()
endif()

find_path(GTEST_INCLUDE_DIR gtest/gtest.h)
find_library(GTEST_LIB NAMES gtest)
find_library(GTEST_MAIN_LIB NAMES gtest_main)
if(GTEST_INCLUDE_DIR AND GTEST_LIB AND GTEST_MAIN_LIB)
    add_library(oil_gas_gtest INTERFACE)
    target_include_directories(oil_gas_gtest INTERFACE ${GTEST_INCLUDE_DIR})
    target_link_libraries(oil_gas_gtest INTERFACE ${GTEST_LIB} ${GTEST_MAIN_LIB} pthread)
    set(OIL_GAS_GTEST_TARGETS oil_gas_gtest)
    message(STATUS "googletest: ${GTEST_LIB}")
    return()
endif()

message(STATUS "googletest: not found locally, fetching ${OIL_GAS_GTEST_URL}")
include(FetchContent)
FetchContent_Declare(googletest URL "${OIL_GAS_GTEST_URL}")
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)
set(OIL_GAS_GTEST_TARGETS gtest gtest_main)
