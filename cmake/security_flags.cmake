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

# 安全编译选项，对应《版本发布网络安全质量要求》中的"要求类"选项。
# 用法：oil_gas_apply_security_flags(<target>)
#
# _FORTIFY_SOURCE 需要开启优化才生效，因此只在非 Debug 构建下加。

include(CheckCXXCompilerFlag)

function(oil_gas_apply_security_flags target)
    if(NOT TARGET ${target})
        message(FATAL_ERROR "oil_gas_apply_security_flags: no such target '${target}'")
    endif()

    if(MSVC)
        return()
    endif()

    target_compile_options(${target} PRIVATE
        -fstack-protector-strong        # 栈保护
        -fPIC                           # 位置无关代码
        -Wall
        -Wextra
    )
    target_compile_definitions(${target} PRIVATE
        $<$<NOT:$<CONFIG:Debug>>:_FORTIFY_SOURCE=2>   # 缓冲区溢出检查
    )

    foreach(_flag "-Wl,-z,relro" "-Wl,-z,now" "-Wl,-z,noexecstack")
        string(MAKE_C_IDENTIFIER "HAVE${_flag}" _cache_var)
        check_cxx_compiler_flag("${_flag}" ${_cache_var})
        if(${_cache_var})
            target_link_options(${target} PRIVATE "${_flag}")
        endif()
    endforeach()
endfunction()
