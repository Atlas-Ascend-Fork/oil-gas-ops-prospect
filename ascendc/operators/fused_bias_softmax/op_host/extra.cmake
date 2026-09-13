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
#
# CANN 9 NnopbaseMatchArgs 按 shape 命中后复用旧 executor，原地写回会空跑。
foreach(_og_name aclnn_fused_bias_softmax.cpp aclnn_fused_softmax_grad.cpp)
    set(_og_file "${ASCEND_AUTOGEN_PATH}/${_og_name}")
    if(EXISTS "${_og_file}")
        file(READ "${_og_file}" _og_src)
        if(NOT _og_src MATCHES "oil_gas_ops_prospect disable MatchArgs")
            string(REGEX REPLACE
                "    if \\(NnopbaseSetMatchArgsFlag != NULL\\) \\{\r?\n        NnopbaseSetMatchArgsFlag\\(\\*executor\\);\r?\n    \\}"
                "    /* oil_gas_ops_prospect disable MatchArgs */"
                _og_src "${_og_src}")
            string(REGEX REPLACE
                "    if \\(NnopbaseMatchArgs != NULL\\) \\{\r?\n        if \\(NnopbaseMatchArgs\\(\\*executor, workspaceSize\\)\\) \\{\r?\n            NnopbaseReportApiInfo\\(timeStamp, dfxId\\);\r?\n            return ACLNN_SUCCESS;\r?\n        \\}\r?\n    \\}"
                "    /* oil_gas_ops_prospect disable MatchArgs hit */"
                _og_src "${_og_src}")
            file(WRITE "${_og_file}" "${_og_src}")
            message(STATUS "[oil_gas] disabled MatchArgs in ${_og_name}")
        endif()
    endif()
endforeach()
