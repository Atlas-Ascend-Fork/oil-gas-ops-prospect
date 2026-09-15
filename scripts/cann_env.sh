#!/bin/bash
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
# CANN 安装目录解析与 aclnn 工程编译环境准备，由 tests/_common/aclnn/run_aclnn.sh 与
# examples/_common/run_example.sh source。对外提供：
#   cann_setup_env            解析根目录，导出 CANN_ROOT / DDK_PATH / NPU_HOST_LIB
#   cann_export_custom_op_api 把自定义算子包 op_api 加入 LD_LIBRARY_PATH

# 显式环境变量优先，其次按候选目录顺序取第一个存在的，都不存在时回落到默认安装位置。
cann_resolve_root() {
    local explicit="${ASCEND_INSTALL_PATH:-${ASCEND_HOME_PATH:-}}"
    if [ -n "${explicit}" ]; then
        printf '%s' "${explicit}"
        return 0
    fi

    local default_root="/usr/local/Ascend/ascend-toolkit/latest"
    local candidate
    for candidate in "${HOME}/Ascend/ascend-toolkit/latest" "/usr/local/Ascend/cann-9.0.0" "${default_root}"; do
        if [ -d "${candidate}" ]; then
            printf '%s' "${candidate}"
            return 0
        fi
    done
    printf '%s' "${default_root}"
}

# CANN 的主机侧库按 <机器架构>-<内核名小写> 分目录存放，例如 aarch64-linux。
cann_platform_dir() {
    printf '%s-%s' "$(uname -m)" "$(uname -s | tr '[:upper:]' '[:lower:]')"
}

cann_setup_env() {
    CANN_ROOT="$(cann_resolve_root)"

    local script
    for script in "${CANN_ROOT}/bin/setenv.bash" "${CANN_ROOT}/set_env.sh"; do
        if [ -f "${script}" ]; then
            # shellcheck disable=SC1090
            source "${script}"
            break
        fi
    done

    local platform
    platform="$(cann_platform_dir)"

    export CANN_ROOT
    export DDK_PATH="${CANN_ROOT}"
    export NPU_HOST_LIB="${CANN_ROOT}/${platform}/lib64"
}

cann_export_custom_op_api() {
    export LD_LIBRARY_PATH="${CANN_ROOT}/opp/vendors/customize/op_api/lib:${LD_LIBRARY_PATH:-}"
}
