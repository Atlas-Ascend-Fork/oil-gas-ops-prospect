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
# 两段式 aclnn ST 驱动。由各算子 tests/st/aclnn*/run.sh source，需已设置：
#   CURRENT_DIR   tests/st/aclnn<OpType> 目录
#   ACLNN_EXE     可执行文件名
#   ACLNN_VERIFY_ARGS  verify_result.py 的参数（成对：output golden ...）
# 若存在 cases.json，则 cmake 一次后按 case 循环 gen_data / 执行 / 对比。

set -u

COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${COMMON_DIR}/../../.." && pwd)"

if [ -z "${CURRENT_DIR:-}" ] || [ -z "${ACLNN_EXE:-}" ]; then
    echo "[ERROR]: CURRENT_DIR and ACLNN_EXE must be set" >&2
    exit 1
fi
export CURRENT_DIR ACLNN_EXE ACLNN_VERIFY_ARGS

# shellcheck source-path=SCRIPTDIR source=../../../scripts/cann_env.sh
source "${REPO_ROOT}/scripts/cann_env.sh"
cann_setup_env

# 任一阶段失败即终止，并带上算子名便于在 -u 汇总日志里定位。
aclnn_stage() {
    local stage="$1"
    shift
    if ! "$@"; then
        echo "[ERROR]: st stage '${stage}' failed for ${ACLNN_EXE}" >&2
        exit 1
    fi
    echo "[INFO]: st stage '${stage}' done for ${ACLNN_EXE}"
}

rm -rf "${HOME}/ascend/log/"*
rm -f "${CURRENT_DIR}"/input/*.bin "${CURRENT_DIR}"/output/*.bin
rm -rf "${CURRENT_DIR}/build"
mkdir -p "${CURRENT_DIR}/build"

cd "${CURRENT_DIR}/build" || exit 1
aclnn_stage cmake cmake ../src -DCMAKE_SKIP_RPATH=TRUE
aclnn_stage make make

cann_export_custom_op_api

cd "${CURRENT_DIR}" || exit 1
if [ -f "${CURRENT_DIR}/cases.json" ]; then
    aclnn_stage cases python3 "${COMMON_DIR}/scripts/run_st_cases.py"
    exit 0
fi

aclnn_stage gen_data python3 scripts/gen_data.py

cd "${CURRENT_DIR}/output" || exit 1
aclnn_stage launch "./${ACLNN_EXE}"

cd "${CURRENT_DIR}" || exit 1
# shellcheck disable=SC2086
aclnn_stage verify python3 "${COMMON_DIR}/scripts/verify_result.py" ${ACLNN_VERIFY_ARGS}
