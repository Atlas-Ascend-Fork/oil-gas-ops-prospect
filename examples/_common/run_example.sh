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
# 由 examples/aclnn/<op>/run.sh source，需已设置：
#   CURRENT_DIR   该算子 examples/aclnn/<op> 目录
#   ACLNN_EXE     可执行文件名（与 CMake ACLNN_TARGET 一致）

set -u

EXAMPLE_COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${EXAMPLE_COMMON_DIR}/../.." && pwd)"

if [ -z "${CURRENT_DIR:-}" ] || [ -z "${ACLNN_EXE:-}" ]; then
    echo "[ERROR]: CURRENT_DIR and ACLNN_EXE must be set" >&2
    exit 1
fi

# shellcheck source-path=SCRIPTDIR source=../../scripts/cann_env.sh
source "${REPO_ROOT}/scripts/cann_env.sh"
cann_setup_env

example_stage() {
    local stage="$1"
    shift
    if ! "$@"; then
        echo "[ERROR]: example stage '${stage}' failed for ${ACLNN_EXE}" >&2
        exit 1
    fi
}

rm -rf "${CURRENT_DIR}/build" "${CURRENT_DIR}/output"
mkdir -p "${CURRENT_DIR}/build"

cd "${CURRENT_DIR}/build" || exit 1
example_stage cmake cmake .. -DCMAKE_SKIP_RPATH=TRUE
example_stage make make

cann_export_custom_op_api

cd "${CURRENT_DIR}/output" || exit 1
echo "[INFO]: Execute example ${ACLNN_EXE}"
example_stage launch "./${ACLNN_EXE}"
echo "[INFO]: example success"
