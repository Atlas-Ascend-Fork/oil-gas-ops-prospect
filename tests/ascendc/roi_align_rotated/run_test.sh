#!/usr/bin/env bash
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

# ROIAlignRotated / ROIAlignRotatedGrad 的 NPU 对比测试。
#
#   bash run_test.sh
#   bash run_test.sh --mode perf
#   OIL_GAS_OPS_RUN_PKG=/path/to/oil_gas_ops_prospect_opp_*.run bash run_test.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ASCEND_HOME_PATH="${ASCEND_HOME_PATH:-/home/developer/Ascend/cann-9.0.0}"
export LD_LIBRARY_PATH="${ASCEND_HOME_PATH}/opp/vendors/customize/op_api/lib:${ASCEND_HOME_PATH}/lib64:${LD_LIBRARY_PATH:-}"
# CANN 9 的 aclnn 执行器/输出地址缓存会干扰多 shape 连续调用，必须在 Python 启动前关闭。
export ACLNN_CACHE_LIMIT=0

OUTPUT_DIR="${OIL_GAS_OPS_OUTPUT_DIR:-${SCRIPT_DIR}/../../../output}"
RUN_PKG="${OIL_GAS_OPS_RUN_PKG:-}"
if [[ -z "${RUN_PKG}" ]]; then
  RUN_PKG="$(ls -t "${OUTPUT_DIR}"/Ascend910B3/*_opp_*.run "${OUTPUT_DIR}"/Ascend910B1/*_opp_*.run 2>/dev/null | head -1 || true)"
fi

if [[ -n "${RUN_PKG}" && -f "${RUN_PKG}" ]]; then
  echo "[1/2] 安装/更新 OPP 包: ${RUN_PKG}"
  bash "${RUN_PKG}" --quiet
else
  echo "[1/2] [WARN] 未找到 OPP 包，沿用当前已安装的 customize vendor" >&2
  echo "            如需构建: bash build.sh --pkg --ops=roi_align_rotated,roi_align_rotated_grad" >&2
fi

echo "[2/2] 编译扩展并运行 torch_npu 测试..."
python3 "${SCRIPT_DIR}/test_roi_align_rotated_npu.py" --build-ext "$@"
