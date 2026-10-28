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

# Conv3DGelu：安装 OPP 后跑精度 / 性能 / 显存，以及 128×256×256 整网一步。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
export ASCEND_HOME_PATH="${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.0.0}"
export LD_LIBRARY_PATH="${ASCEND_HOME_PATH}/opp/vendors/customize/op_api/lib:${ASCEND_HOME_PATH}/lib64:${LD_LIBRARY_PATH:-}"
export ACLNN_CACHE_LIMIT="${ACLNN_CACHE_LIMIT:-0}"
export OIL_GAS_OPS_CONV3D_GELU_IMPL=ascendc
export OIL_GAS_OPS_REQUIRE_ASCENDC_CONV3D_GELU=1

OUTPUT_DIR="${OIL_GAS_OPS_OUTPUT_DIR:-${ROOT}/output}"
DEFAULT_RUN="${OUTPUT_DIR}/Ascend910B1/oil_gas_ops_prospect_opp_openEuler_aarch64.run"
RUN_PKG="${OIL_GAS_OPS_RUN_PKG:-${DEFAULT_RUN}}"

if [[ -f "${RUN_PKG}" ]]; then
  echo "[1/3] 安装/更新 OPP 包: ${RUN_PKG}"
  bash "${RUN_PKG}" --quiet
else
  echo "[1/3] 未找到 ${RUN_PKG}，沿用已安装的 customize vendor" >&2
fi

if ! python3 -c "import conv3d_gelu_npu" >/dev/null 2>&1; then
  echo "[2/3] 编译 conv3d_gelu_npu 扩展"
  bash "${SCRIPT_DIR}/build_ext.sh"
else
  echo "[2/3] 复用已有 conv3d_gelu_npu 扩展"
fi
export PYTHONPATH="${ROOT}/ascendc:${ROOT}/ascendc/pybind:${PYTHONPATH:-}"

echo "[3/3] 单算子精度/性能/显存 + FDPOT 128×256×256 训练步"
python3 -u "${SCRIPT_DIR}/test_conv3d_gelu_npu.py"
python3 -u "${SCRIPT_DIR}/test_fdpot_train_128.py"
