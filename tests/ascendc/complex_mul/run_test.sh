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

# ComplexMul / ComplexMulGrad 的 NPU 对比测试。
#
#   bash run_test.sh                        # 默认 shape
#   bash run_test.sh --shape 1 1 96 160 160
#
# 三步：安装/更新 OPP 包 → 编译 pybind 扩展 → 跑 torch_npu 精度/性能对比。
# 编译可在无卡容器内完成，但**跑测试需要真实 NPU**。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ASCEND_HOME_PATH="${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.0.0}"
export LD_LIBRARY_PATH="${ASCEND_HOME_PATH}/opp/vendors/customize/op_api/lib:${ASCEND_HOME_PATH}/lib64:${LD_LIBRARY_PATH:-}"

OUTPUT_DIR="${OIL_GAS_OPS_OUTPUT_DIR:-${SCRIPT_DIR}/../../../output}"
DEFAULT_RUN="${OUTPUT_DIR}/Ascend910B1/oil_gas_ops_prospect_opp_openEuler_aarch64.run"
RUN_PKG="${OIL_GAS_OPS_RUN_PKG:-${DEFAULT_RUN}}"

if [[ -f "${RUN_PKG}" ]]; then
  echo "[1/2] 安装/更新 OPP 包: ${RUN_PKG}"
  bash "${RUN_PKG}" --quiet
else
  echo "[1/2] [WARN] 未找到 OPP 包 ${RUN_PKG}，沿用当前已安装的 customize vendor" >&2
  echo "            如需构建: cd ascendc && bash build.sh --pkg" >&2
fi

echo "[2/2] 编译扩展并运行 torch_npu 测试..."
python3 "${SCRIPT_DIR}/test_complex_mul_npu.py" --build-ext "$@"
