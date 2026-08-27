#!/usr/bin/env bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# ComplexMulOg / ComplexMulGradOg 的 NPU 对比测试。
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
