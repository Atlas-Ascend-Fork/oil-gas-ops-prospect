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

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
FAIL=0

# On NPU, force using compiled kernel — no silent fallback (P1-2)
if python3 -c "import torch; import torch_npu; exit(0 if torch.npu.is_available() else 1)" 2>/dev/null; then
  export OIL_GAS_OPS_REQUIRE_ASCENDC_CUSTOM_GRU_V2=1
  echo "[env] NPU detected — OIL_GAS_OPS_REQUIRE_ASCENDC_CUSTOM_GRU_V2=1 (no silent fallback)"
fi

echo "============================================"
echo "  CustomGRUV2 Quality Assurance Test Suite"
echo "============================================"
echo ""

echo "[1/4] Unit Tests"
python3 -m pytest "${SCRIPT_DIR}/unit/" -v --tb=short || FAIL=1
echo ""

echo "[2/4] Integration Tests"
python3 -m pytest "${SCRIPT_DIR}/integration/" -v --tb=short || FAIL=1
echo ""

echo "[3/4] Performance Tests"
python3 -m pytest "${SCRIPT_DIR}/performance/" -v -s || FAIL=1
echo ""

echo "[4/4] Coverage"
python3 "${SCRIPT_DIR}/coverage/run_coverage.py" || FAIL=1
echo ""

echo "============================================"
if [ "$FAIL" -eq 0 ]; then
  echo "  All tests completed successfully."
else
  echo "  Some test suites failed. See output above."
fi
echo "============================================"
exit "$FAIL"
