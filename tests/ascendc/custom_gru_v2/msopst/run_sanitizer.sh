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

# mssanitizer: memory safety check for CustomGRUV2 kernel
# Checks: memcheck (UB overflow, GM out-of-bounds), racecheck (multi-core data races)
# Note: mssanitizer does not accept python3 -c; must use script files.
set -uo pipefail

SANITIZER=/usr/local/Ascend/cann-8.5.0/tools/mssanitizer/bin/mssanitizer
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OP_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RESULT_DIR="$OP_DIR/msopst/results_sanitizer"
mkdir -p "$RESULT_DIR"

MEMCHECK_SCRIPT="$SCRIPT_DIR/_memcheck_run.py"
RACECHECK_SCRIPT="$SCRIPT_DIR/_racecheck_run.py"

cat > "$MEMCHECK_SCRIPT" << 'PYEOF'
import torch, torch_npu
torch.npu.set_device(0)
from oil_gas_ops_prospect.custom_gru_v2 import custom_gru_v2
x = torch.randn(1, 100, 3, device='npu:0')
wi = torch.randn(72, 3, device='npu:0') * 0.1
wh = torch.randn(72, 24, device='npu:0') * 0.1
bi = torch.randn(72, device='npu:0') * 0.1
bh = torch.randn(72, device='npu:0') * 0.1
y, h = custom_gru_v2(x, wi, wh, bi, bh, 24, False)
print(f'memcheck standard: y.shape={y.shape}, h.shape={h.shape}')
wi2 = torch.randn(48, 3, device='npu:0') * 0.1
wh2 = torch.randn(48, 8, device='npu:0') * 0.1
bi2 = torch.randn(48, device='npu:0') * 0.1
bh2 = torch.randn(48, device='npu:0') * 0.1
x2 = torch.randn(1, 50, 3, device='npu:0')
y2, h2 = custom_gru_v2(x2, wi2, wh2, bi2, bh2, 8, True)
print(f'memcheck bidir: y2.shape={y2.shape}, h2.shape={h2.shape}')
x3 = torch.randn(1, 1, 3, device='npu:0')
wi3 = torch.randn(24, 3, device='npu:0') * 0.1
wh3 = torch.randn(24, 8, device='npu:0') * 0.1
bi3 = torch.randn(24, device='npu:0') * 0.1
bh3 = torch.randn(24, device='npu:0') * 0.1
y3, h3 = custom_gru_v2(x3, wi3, wh3, bi3, bh3, 8, False)
print(f'memcheck single_step: y3.shape={y3.shape}')
torch.npu.synchronize()
print('memcheck: ALL PASSED')
PYEOF

cat > "$RACECHECK_SCRIPT" << 'PYEOF'
import torch, torch_npu
torch.npu.set_device(0)
from oil_gas_ops_prospect.custom_gru_v2 import custom_gru_v2
x = torch.randn(32, 100, 3, device='npu:0')
wi = torch.randn(72, 3, device='npu:0') * 0.1
wh = torch.randn(72, 24, device='npu:0') * 0.1
bi = torch.randn(72, device='npu:0') * 0.1
bh = torch.randn(72, device='npu:0') * 0.1
y, h = custom_gru_v2(x, wi, wh, bi, bh, 24, False)
print(f'racecheck batch32: y.shape={y.shape}')
torch.npu.synchronize()
print('racecheck: PASSED')
PYEOF

echo "=== mssanitizer memcheck: CustomGRUV2 ==="
echo "[1/2] memcheck (UB overflow + GM out-of-bounds)"
$SANITIZER --tool=memcheck \
    --log-file="$RESULT_DIR/memcheck.log" \
    --log-level=info \
    --kernel-name=custom_gruv2 \
    python3 "$MEMCHECK_SCRIPT" 2>&1 | tee -a "$RESULT_DIR/memcheck.log"

echo ""
echo "[2/2] racecheck (multi-core data race detection)"
$SANITIZER --tool=racecheck \
    --log-file="$RESULT_DIR/racecheck.log" \
    --log-level=info \
    --kernel-name=custom_gruv2 \
    python3 "$RACECHECK_SCRIPT" 2>&1 | tee -a "$RESULT_DIR/racecheck.log"

echo ""
echo "=== mssanitizer complete. Results in $RESULT_DIR/ ==="
