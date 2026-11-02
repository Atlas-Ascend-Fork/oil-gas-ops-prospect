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

# msprof: NPU profiling for CustomGRUV2 kernel
# Collects: task-time, aicore metrics, memory bandwidth
set -uo pipefail

MSPROF=msprof
OP_DIR="$(cd "$(dirname "$0")/.." && pwd)"
RESULT_DIR="$OP_DIR/msopst/results_prof"
mkdir -p "$RESULT_DIR"

echo "=== msprof: CustomGRUV2 profiling ==="
echo "[1/2] Single sequence (B=1, T=1000, H=24)"
$MSPROF --application="python3 $(dirname "$0")/_prof_run.py --case single" \
    --output="$RESULT_DIR/single" \
    --ascendcl=on \
    --task-time=on \
    --storage-limit=200MB 2>&1 | tail -20

echo ""
echo "[2/2] Batch (B=32, T=100, H=24) — multi-core"
$MSPROF --application="python3 $(dirname "$0")/_prof_run.py --case batch" \
    --output="$RESULT_DIR/batch" \
    --ascendcl=on \
    --task-time=on \
    --storage-limit=200MB 2>&1 | tail -20

echo ""
echo "=== msprof complete. Results in $RESULT_DIR/ ==="
echo "View with: msprof --export=on --output=$RESULT_DIR/single"
