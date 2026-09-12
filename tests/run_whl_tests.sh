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

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPS_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WHEEL_DIR="${OIL_GAS_OPS_OUTPUT_DIR:-${OPS_ROOT}/output}/wheels"

mkdir -p "${WHEEL_DIR}"
cd "${OPS_ROOT}"

python3 setup.py bdist_wheel --dist-dir "${WHEEL_DIR}"

WHEEL_PATH="$(
python3 - <<'PY' "${WHEEL_DIR}"
from pathlib import Path
import sys

wheel_dir = Path(sys.argv[1])
wheels = sorted(wheel_dir.glob("oil_gas_ops_prospect-*.whl"), key=lambda p: p.stat().st_mtime)
if not wheels:
    raise SystemExit("no oil_gas_ops_prospect wheel found")
print(wheels[-1])
PY
)"

python3 -m pip uninstall -y oil-gas-ops-prospect >/dev/null 2>&1 || true
python3 - <<'PY'
from pathlib import Path
import shutil
import site

roots = []
for getter in (site.getsitepackages,):
    try:
        roots.extend(getter())
    except Exception:
        pass
try:
    roots.append(site.getusersitepackages())
except Exception:
    pass

for root in {Path(p) for p in roots if p}:
    shutil.rmtree(root / "oil_gas_ops_prospect", ignore_errors=True)
    for path in root.glob("oil_gas_ops_prospect-*.dist-info"):
        shutil.rmtree(path, ignore_errors=True)
    for path in root.glob("oil_gas_ops_prospect-*.egg-info"):
        shutil.rmtree(path, ignore_errors=True)
    for path in root.glob("complex_mul_npu*.so"):
        path.unlink(missing_ok=True)
PY
python3 -m pip install --force-reinstall --no-deps "${WHEEL_PATH}"

# OIL_GAS_OPS_WHL_TEST=1 让 tests/conftest.py 不要把仓内源码挂到导入名下，测的是刚装上的 whl。
OIL_GAS_OPS_WHL_TEST=1 python3 -m pytest "${SCRIPT_DIR}/test_oil_gas_ops_prospect_whl.py" "$@"
