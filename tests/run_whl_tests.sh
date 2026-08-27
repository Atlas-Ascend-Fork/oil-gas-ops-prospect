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
python3 -m pytest "${SCRIPT_DIR}/test_oil_gas_ops_prospect_whl.py" "$@"
