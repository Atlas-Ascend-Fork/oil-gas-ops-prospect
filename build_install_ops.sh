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
ASCENDC_ROOT="${SCRIPT_DIR}/ascendc"
# 所有编译产物统一落到仓库根目录 output/
OUTPUT_DIR="${OIL_GAS_OPS_OUTPUT_DIR:-${SCRIPT_DIR}/output}"
WHEEL_DIR="${OUTPUT_DIR}/wheels"

SOC_VERSION="Ascend910B1"
OPS_LIST="all"
RUN_TESTS=1
CLEAN=0
JOBS=""

usage() {
  cat <<'EOF'
Build and install the oil & gas exploration custom operators.

This script builds the AscendC OPP .run package, installs it into CANN, builds
the unified oil_gas_ops_prospect wheel, installs the wheel, and runs whl smoke tests.

Usage:
  bash build_install_ops.sh [options]

Options:
  --soc <SOC>        msopgen SOC version, default: Ascend910B1
  --ops <ops>        comma/space separated AscendC ops, default: all
  -j, --jobs <N>     build parallelism forwarded to ascendc/build.py
  --clean           clean ascendc/_work and output/ before building
  --skip-tests      install packages but skip whl smoke tests
  -h, --help        show this help

Artifacts (repo root output/):
  output/<SOC>/oil_gas_ops_prospect_opp_<os>_<arch>.run
  output/wheels/oil_gas_ops_prospect-<ver>-<py>-<abi>-<plat>.whl

Environment:
  ASCEND_HOME_PATH        CANN root, default: /usr/local/Ascend/cann-9.0.0
  OIL_GAS_OPS_OUTPUT_DIR  override artifact output dir, default: <repo>/output

AscendC package / UT (from ascendc/):
  bash build.sh --pkg
  bash build.sh -u --ophost

Recommended full install inside ascendc-dev:
  cd /root/projects/oil-gas-ops-prospect
  bash build_install_ops.sh
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --soc)
      SOC_VERSION="$2"
      shift 2
      ;;
    --ops)
      OPS_LIST="$2"
      shift 2
      ;;
    -j|--jobs)
      JOBS="$2"
      shift 2
      ;;
    --clean)
      CLEAN=1
      shift
      ;;
    --skip-tests)
      RUN_TESTS=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

export ASCEND_HOME_PATH="${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.0.0}"
export LD_LIBRARY_PATH="${ASCEND_HOME_PATH}/opp/vendors/customize/op_api/lib:${ASCEND_HOME_PATH}/lib64:${LD_LIBRARY_PATH:-}"

echo "[oil_gas_ops_prospect] ASCEND_HOME_PATH=${ASCEND_HOME_PATH}"
echo "[oil_gas_ops_prospect] SOC_VERSION=${SOC_VERSION}"
echo "[oil_gas_ops_prospect] OPS_LIST=${OPS_LIST}"

if [[ "${CLEAN}" == "1" ]]; then
  echo "[1/5] Cleaning AscendC build outputs..."
  (cd "${ASCENDC_ROOT}" && OIL_GAS_OPS_OUTPUT_DIR="${OUTPUT_DIR}" python3 build.py clean)
else
  echo "[1/5] Skip clean (use --clean to clean ascendc/_work and output/ first)."
fi

echo "[2/5] Building AscendC OPP package (bash build.sh --pkg)..."
mapfile -t OPS_ARRAY < <(python3 - <<'PY' "${OPS_LIST}"
import sys
raw = sys.argv[1].replace(",", " ")
for item in raw.split():
    print(item)
PY
)

BUILD_ARGS=(--pkg -v "${SOC_VERSION}" --ops "${OPS_ARRAY[@]}")
if [[ -n "${JOBS}" ]]; then
  BUILD_ARGS+=(-j "${JOBS}")
fi
(cd "${ASCENDC_ROOT}" && OIL_GAS_OPS_OUTPUT_DIR="${OUTPUT_DIR}" bash build.sh "${BUILD_ARGS[@]}")

RUN_PKG="$(
python3 - <<'PY' "${OUTPUT_DIR}/${SOC_VERSION}"
from pathlib import Path
import sys

out_dir = Path(sys.argv[1])
candidates = sorted(out_dir.glob("*_opp_*.run"), key=lambda p: p.stat().st_mtime)
if not candidates:
    raise SystemExit(f"no OPP run package found in {out_dir}")
print(candidates[-1])
PY
)"

echo "[3/5] Installing OPP package: ${RUN_PKG}"
bash "${RUN_PKG}" --quiet

echo "[4/5] Building and installing oil_gas_ops_prospect wheel..."
rm -rf "${SCRIPT_DIR}/build" "${SCRIPT_DIR}"/*.egg-info
mkdir -p "${WHEEL_DIR}"
(cd "${SCRIPT_DIR}" && python3 setup.py bdist_wheel --dist-dir "${WHEEL_DIR}")

WHEEL_PATH="$(
python3 - <<'PY' "${WHEEL_DIR}"
from pathlib import Path
import sys

wheel_dir = Path(sys.argv[1])
candidates = sorted(wheel_dir.glob("oil_gas_ops_prospect-*.whl"), key=lambda p: p.stat().st_mtime)
if not candidates:
    raise SystemExit(f"no oil_gas_ops_prospect wheel found in {wheel_dir}")
print(candidates[-1])
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
rm -rf "${SCRIPT_DIR}/build" "${SCRIPT_DIR}"/*.egg-info

if [[ "${RUN_TESTS}" == "1" ]]; then
  echo "[5/5] Running whl smoke tests..."
  python3 -m pytest "${SCRIPT_DIR}/tests/test_oil_gas_ops_prospect_whl.py" -q
else
  echo "[5/5] Skip tests (--skip-tests)."
fi

cat <<EOF

[oil_gas_ops_prospect] Done.
OPP package : ${RUN_PKG}
Wheel       : ${WHEEL_PATH}

For later shells, keep this runtime library path:
export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/opp/vendors/customize/op_api/lib:${ASCEND_HOME_PATH}/lib64:\$LD_LIBRARY_PATH
EOF
