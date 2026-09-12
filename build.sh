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
ASCENDC_ROOT="${SCRIPT_DIR}/ascendc"
# 所有编译产物统一落到仓库根目录 output/
OUTPUT_DIR="${OIL_GAS_OPS_OUTPUT_DIR:-${SCRIPT_DIR}/output}"
WHEEL_DIR="${OUTPUT_DIR}/wheels"

# shellcheck source=scripts/build_options.sh
source "${SCRIPT_DIR}/scripts/build_options.sh"

run_ophost_ut() {
  local build_dir="${ASCENDC_ROOT}/_work/ophost_ut"
  local ut_src="${SCRIPT_DIR}/tests/ascendc/ophost"
  local cmake_flags=(-DCMAKE_BUILD_TYPE=Release)

  if [[ "${ENABLE_COVERAGE}" == "TRUE" ]]; then
    cmake_flags+=(-DCMAKE_CXX_FLAGS=--coverage -DCMAKE_EXE_LINKER_FLAGS=--coverage)
  fi

  echo "[ut] ophost tiling UT (googletest)..."
  cmake -S "${ut_src}" -B "${build_dir}" "${cmake_flags[@]}"
  cmake --build "${build_dir}" -j "${THREAD_NUM}"

  if [[ "${ENABLE_UT_EXEC}" != "TRUE" ]]; then
    echo "[ut] --noexec: skip running ophost_ut"
    return 0
  fi

  if [[ -n "${GTEST_FILTER}" ]]; then
    echo "[ut] --gtest_filter=${GTEST_FILTER}"
    "${build_dir}/ophost_ut" --gtest_filter="${GTEST_FILTER}"
  else
    ctest --test-dir "${build_dir}" --output-on-failure
  fi
}

run_triton_ut() {
  local pytest_args=("${SCRIPT_DIR}/tests/triton/real_fft" -q)

  if [[ "${ENABLE_UT_EXEC}" != "TRUE" ]]; then
    echo "[ut] --noexec: skip Triton CPU-fallback pytest"
    return 0
  fi

  if [[ "${ENABLE_COVERAGE}" == "TRUE" ]]; then
    if python3 -c "import pytest_cov" >/dev/null 2>&1; then
      pytest_args+=(--cov --cov-report=term-missing)
    else
      echo "[WARN] --cov requested but pytest-cov is not installed; skip Python coverage"
    fi
  fi

  echo "[ut] Triton CPU-fallback accuracy tests..."
  python3 -m pytest "${pytest_args[@]}"
}

run_aot_check() {
  if [[ "${ENABLE_UT_EXEC}" != "TRUE" ]]; then
    echo "[ut] --noexec: skip Triton AOT compile check"
    return 0
  fi
  echo "[ut] Triton AOT compile check..."
  python3 "${SCRIPT_DIR}/triton-ascend/tools/aot_compile_check.py"
}

run_device_free_ut() {
  local run_ophost=FALSE
  local run_triton=FALSE
  local run_aot=FALSE

  if [[ "${UT_TEST_ALL}" == "TRUE" || "${OP_HOST_UT}" == "TRUE" ]]; then
    run_ophost=TRUE
  fi
  if [[ "${UT_TEST_ALL}" == "TRUE" || "${TRITON_UT}" == "TRUE" ]]; then
    run_triton=TRUE
  fi
  if [[ "${UT_TEST_ALL}" == "TRUE" || "${AOT_UT}" == "TRUE" ]]; then
    run_aot=TRUE
  fi

  if [[ "${run_ophost}" == "TRUE" ]]; then
    run_ophost_ut
  fi
  if [[ "${run_triton}" == "TRUE" ]]; then
    run_triton_ut
  fi
  if [[ "${run_aot}" == "TRUE" ]]; then
    run_aot_check
  fi

  echo "[oil_gas_ops_prospect] Device-free unit tests passed."
}

split_ops_list() {
  python3 - <<'PY' "${OPS_LIST}"
import sys
raw = sys.argv[1].replace(",", " ")
for item in raw.split():
    print(item)
PY
}

run_full_pipeline() {
  echo "[oil_gas_ops_prospect] SOC_VERSION=${SOC_VERSION}"
  echo "[oil_gas_ops_prospect] OPS_LIST=${OPS_LIST}"

  if [[ "${ENABLE_CLEAN}" == "TRUE" ]]; then
    echo "[1/5] Cleaning AscendC build outputs..."
    (cd "${ASCENDC_ROOT}" && OIL_GAS_OPS_OUTPUT_DIR="${OUTPUT_DIR}" python3 build.py clean)
  else
    echo "[1/5] Skip clean (use --clean to clean ascendc/_work and output/ first)."
  fi

  echo "[2/5] Building AscendC OPP package (bash build.sh --pkg)..."
  local ops_item
  local ops_array=()
  while IFS= read -r ops_item; do
    [[ -n "${ops_item}" ]] && ops_array+=("${ops_item}")
  done < <(split_ops_list)

  local build_args=(--pkg -v "${SOC_VERSION}" --ops "${ops_array[@]}")
  if [[ -n "${JOBS}" ]]; then
    build_args+=(-j "${JOBS}")
  fi
  (cd "${ASCENDC_ROOT}" && OIL_GAS_OPS_OUTPUT_DIR="${OUTPUT_DIR}" bash build.sh "${build_args[@]}")

  local run_pkg
  run_pkg="$(
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

  echo "[3/5] Installing OPP package: ${run_pkg}"
  bash "${run_pkg}" --quiet

  echo "[4/5] Building and installing oil_gas_ops_prospect wheel..."
  rm -rf "${SCRIPT_DIR}/build" "${SCRIPT_DIR}"/*.egg-info
  mkdir -p "${WHEEL_DIR}"
  (cd "${SCRIPT_DIR}" && python3 setup.py bdist_wheel --dist-dir "${WHEEL_DIR}")

  local wheel_path
  wheel_path="$(
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
  python3 -m pip install --force-reinstall --no-deps "${wheel_path}"
  rm -rf "${SCRIPT_DIR}/build" "${SCRIPT_DIR}"/*.egg-info

  if [[ "${ENABLE_SKIP_SMOKE}" != "TRUE" ]]; then
    echo "[5/5] Running whl smoke tests..."
    # OIL_GAS_OPS_WHL_TEST=1 让 tests/conftest.py 不要把仓内源码挂到导入名下。
    OIL_GAS_OPS_WHL_TEST=1 \
      python3 -m pytest "${SCRIPT_DIR}/tests/test_oil_gas_ops_prospect_whl.py" -q
  else
    echo "[5/5] Skip tests (--skip-tests)."
  fi

  cat <<EOF

[oil_gas_ops_prospect] Done.
OPP package : ${run_pkg}
Wheel       : ${wheel_path}

For later shells, keep this runtime library path:
export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/opp/vendors/customize/op_api/lib:${ASCEND_HOME_PATH}/lib64:\$LD_LIBRARY_PATH
EOF
}

main() {
  checkopts "$@"

  export ASCEND_HOME_PATH="${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.0.0}"
  export LD_LIBRARY_PATH="${ASCEND_HOME_PATH}/opp/vendors/customize/op_api/lib:${ASCEND_HOME_PATH}/lib64:${LD_LIBRARY_PATH:-}"

  echo "[oil_gas_ops_prospect] ASCEND_HOME_PATH=${ASCEND_HOME_PATH}"

  if [[ "${ENABLE_TEST}" == "TRUE" ]]; then
    if [[ "${ENABLE_CLEAN}" == "TRUE" ]]; then
      echo "[ut] cleaning ophost UT build dir..."
      rm -rf "${ASCENDC_ROOT}/_work/ophost_ut"
    fi
    run_device_free_ut
    return 0
  fi

  run_full_pipeline
}

main "$@"
