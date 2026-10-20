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

run_unified_tests() {
  local suites=()
  [[ "${OP_HOST_UT}" == "TRUE" ]] && suites+=("ophost")
  [[ "${TRITON_UT}" == "TRUE" ]] && suites+=("triton")
  [[ "${COMPOSITE_UT}" == "TRUE" ]] && suites+=("composite")
  [[ "${AOT_UT}" == "TRUE" ]] && suites+=("aot")
  [[ "${ST_UT}" == "TRUE" ]] && suites+=("st")
  [[ "${PRECISION_UT}" == "TRUE" ]] && suites+=("precision")
  [[ "${PERF_UT}" == "TRUE" ]] && suites+=("perf")
  [[ "${EXAMPLES_UT}" == "TRUE" ]] && suites+=("examples")
  [[ "${FFT_UT}" == "TRUE" ]] && suites+=("fft")
  [[ "${GATE_UT}" == "TRUE" ]] && suites+=("gate")

  if [[ ${#suites[@]} -eq 0 ]]; then
    echo "[ERROR] no test suites selected" >&2
    exit 1
  fi

  local suite_csv
  suite_csv="$(IFS=,; echo "${suites[*]}")"

  # 取值一律用 --flag=value。裸 --orig-argv -u 会被 argparse 当成新选项，
  # 导致 `bash build.sh -u` / `--ut` / `--st` 等单 flag 入口立刻失败。
  local args=(
    python3 -u "${SCRIPT_DIR}/scripts/run_tests.py"
    --suites="${suite_csv}"
    --jobs="${THREAD_NUM}"
    --orig-argv="${USE_CMD}"
    --results-json="${SCRIPT_DIR}/test_result/test_results.json"
  )
  if [[ -n "${GTEST_FILTER}" ]]; then
    args+=(--gtest-filter="${GTEST_FILTER}")
  fi
  if [[ "${ENABLE_UT_EXEC}" != "TRUE" ]]; then
    args+=(--noexec)
  fi
  if [[ "${ENABLE_COVERAGE}" == "TRUE" ]]; then
    args+=(--cov)
  fi
  if [[ "${REQUIRE_NPU}" == "TRUE" ]]; then
    args+=(--require-npu)
  fi
  if [[ -n "${REPORT_PATH}" ]]; then
    args+=(--report="${REPORT_PATH}")
  fi
  if [[ -n "${OPENSOURCE_REPORT}" ]]; then
    args+=(--opensource-report="${OPENSOURCE_REPORT}")
  fi
  if [[ "${OPS_LIST}" != "all" && -n "${OPS_LIST}" ]]; then
    args+=(--ops="${OPS_LIST}")
  fi

  echo "[oil_gas_ops_prospect] test suites: ${suite_csv}"
  "${args[@]}"
}

split_ops_list() {
  python3 - <<'PY' "${OPS_LIST}"
import sys
raw = sys.argv[1].replace(",", " ")
for item in raw.split():
    print(item)
PY
}

latest_artifact() {
  python3 - <<'PY' "$1" "$2"
from pathlib import Path
import sys

out_dir = Path(sys.argv[1])
pattern = sys.argv[2]
candidates = sorted(out_dir.glob(pattern), key=lambda p: p.stat().st_mtime)
if not candidates:
    raise SystemExit(f"no artifact matching {pattern} in {out_dir}")
print(candidates[-1])
PY
}

maybe_clean_outputs() {
  local step="$1"
  if [[ "${ENABLE_CLEAN}" == "TRUE" ]]; then
    echo "[${step}] Cleaning AscendC build outputs..."
    (cd "${ASCENDC_ROOT}" && OIL_GAS_OPS_OUTPUT_DIR="${OUTPUT_DIR}" python3 build.py clean)
  else
    echo "[${step}] Skip clean (use --clean to clean ascendc/_work and output/ first)."
  fi
}

build_opp_run() {
  local step="${1:-}"
  if [[ -n "${step}" ]]; then
    echo "[${step}] Building AscendC OPP package..."
  fi
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
  RUN_PKG="$(latest_artifact "${OUTPUT_DIR}/${SOC_VERSION}" "*_opp_*.run")"
}

install_opp_run() {
  local step="${1:-}"
  if [[ -n "${step}" ]]; then
    echo "[${step}] Installing OPP package: ${RUN_PKG}"
  fi
  bash "${RUN_PKG}" --quiet
}

build_python_wheel() {
  local step="${1:-}"
  if [[ -n "${step}" ]]; then
    echo "[${step}] Building oil_gas_ops_prospect wheel..."
  fi
  rm -rf "${SCRIPT_DIR}/build" "${SCRIPT_DIR}"/*.egg-info
  mkdir -p "${WHEEL_DIR}"
  (cd "${SCRIPT_DIR}" && OIL_GAS_OPS_ASCENDC_OPS="${OPS_LIST}" python3 setup.py bdist_wheel --dist-dir "${WHEEL_DIR}")
  WHEEL_PATH="$(latest_artifact "${WHEEL_DIR}" "oil_gas_ops_prospect-*.whl")"
  rm -rf "${SCRIPT_DIR}/build" "${SCRIPT_DIR}"/*.egg-info
}

install_python_wheel() {
  local step="${1:-}"
  if [[ -n "${step}" ]]; then
    echo "[${step}] Installing wheel: ${WHEEL_PATH}"
  fi
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
}

run_whl_smoke() {
  local step="$1"
  if [[ "${ENABLE_SKIP_SMOKE}" == "TRUE" ]]; then
    echo "[${step}] Skip tests (--skip-tests)."
    return 0
  fi
  echo "[${step}] Running whl smoke tests..."
  # OIL_GAS_OPS_WHL_TEST=1 让 tests/conftest.py 不要把仓内源码挂到导入名下。
  OIL_GAS_OPS_WHL_TEST=1 \
    python3 -m pytest "${SCRIPT_DIR}/tests/test_oil_gas_ops_prospect_whl.py" -q
}

print_build_summary() {
  local installed="$1"
  echo
  echo "[oil_gas_ops_prospect] Done."
  echo "OPP package : ${RUN_PKG}"
  if [[ -n "${WHEEL_PATH}" ]]; then
    echo "Wheel       : ${WHEEL_PATH}"
  fi
  if [[ "${installed}" != "TRUE" ]]; then
    echo "Install     : skipped (packages left in output/)"
    return 0
  fi
  cat <<EOF

For later shells, keep this runtime library path:
export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/opp/vendors/customize/op_api/lib:${ASCEND_HOME_PATH}/lib64:\$LD_LIBRARY_PATH
EOF
}

# --pkg: 只编 OPP .run。默认: 编 .run 与 whl，不安装。--install: 编译安装一体并冒烟。
run_build_pipeline() {
  echo "[oil_gas_ops_prospect] SOC_VERSION=${SOC_VERSION}"
  echo "[oil_gas_ops_prospect] OPS_LIST=${OPS_LIST}"

  RUN_PKG=""
  WHEEL_PATH=""

  if [[ "${ENABLE_PACKAGE}" == "TRUE" ]]; then
    maybe_clean_outputs "1/2"
    build_opp_run "2/2"
    print_build_summary FALSE
    return 0
  fi

  if [[ "${ENABLE_INSTALL}" != "TRUE" ]]; then
    maybe_clean_outputs "1/3"
    build_opp_run "2/3"
    build_python_wheel "3/3"
    print_build_summary FALSE
    return 0
  fi

  maybe_clean_outputs "1/5"
  build_opp_run "2/5"
  install_opp_run "3/5"
  echo "[4/5] Building and installing oil_gas_ops_prospect wheel..."
  build_python_wheel
  install_python_wheel
  run_whl_smoke "5/5"
  print_build_summary TRUE
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
    run_unified_tests
    return 0
  fi

  if [[ -n "${OPENSOURCE_REPORT}" ]]; then
    local json_path="${SCRIPT_DIR}/test_result/test_results.json"
    echo "[oil_gas_ops_prospect] generating opensource report from ${json_path}"
    python3 "${SCRIPT_DIR}/scripts/opensource_report.py" \
      --from-json="${json_path}" \
      --out="${OPENSOURCE_REPORT}"
    return 0
  fi

  run_build_pipeline
}

main "$@"
