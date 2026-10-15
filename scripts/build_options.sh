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
#
# Option parsing, validation, and help display.
# Sourced by the repo-root build.sh. Layout follows cann/ops-math scripts/build_options.sh;
# only flags this repository actually implements are listed.

dotted_line="--------------------------------------------------------------------------------"

usage() {
  local specific_help="${1:-}"

  if [[ -n "${specific_help}" ]]; then
    case "${specific_help}" in
      package)
        echo "Package Build Options:"
        echo "${dotted_line}"
        echo "    (default)              Build OPP .run and wheel into output/; do not install"
        echo "    --pkg                  Build only the OPP .run; do not build wheel or install"
        echo "    --install              Build both packages, install them, then run smoke tests"
        echo "    --soc=<SOC>            msopgen SOC version, default: Ascend910B1"
        echo "    --ops=op1,op2,...      Compile specified AscendC operators (comma/space separated), default: all"
        echo "    -j[n], --jobs <N>      Compile parallelism forwarded to ascendc/build.py"
        echo "    --clean                Clean ascendc/_work and output/ before building"
        echo "    --skip-tests           With --install: skip whl smoke tests"
        echo "${dotted_line}"
        echo "Examples:"
        echo "    bash build.sh --pkg"
        echo "    bash build.sh"
        echo "    bash build.sh --install"
        echo "    bash build.sh --clean"
        echo "    bash build.sh --ops=complex_mul,complex_mul_grad"
        echo "    bash build.sh --install --skip-tests"
        echo "    bash build.sh --soc=Ascend910B1 -j16 --install"
        return
        ;;
      test)
        echo "Test Options:"
        echo "${dotted_line}"
        echo "    -u                   Enter test mode (default: all categories; on-device SKIP without NPU)"
        echo "    --all                Explicit all categories (same as bare -u)"
        echo "    --ut                 UT: host tiling + on-device small aclnn launch (needs NPU)"
        echo "    --ophost             UT-host only: tiling googletest (no NPU)"
        echo "    --st                 ST: multi-shape C++ aclnn + golden (needs NPU)"
        echo "    --precision          Python accuracy vs PyTorch (needs NPU)"
        echo "    --perf               Python performance (needs NPU)"
        echo "    --examples           UT-device only: C++ small aclnn launch (needs NPU)"
        echo "    --triton             Triton CPU-fallback pytest"
        echo "    --composite          Composite (pure PyTorch) CPU pytest"
        echo "    --aot                Triton AOT compile check"
        echo "    --ops=op1,op2        Filter UT-device / ST / precision / perf by operator"
        echo "    --report=<path>      Markdown execution report (default: test_result/test_report_<ts>.md)"
        echo "    --opensource-report[=path]  Copy community opensource report to test_result/"
        echo "                         default: test_result/oil-gas-ops-prospect仓开源测试报告.md"
        echo "                         JSON only gates FAIL; without -u: copy from last JSON, no re-run"
        echo "    --noexec             Only compile UT-host, do not execute"
        echo "    --cov                Enable coverage for host unit tests"
        echo "    --gtest_filter=pattern  gtest filter for UT-host"
        echo "${dotted_line}"
        echo "See tests/README.md for the category table."
        echo "Examples:"
        echo "    bash build.sh -u"
        echo "    bash build.sh -u --ut"
        echo "    bash build.sh -u --st --precision --perf"
        echo "    bash build.sh -u --ut --st --precision --perf"
        echo "    bash build.sh -u --ophost"
        echo "    bash build.sh -u --perf --ops=complex_mul"
        echo "    bash build.sh -u --report=test_result/my.md"
        echo "    bash build.sh -u --opensource-report"
        echo "    bash build.sh --opensource-report   # from last -u JSON, no re-run"
        return
        ;;
      clean)
        echo "Clean Options:"
        echo "${dotted_line}"
        echo "    --clean                Clean ascendc/_work and output/ (then continue the selected action)"
        echo "${dotted_line}"
        echo "Examples:"
        echo "    bash build.sh --clean"
        echo "    bash build.sh --clean -u --noexec"
        return
        ;;
    esac
  fi

  echo "build script for oil-gas-ops-prospect repository"
  echo "Usage:"
  echo "    bash build.sh [-h] [-j[n]] [-u]"
  echo ""
  echo "Default (no -u): build the AscendC OPP .run and the oil_gas_ops_prospect"
  echo "wheel into output/; do not install."
  echo ""
  echo "    bash build.sh --pkg           Build only the OPP .run (no wheel, no install)"
  echo "    bash build.sh                 Build .run and wheel into output/; do not install"
  echo "    bash build.sh --install        Build both packages, install them, then smoke-test"
  echo ""
  echo "Options:"
  echo "${dotted_line}"
  echo "    Build parameters"
  echo "${dotted_line}"
  echo "    -h, --help           Print usage; combine with -u / --clean for section help"
  echo "    -j[n], --jobs <N>    Compile thread nums, eg: -j8"
  echo "    --soc=<SOC>          msopgen SOC version, default: Ascend910B1"
  echo "    --ops=op1,op2,...    Compile specified AscendC operators, default: all"
  echo "    --pkg                Build only the OPP .run; do not build wheel or install"
  echo "    --install            Build .run and wheel, install both, then run smoke tests"
  echo "    --clean              Clean ascendc/_work and output/ before building"
  echo "    --skip-tests         With --install: skip whl smoke tests"
  echo "    -u                   Enter test mode (all categories by default; write Markdown report)"
  echo "${dotted_line}"
  echo "    examples, run UT (host tiling + on-device launch):"
  echo "    bash build.sh -u --ut"
  echo "${dotted_line}"
  echo "    The following are all supported arguments:"
  echo "${dotted_line}"
  echo "    --all                With -u: all categories (default when no subset flag)"
  echo "    --ut                 UT: tiling + on-device small aclnn (needs NPU; may omit -u)"
  echo "    --ophost             UT-host only: tiling googletest"
  echo "    --st                 ST: multi-shape C++ aclnn"
  echo "    --precision          NPU Python accuracy"
  echo "    --perf               NPU Python performance"
  echo "    --examples           UT-device only: C++ small aclnn"
  echo "    --triton             Triton CPU-fallback pytest"
  echo "    --composite          Composite (pure PyTorch) CPU pytest"
  echo "    --aot                Triton AOT compile check"
  echo "    --report=<path>      Markdown execution report path"
  echo "    --opensource-report[=path]  Copy community opensource report (JSON only gates FAIL)"
  echo "    --noexec             Only compile UT-host, do not execute"
  echo "    --cov                Enable coverage for host unit tests"
  echo "    --gtest_filter=pattern  gtest filter for UT-host, like: --gtest_filter=ComplexMul*"
  echo "    --pkg                Build only the OPP .run; cannot combine with -u / --install"
  echo "    --install            Build, install, and smoke-test; cannot combine with -u / --pkg"
  echo ""
  echo "Artifacts (repo root output/):"
  echo "    output/<SOC>/oil_gas_ops_prospect_opp_<os>_<arch>.run"
  echo "    output/wheels/oil_gas_ops_prospect-<ver>-<py>-<abi>-<plat>.whl"
  echo ""
  echo "Environment:"
  echo "    ASCEND_HOME_PATH        CANN root, default: /usr/local/Ascend/cann-9.0.0"
  echo "    OIL_GAS_OPS_OUTPUT_DIR  override artifact output dir, default: <repo>/output"
  echo ""
  echo "Section help:"
  echo "    bash build.sh -h"
  echo "    bash build.sh -u -h"
  echo "    bash build.sh --pkg -h"
  echo "    bash build.sh --install -h"
  echo "    bash build.sh --clean -h"
}

check_help_combinations() {
  local args=("$@")
  local has_u=false
  local has_package=false

  for arg in "${args[@]}"; do
    case "${arg}" in
      -u|--ut|--ophost|--st|--precision|--perf|--examples|--triton|--composite|--aot|--all) has_u=true ;;
      --pkg|--install) has_package=true ;;
    esac
  done

  if [[ "${has_package}" == "true" && "${has_u}" == "true" ]]; then
    echo "[ERROR] --pkg / --install cannot be used with test (-u / --ut / --st / ...)" >&2
    return 1
  fi
  return 0
}

set_ut_mode() {
  if [[ "${ENABLE_TEST}" != "TRUE" ]]; then
    return
  fi
  UT_TEST_ALL=TRUE
  local has_subset=FALSE
  if [[ "${OP_UT}" == "TRUE" || "${OP_HOST}" == "TRUE" || "${OP_TRITON}" == "TRUE" || "${OP_COMPOSITE}" == "TRUE" || \
        "${OP_AOT}" == "TRUE" || "${OP_ST}" == "TRUE" || "${OP_PRECISION}" == "TRUE" || "${OP_PERF}" == "TRUE" || \
        "${OP_EXAMPLES}" == "TRUE" || "${OP_ALL}" == "TRUE" ]]; then
    has_subset=TRUE
  fi

  if [[ "${OP_ALL}" == "TRUE" || "${has_subset}" == "FALSE" ]]; then
    OP_HOST_UT=TRUE
    TRITON_UT=TRUE
    COMPOSITE_UT=TRUE
    AOT_UT=TRUE
    ST_UT=TRUE
    PRECISION_UT=TRUE
    PERF_UT=TRUE
    EXAMPLES_UT=TRUE
    FFT_UT=TRUE
    GATE_UT=TRUE
    UT_TEST_ALL=TRUE
    REQUIRE_NPU=FALSE
    return
  fi

  UT_TEST_ALL=FALSE
  REQUIRE_NPU=FALSE
  if [[ "${OP_UT}" == "TRUE" ]]; then
    # UT = host tiling + on-device small launch
    OP_HOST_UT=TRUE
    EXAMPLES_UT=TRUE
    REQUIRE_NPU=TRUE
  fi
  if [[ "${OP_HOST}" == "TRUE" ]]; then
    OP_HOST_UT=TRUE
  fi
  if [[ "${OP_TRITON}" == "TRUE" ]]; then
    TRITON_UT=TRUE
  fi
  if [[ "${OP_COMPOSITE}" == "TRUE" ]]; then
    COMPOSITE_UT=TRUE
  fi
  if [[ "${OP_AOT}" == "TRUE" ]]; then
    AOT_UT=TRUE
  fi
  if [[ "${OP_ST}" == "TRUE" ]]; then
    ST_UT=TRUE
    REQUIRE_NPU=TRUE
  fi
  if [[ "${OP_PRECISION}" == "TRUE" ]]; then
    PRECISION_UT=TRUE
    REQUIRE_NPU=TRUE
  fi
  if [[ "${OP_PERF}" == "TRUE" ]]; then
    PERF_UT=TRUE
    REQUIRE_NPU=TRUE
  fi
  if [[ "${OP_EXAMPLES}" == "TRUE" ]]; then
    EXAMPLES_UT=TRUE
    REQUIRE_NPU=TRUE
  fi
}

check_param() {
  if [[ "${ENABLE_PACKAGE}" == "TRUE" && "${ENABLE_INSTALL}" == "TRUE" ]]; then
    echo "[ERROR] --pkg and --install cannot be used together" >&2
    exit 1
  fi

  if [[ "${ENABLE_PACKAGE}" == "TRUE" && "${ENABLE_TEST}" == "TRUE" ]]; then
    echo "[ERROR] --pkg cannot be used with -u / --ut / --st / --precision / ..." >&2
    exit 1
  fi

  if [[ "${ENABLE_INSTALL}" == "TRUE" && "${ENABLE_TEST}" == "TRUE" ]]; then
    echo "[ERROR] --install cannot be used with -u / --ut / --st / --precision / ..." >&2
    exit 1
  fi

  if [[ "${ENABLE_SKIP_SMOKE}" == "TRUE" && "${ENABLE_INSTALL}" != "TRUE" ]]; then
    echo "[ERROR] --skip-tests can only be used with --install" >&2
    exit 1
  fi

  if [[ "${ENABLE_TEST}" != "TRUE" ]]; then
    if [[ "${ENABLE_UT_EXEC}" == "FALSE" || "${ENABLE_COVERAGE}" == "TRUE" || -n "${GTEST_FILTER}" || -n "${REPORT_PATH}" ]]; then
      echo "[ERROR] --noexec / --cov / --gtest_filter / --report can only be used with test mode (-u / --ut / --st / ...)" >&2
      usage test
      exit 1
    fi
  fi

  if [[ "${ENABLE_PACKAGE}" == "TRUE" && -n "${OPENSOURCE_REPORT}" ]]; then
    echo "[ERROR] --opensource-report cannot be used with --pkg" >&2
    exit 1
  fi

  if [[ "${ENABLE_INSTALL}" == "TRUE" && -n "${OPENSOURCE_REPORT}" ]]; then
    echo "[ERROR] --opensource-report cannot be used with --install" >&2
    exit 1
  fi

  if [[ -n "${GTEST_FILTER}" && "${ENABLE_TEST}" == "TRUE" ]]; then
    if [[ "${OP_HOST_UT}" != "TRUE" ]]; then
      echo "[ERROR] --gtest_filter can only be used with UT-host (-u --ophost, -u --ut, or -u with default all)" >&2
      exit 1
    fi
  fi
}

_default_jobs() {
  if command -v nproc >/dev/null 2>&1; then
    nproc
  elif command -v sysctl >/dev/null 2>&1; then
    sysctl -n hw.ncpu 2>/dev/null || echo 8
  else
    echo 8
  fi
}

checkopts() {
  THREAD_NUM="$(_default_jobs)"
  JOBS=""
  SOC_VERSION="Ascend910B1"
  OPS_LIST="all"
  GTEST_FILTER=""
  USE_CMD="$*"

  ENABLE_TEST=FALSE
  ENABLE_PACKAGE=FALSE
  ENABLE_INSTALL=FALSE
  ENABLE_CLEAN=FALSE
  ENABLE_SKIP_SMOKE=FALSE
  ENABLE_UT_EXEC=TRUE
  ENABLE_COVERAGE=FALSE
  OP_HOST=FALSE
  OP_TRITON=FALSE
  OP_COMPOSITE=FALSE
  OP_AOT=FALSE
  OP_UT=FALSE
  OP_ST=FALSE
  OP_PRECISION=FALSE
  OP_PERF=FALSE
  OP_EXAMPLES=FALSE
  OP_ALL=FALSE
  OP_HOST_UT=FALSE
  TRITON_UT=FALSE
  COMPOSITE_UT=FALSE
  AOT_UT=FALSE
  ST_UT=FALSE
  PRECISION_UT=FALSE
  PERF_UT=FALSE
  EXAMPLES_UT=FALSE
  FFT_UT=FALSE
  GATE_UT=FALSE
  UT_TEST_ALL=FALSE
  REQUIRE_NPU=FALSE
  REPORT_PATH=""
  OPENSOURCE_REPORT=""

  local arg
  for arg in "$@"; do
    if [[ "${arg}" == "--help" || "${arg}" == "-h" ]]; then
      if ! check_help_combinations "$@"; then
        exit 1
      fi
      local show_help="general"
      local prev
      for prev in "$@"; do
        case "${prev}" in
          -u|--ut|--ophost|--triton|--composite|--aot|--st|--precision|--perf|--examples|--all|--noexec|--cov|--gtest_filter*|--report*|--opensource-report*) show_help="test" ;;
          --clean) show_help="clean" ;;
          --pkg|--install|--skip-tests) show_help="package" ;;
        esac
      done
      usage "${show_help}"
      exit 0
    fi
  done

  while [[ $# -gt 0 ]]; do
    case "$1" in
      -u)
        ENABLE_TEST=TRUE
        shift
        ;;
      --ut)
        ENABLE_TEST=TRUE
        OP_UT=TRUE
        shift
        ;;
      --noexec)
        ENABLE_UT_EXEC=FALSE
        shift
        ;;
      --cov)
        ENABLE_COVERAGE=TRUE
        shift
        ;;
      --gtest_filter=*)
        GTEST_FILTER="${1#*=}"
        shift
        ;;
      --gtest_filter)
        GTEST_FILTER="${2:-}"
        if [[ -z "${GTEST_FILTER}" ]]; then
          echo "[ERROR] --gtest_filter requires a pattern" >&2
          exit 1
        fi
        shift 2
        ;;
      --ophost)
        ENABLE_TEST=TRUE
        OP_HOST=TRUE
        shift
        ;;
      --triton)
        ENABLE_TEST=TRUE
        OP_TRITON=TRUE
        shift
        ;;
      --composite)
        ENABLE_TEST=TRUE
        OP_COMPOSITE=TRUE
        shift
        ;;
      --aot)
        ENABLE_TEST=TRUE
        OP_AOT=TRUE
        shift
        ;;
      --st)
        ENABLE_TEST=TRUE
        OP_ST=TRUE
        shift
        ;;
      --precision)
        ENABLE_TEST=TRUE
        OP_PRECISION=TRUE
        shift
        ;;
      --perf)
        ENABLE_TEST=TRUE
        OP_PERF=TRUE
        shift
        ;;
      --examples)
        ENABLE_TEST=TRUE
        OP_EXAMPLES=TRUE
        shift
        ;;
      --all)
        ENABLE_TEST=TRUE
        OP_ALL=TRUE
        shift
        ;;
      --report=*)
        REPORT_PATH="${1#*=}"
        shift
        ;;
      --report)
        REPORT_PATH="${2:-}"
        if [[ -z "${REPORT_PATH}" ]]; then
          echo "[ERROR] --report requires a path" >&2
          exit 1
        fi
        shift 2
        ;;
      --opensource-report=*)
        OPENSOURCE_REPORT="${1#*=}"
        if [[ -z "${OPENSOURCE_REPORT}" ]]; then
          OPENSOURCE_REPORT="test_result/oil-gas-ops-prospect仓开源测试报告.md"
        fi
        shift
        ;;
      --opensource-report)
        if [[ -n "${2:-}" && "${2}" != -* ]]; then
          OPENSOURCE_REPORT="$2"
          shift 2
        else
          OPENSOURCE_REPORT="test_result/oil-gas-ops-prospect仓开源测试报告.md"
          shift
        fi
        ;;
      --pkg)
        ENABLE_PACKAGE=TRUE
        shift
        ;;
      --install)
        ENABLE_INSTALL=TRUE
        shift
        ;;
      --soc=*)
        SOC_VERSION="${1#*=}"
        shift
        ;;
      --soc)
        SOC_VERSION="${2:-}"
        if [[ -z "${SOC_VERSION}" ]]; then
          echo "[ERROR] --soc requires a value" >&2
          exit 1
        fi
        shift 2
        ;;
      --ops=*)
        OPS_LIST="${1#*=}"
        shift
        ;;
      --ops)
        OPS_LIST="${2:-}"
        if [[ -z "${OPS_LIST}" ]]; then
          echo "[ERROR] --ops requires a value" >&2
          exit 1
        fi
        shift 2
        ;;
      -j)
        THREAD_NUM="${2:-}"
        JOBS="${THREAD_NUM}"
        if [[ -z "${THREAD_NUM}" ]]; then
          echo "[ERROR] -j requires a value" >&2
          exit 1
        fi
        shift 2
        ;;
      -j*)
        THREAD_NUM="${1#-j}"
        JOBS="${THREAD_NUM}"
        if [[ -z "${THREAD_NUM}" ]]; then
          echo "[ERROR] -j requires a value" >&2
          exit 1
        fi
        shift
        ;;
      --jobs)
        THREAD_NUM="${2:-}"
        JOBS="${THREAD_NUM}"
        if [[ -z "${THREAD_NUM}" ]]; then
          echo "[ERROR] --jobs requires a value" >&2
          exit 1
        fi
        shift 2
        ;;
      --clean)
        ENABLE_CLEAN=TRUE
        shift
        ;;
      --skip-tests)
        ENABLE_SKIP_SMOKE=TRUE
        shift
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      *)
        echo "[ERROR] Invalid param $1, Use 'bash build.sh --help' for more information." >&2
        usage >&2
        exit 2
        ;;
    esac
  done

  set_ut_mode
  check_param
}
