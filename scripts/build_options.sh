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
        echo "    (default)              Build OPP .run, install it, build/install the wheel, run smoke tests"
        echo "    --soc=<SOC>            msopgen SOC version, default: Ascend910B1"
        echo "    --ops=op1,op2,...      Compile specified AscendC operators (comma/space separated), default: all"
        echo "    -j[n], --jobs <N>      Compile parallelism forwarded to ascendc/build.py"
        echo "    --clean                Clean ascendc/_work and output/ before building"
        echo "    --skip-tests           Install packages but skip whl smoke tests"
        echo "${dotted_line}"
        echo "Examples:"
        echo "    bash build.sh"
        echo "    bash build.sh --clean"
        echo "    bash build.sh --ops=complex_mul,complex_mul_grad"
        echo "    bash build.sh --soc=Ascend910B1 -j16 --skip-tests"
        return
        ;;
      test)
        echo "Test Options:"
        echo "${dotted_line}"
        echo "    -u                     Build and run all device-free unit tests"
        echo "    --ut                   Alias of -u"
        echo "    --noexec               Only compile ophost UT, do not execute"
        echo "    --cov                  Enable coverage for unit tests"
        echo "    --gtest_filter=pattern Run only ophost tests matching the gtest filter"
        echo "    --ophost -u            Build and run ophost tiling unit tests"
        echo "    --triton -u            Run Triton CPU-fallback pytest"
        echo "    --aot -u               Run Triton AOT compile check"
        echo "${dotted_line}"
        echo "Examples:"
        echo "    bash build.sh -u"
        echo "    bash build.sh -u --noexec"
        echo "    bash build.sh -u --ophost"
        echo "    bash build.sh -u --ophost --gtest_filter=ComplexMul*"
        echo "    bash build.sh -u --triton --aot"
        echo "    bash build.sh -u --ophost --cov"
        echo "    bash build.sh --ophost --triton --aot -u"
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
  echo "Default (no -u): build the AscendC OPP .run package, install it into CANN,"
  echo "build and install the oil_gas_ops_prospect wheel, then run whl smoke tests."
  echo ""
  echo "Options:"
  echo "${dotted_line}"
  echo "    Build parameters"
  echo "${dotted_line}"
  echo "    -h, --help           Print usage; combine with -u / --clean for section help"
  echo "    -j[n], --jobs <N>    Compile thread nums, eg: -j8"
  echo "    --soc=<SOC>          msopgen SOC version, default: Ascend910B1"
  echo "    --ops=op1,op2,...    Compile specified AscendC operators, default: all"
  echo "    --clean              Clean ascendc/_work and output/ before building"
  echo "    --skip-tests         Install packages but skip whl smoke tests"
  echo "    -u, --ut             Build and run all device-free unit tests"
  echo "${dotted_line}"
  echo "    examples, run ophost UT only and do not execute:"
  echo "    bash build.sh -u --ophost --noexec"
  echo "${dotted_line}"
  echo "    The following are all supported arguments:"
  echo "${dotted_line}"
  echo "    --noexec             Only compile ophost UT, do not execute"
  echo "    --cov                Enable coverage for unit tests"
  echo "    --gtest_filter=pattern  gtest filter for ophost UT, like: --gtest_filter=ComplexMul*"
  echo "    --ophost             With -u: ophost tiling UT (googletest)"
  echo "    --triton             With -u: Triton CPU-fallback pytest"
  echo "    --aot                With -u: Triton AOT compile check"
  echo "    --pkg                Cannot be combined with -u (package path is the default without -u)"
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
  echo "    bash build.sh --clean -h"
}

check_help_combinations() {
  local args=("$@")
  local has_u=false
  local has_package=false

  for arg in "${args[@]}"; do
    case "${arg}" in
      -u|--ut) has_u=true ;;
      --pkg) has_package=true ;;
    esac
  done

  if [[ "${has_package}" == "true" && "${has_u}" == "true" ]]; then
    echo "[ERROR] --pkg cannot be used with test (-u / --ut)" >&2
    return 1
  fi
  return 0
}

set_ut_mode() {
  if [[ "${ENABLE_TEST}" != "TRUE" ]]; then
    return
  fi
  UT_TEST_ALL=TRUE
  if [[ "${OP_HOST}" == "TRUE" ]]; then
    OP_HOST_UT=TRUE
    UT_TEST_ALL=FALSE
  fi
  if [[ "${OP_TRITON}" == "TRUE" ]]; then
    TRITON_UT=TRUE
    UT_TEST_ALL=FALSE
  fi
  if [[ "${OP_AOT}" == "TRUE" ]]; then
    AOT_UT=TRUE
    UT_TEST_ALL=FALSE
  fi

  if [[ "${UT_TEST_ALL}" == "FALSE" && "${OP_HOST_UT}" == "FALSE" && "${TRITON_UT}" == "FALSE" && "${AOT_UT}" == "FALSE" ]]; then
    echo "[ERROR] At least one test target must be specified (use -u with one of: --ophost, --triton, --aot)" >&2
    usage test
    exit 1
  fi
}

check_param() {
  if [[ "${ENABLE_PACKAGE}" == "TRUE" && "${ENABLE_TEST}" == "TRUE" ]]; then
    echo "[ERROR] --pkg cannot be used with -u / --ut" >&2
    exit 1
  fi

  if [[ "${ENABLE_TEST}" != "TRUE" ]]; then
    if [[ "${OP_HOST}" == "TRUE" || "${OP_TRITON}" == "TRUE" || "${OP_AOT}" == "TRUE" ]]; then
      echo "[ERROR] --ophost / --triton / --aot require -u (this repo has no standalone library build for those flags)" >&2
      usage test
      exit 1
    fi
    if [[ "${ENABLE_UT_EXEC}" == "FALSE" || "${ENABLE_COVERAGE}" == "TRUE" || -n "${GTEST_FILTER}" ]]; then
      echo "[ERROR] --noexec / --cov / --gtest_filter can only be used with -u" >&2
      usage test
      exit 1
    fi
  fi

  if [[ -n "${GTEST_FILTER}" && "${ENABLE_TEST}" == "TRUE" ]]; then
    if [[ "${UT_TEST_ALL}" != "TRUE" && "${OP_HOST_UT}" != "TRUE" ]]; then
      echo "[ERROR] --gtest_filter can only be used with ophost UT (-u or -u --ophost)" >&2
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
  ENABLE_CLEAN=FALSE
  ENABLE_SKIP_SMOKE=FALSE
  ENABLE_UT_EXEC=TRUE
  ENABLE_COVERAGE=FALSE
  OP_HOST=FALSE
  OP_TRITON=FALSE
  OP_AOT=FALSE
  OP_HOST_UT=FALSE
  TRITON_UT=FALSE
  AOT_UT=FALSE
  UT_TEST_ALL=FALSE

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
          -u|--ut|--ophost|--triton|--aot|--noexec|--cov|--gtest_filter*) show_help="test" ;;
          --clean) show_help="clean" ;;
          --pkg|--skip-tests) show_help="package" ;;
        esac
      done
      usage "${show_help}"
      exit 0
    fi
  done

  while [[ $# -gt 0 ]]; do
    case "$1" in
      -u|--ut)
        ENABLE_TEST=TRUE
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
        OP_HOST=TRUE
        shift
        ;;
      --triton)
        OP_TRITON=TRUE
        shift
        ;;
      --aot)
        OP_AOT=TRUE
        shift
        ;;
      --pkg)
        ENABLE_PACKAGE=TRUE
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
