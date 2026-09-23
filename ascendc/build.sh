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

# AscendC 构建入口（对齐 ops-nn / ops-transformer 用法）
#
#   bash build.sh --pkg                 # 编译并打包 OPP .run → <repo>/output/<SOC>/
#   bash build.sh -u --ophost           # 编译并运行 ophost 侧 UT
#
# 兼容旧用法：
#   bash build.sh -v Ascend910B1 --ops complex_mul complex_mul_grad
#   bash build.sh clean | list-ops
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_DIR="${REPO_ROOT}/output"
BUILD_PY="${SCRIPT_DIR}/build.py"
DEFAULT_SOC="Ascend910B1"

MODE=""
UT_TARGET=""
HAS_SOC=0
FORWARD_ARGS=()

usage() {
  cat <<'EOF'
Usage:
  bash build.sh                              Build OPP .run and Python whl, no install (default)
  bash build.sh --pkg [options]              Build and package OPP .run only
  bash build.sh -u --ophost [options]        Build and run ophost unit tests
  bash build.sh clean                        Clean ascendc/_work/ and <repo>/output/
  bash build.sh list-ops                     List operators in manifest

Options (forwarded to build.py when packaging):
  -v, --soc-version <SOC>   msopgen SOC, default: Ascend910B1
  --ops <ops...>            operator names (default: all)
  -j, --jobs <N>            compile parallelism
  --vendor-name <name>      OPP vendor dir name
  --package-name <name>     .run filename prefix

Examples:
  bash build.sh --pkg
  bash build.sh --pkg -v Ascend910B1 --ops complex_mul complex_mul_grad -j 16
  bash build.sh -u --ophost
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --pkg)
      MODE="pkg"
      shift
      ;;
    -u)
      MODE="ut"
      shift
      ;;
    --ophost|--opapi|--opkernel)
      UT_TARGET="${1#--}"
      shift
      ;;
    -v|--soc-version)
      HAS_SOC=1
      FORWARD_ARGS+=("-v" "$2")
      shift 2
      ;;
    clean|list-ops)
      exec python3 "${BUILD_PY}" "$1"
      ;;
    *)
      FORWARD_ARGS+=("$1")
      shift
      ;;
  esac
done

run_pkg() {
  local args=()
  if [[ "${HAS_SOC}" -eq 0 ]]; then
    args+=("-v" "${DEFAULT_SOC}")
  fi
  if [[ ${#FORWARD_ARGS[@]} -gt 0 ]]; then
    args+=("${FORWARD_ARGS[@]}")
  fi
  echo "[build.sh] packaging via: python3 build.py ${args[*]}"
  python3 "${BUILD_PY}" "${args[@]}"
}

build_wheel() {
  # 与根目录 build.sh 的 build_python_wheel 一致：whl 落 <repo>/output/wheels/
  local repo_root wheel_dir
  repo_root="$(cd "${SCRIPT_DIR}/.." && pwd)"
  wheel_dir="${repo_root}/output/wheels"
  echo "[build.sh] building wheel via: python3 setup.py bdist_wheel --dist-dir ${wheel_dir}"
  (
    cd "${repo_root}"
    python3 setup.py bdist_wheel --dist-dir "${wheel_dir}"
    rm -rf build ./*.egg-info
  )
}

run_ut() {
  if [[ -z "${UT_TARGET}" ]]; then
    echo "错误: -u 需指定测试目标，例如 --ophost" >&2
    usage >&2
    exit 2
  fi
  if [[ "${UT_TARGET}" != "ophost" ]]; then
    echo "错误: 当前仅支持 -u --ophost（收到 --${UT_TARGET}）" >&2
    exit 2
  fi

  echo "[build.sh] running ophost UT (bash build.sh -u --ophost)..."
  local args=("ut-ophost")
  if [[ ${#FORWARD_ARGS[@]} -gt 0 ]]; then
    args+=("${FORWARD_ARGS[@]}")
  fi
  exec python3 "${BUILD_PY}" "${args[@]}"
}

case "${MODE}" in
  pkg)
    run_pkg
    ;;
  ut)
    run_ut
    ;;
  "")
    # 无参数 = 编译 .run 与 whl，不安装
    run_pkg
    build_wheel
    ;;
  *)
    echo "未知模式: ${MODE}" >&2
    usage >&2
    exit 2
    ;;
esac
