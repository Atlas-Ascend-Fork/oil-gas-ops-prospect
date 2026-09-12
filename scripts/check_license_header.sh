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
#
# 检查源文件是否带 Apache License 2.0 许可证头。
# OAT.xml 中被 defaultPolicyFilter / copyrightPolicyFilter 过滤掉的文件类型不在检查范围内。
#
# 用法:
#   bash scripts/check_license_header.sh                 # 检查全部纳管源文件
#   bash scripts/check_license_header.sh <file>...       # 检查指定文件（pre-commit 走这条）

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MARKER="Licensed under the Apache License, Version 2.0"

# 只检查这些扩展名；其余（*.md / *.yaml / *.tsv / requirements.txt 等）按 OAT.xml 豁免。
is_checked() {
  case "$1" in
    *.c | *.h | *.cpp | *.hpp | *.cc | *.py | *.sh | *.cmake | */CMakeLists.txt | CMakeLists.txt) return 0 ;;
    *) return 1 ;;
  esac
}

# 构建产物与临时工程不纳入检查。
is_excluded() {
  case "$1" in
    output/* | ascendc/_work/* | build/*) return 0 ;;
    *) return 1 ;;
  esac
}

cd "${REPO_ROOT}"

if [[ $# -gt 0 ]]; then
  files=("$@")
else
  mapfile -t files < <(git ls-files)
fi

missing=()
for f in "${files[@]}"; do
  # pre-commit 传绝对路径时归一到仓库相对路径
  rel="${f#"${REPO_ROOT}/"}"
  [[ -f "${rel}" ]] || continue
  is_checked "${rel}" || continue
  is_excluded "${rel}" && continue
  if ! head -n 20 "${rel}" | grep -qF "${MARKER}"; then
    missing+=("${rel}")
  fi
done

if [[ ${#missing[@]} -gt 0 ]]; then
  echo "以下文件缺少 Apache License 2.0 许可证头:" >&2
  printf '  %s\n' "${missing[@]}" >&2
  echo "" >&2
  echo "请参考仓内已有源文件补齐文件头后重新提交。" >&2
  exit 1
fi

echo "license header check passed (${#files[@]} file(s) considered)."
