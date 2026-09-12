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

"""配置 OPP vendor 名与 .run 包名前缀。"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from scripts.config import DEFAULT_PACKAGE_NAME, DEFAULT_VENDOR_NAME


def _validate_name(value: str, label: str) -> None:
    if not value or not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise ValueError(f"{label} 无效: '{value}'（仅允许字母、数字、_、-）")


def renamed_run_filename(original_name: str, package_prefix: str) -> str:
    """custom_opp_<os>_<arch>.run → <prefix>_opp_<os>_<arch>.run"""
    if package_prefix == "custom":
        return original_name
    if original_name.startswith("custom_opp_"):
        return f"{package_prefix}_opp_{original_name[len('custom_opp_'):]}"
    return original_name.replace("custom", package_prefix, 1)


def patch_package_config(
    custom_op: Path,
    vendor_name: str = DEFAULT_VENDOR_NAME,
    package_name: str = DEFAULT_PACKAGE_NAME,
) -> None:
    """写入 CMakePresets 的 vendor_name；.run 文件名在构建后由前缀替换生成。"""
    _validate_name(vendor_name, "vendor_name")
    _validate_name(package_name, "package_name")

    preset_path = custom_op / "CMakePresets.json"
    data = json.loads(preset_path.read_text(encoding="utf-8"))
    cache = data["configurePresets"][0]["cacheVariables"]
    cache["vendor_name"]["value"] = vendor_name
    preset_path.write_text(json.dumps(data, indent=4) + "\n", encoding="utf-8")

    example = renamed_run_filename("custom_opp_openEuler_aarch64.run", package_name)
    print(f"[patch] vendor_name={vendor_name}, run 前缀: custom → {package_name} (例: {example})")


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if len(args) not in (1, 3):
        print(
            f"用法: {sys.argv[0]} <CustomOp目录> [vendor_name package_prefix]",
            file=sys.stderr,
        )
        return 1
    vendor = args[1] if len(args) == 3 else DEFAULT_VENDOR_NAME
    package = args[2] if len(args) == 3 else DEFAULT_PACKAGE_NAME
    patch_package_config(Path(args[0]), vendor, package)
    return 0


if __name__ == "__main__":
    sys.exit(main())
