# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""对齐 msopgen CMakePresets 中的 ASCEND_COMPUTE_UNIT。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from scripts.config import SOC_COMPUTE_OVERRIDE


def patch_presets(custom_op: Path, soc_version: str) -> bool:
    """若需要 override 则写入 Presets，返回是否发生修改。"""
    override = SOC_COMPUTE_OVERRIDE.get(soc_version)
    if not override:
        return False

    preset_path = custom_op / "CMakePresets.json"
    data = json.loads(preset_path.read_text(encoding="utf-8"))
    cu = data["configurePresets"][0]["cacheVariables"]["ASCEND_COMPUTE_UNIT"]
    old = cu["value"]
    if old == override:
        return False

    cu["value"] = override
    preset_path.write_text(json.dumps(data, indent=4) + "\n", encoding="utf-8")
    print(f"[patch] ASCEND_COMPUTE_UNIT: {old} -> {override} (SOC={soc_version})")
    return True


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if len(args) != 2:
        print(f"用法: {sys.argv[0]} <CustomOp目录> <SOC_VERSION>", file=sys.stderr)
        return 1
    patch_presets(Path(args[0]), args[1])
    return 0


if __name__ == "__main__":
    sys.exit(main())
