#!/usr/bin/python3
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

"""按 cases.json 循环：gen_data → 两段式 aclnn 可执行文件 → verify_result.py。

对齐 ops-nn tests/st 的多 case 组织；本仓没有 ATK，用 numpy 真值 + 已编译的
execute_*_op 代替 ATK executor。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    current = Path(os.environ["CURRENT_DIR"]).resolve()
    exe_name = os.environ["ACLNN_EXE"]
    verify_args = os.environ["ACLNN_VERIFY_ARGS"].split()
    common = Path(__file__).resolve().parent
    cases_path = current / "cases.json"
    cfg = json.loads(cases_path.read_text())
    cases = cfg["cases"]
    exe = current / "output" / exe_name
    if not exe.is_file():
        print(f"[ERROR]: missing {exe}", file=sys.stderr)
        return 1

    for case in cases:
        name = case["name"]
        shape = case["shape"]
        print(f"[INFO]: ===== ST case {name} shape={shape} =====")
        for folder in ("input", "output"):
            d = current / folder
            d.mkdir(exist_ok=True)
            for leftover in d.glob("*.bin"):
                leftover.unlink()
        gen_cmd = ["python3", str(current / "scripts" / "gen_data.py"), "--shape", ",".join(str(d) for d in shape)]
        if "bias1" in case:
            gen_cmd += ["--bias1", str(case["bias1"])]
        subprocess.check_call(gen_cmd, cwd=current)
        subprocess.check_call([str(exe)], cwd=current / "output")
        verify = ["python3", str(common / "verify_result.py"), *verify_args]
        subprocess.check_call(verify, cwd=current)
        print(f"[INFO]: case {name} pass")
    print(f"[INFO]: all {len(cases)} ST cases pass")
    return 0


if __name__ == "__main__":
    sys.exit(main())
