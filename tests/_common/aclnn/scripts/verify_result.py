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

"""ST 结果比对：按 (output, golden) 成对校验 float32 bin 文件。

判定规则：逐元素按 rtol / atol 判近似，不达标元素占比超过 ERROR_TOL 即失败（返回码 1）。
用法：verify_result.py output.bin golden.bin [output.bin golden.bin ...]
"""

from __future__ import annotations

import sys

import numpy as np

RELATIVE_TOL = 1e-5
ABSOLUTE_TOL = 1e-5
ERROR_TOL = 1e-4
MAX_LISTED = 100


def verify_pair(output_path: str, golden_path: str) -> bool:
    output = np.fromfile(output_path, dtype=np.float32).reshape(-1)
    golden = np.fromfile(golden_path, dtype=np.float32).reshape(-1)
    if output.size != golden.size:
        print(f"[ERROR] size mismatch: {output_path}={output.size} {golden_path}={golden.size}")
        return False

    mismatch = np.flatnonzero(
        ~np.isclose(output, golden, rtol=RELATIVE_TOL, atol=ABSOLUTE_TOL, equal_nan=True)
    )
    for index in mismatch[:MAX_LISTED]:
        want = float(golden[index])
        got = float(output[index])
        rdiff = abs(got - want) / (abs(want) if want else 1.0)
        print(f"[DIFF] {output_path}[{index}] golden={want:.9g} actual={got:.9g} rdiff={rdiff:.6g}")
    if mismatch.size > MAX_LISTED:
        print(f"[DIFF] {output_path}: 另有 {mismatch.size - MAX_LISTED} 个不达标元素未列出")

    ratio = float(mismatch.size) / golden.size
    # mismatch_ratio= 是 scripts/run_tests.py 摘要列提取的键，改名需同步那边的正则。
    print(f"[INFO] {output_path}: mismatch_ratio={ratio:.4f} limit={ERROR_TOL:.4f} elements={golden.size}")
    return ratio <= ERROR_TOL


def main(argv: list[str]) -> int:
    if not argv or len(argv) % 2 != 0:
        print("usage: verify_result.py output.bin golden.bin [output.bin golden.bin ...]")
        return 1
    ok = True
    for output_path, golden_path in zip(argv[::2], argv[1::2]):
        ok = verify_pair(output_path, golden_path) and ok
    if not ok:
        print("[ERROR] 结果比对未通过")
        return 1
    print("[INFO] 结果比对通过")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
