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

"""将 operators/<name>/ 合并进 msopgen 生成的 CustomOp 工程。"""

from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

from scripts.config import (
    COMMON_DIR,
    DUMMY_HOST_FILES,
    DUMMY_KERNEL_FILES,
    KERNEL_CMAKE_FOOTER,
    KERNEL_CMAKE_HEADER,
    OPERATORS_DIR,
)
from scripts.manifest import OperatorEntry, lookup_entry

INCLUDE_PATTERN = re.compile(r'#include "../op_kernel/')


def _remove_dummy_files(custom_op: Path) -> None:
    for name in DUMMY_HOST_FILES:
        (custom_op / "op_host" / name).unlink(missing_ok=True)
    for name in DUMMY_KERNEL_FILES:
        (custom_op / "op_kernel" / name).unlink(missing_ok=True)


def _copy_common_host_headers(custom_op: Path) -> list[str]:
    """把 _common/op_host/*.h 平铺进 CustomOp/op_host/（与各算子 host 源同目录）。"""
    src_dir = COMMON_DIR / "op_host"
    if not src_dir.is_dir():
        return []
    dst_dir = custom_op / "op_host"
    dst_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for src in sorted(src_dir.glob("*.h")):
        shutil.copy2(src, dst_dir / src.name)
        copied.append(src.name)
    return copied


def _copy_common_kernel_headers(kernel_dst: Path) -> list[str]:
    """把 _common/op_kernel/*.h 拷进某个算子合并后的 op_kernel/<op>/ 目录。"""
    src_dir = COMMON_DIR / "op_kernel"
    if not src_dir.is_dir():
        return []
    kernel_dst.mkdir(parents=True, exist_ok=True)
    copied = []
    for src in sorted(src_dir.glob("*.h")):
        shutil.copy2(src, kernel_dst / src.name)
        copied.append(src.name)
    return copied


def _copy_host_sources(op: str, entry: OperatorEntry, custom_op: Path) -> None:
    op_src = OPERATORS_DIR / op
    host_src_dir = op_src / "op_host"
    host_dst_dir = custom_op / "op_host"

    # 同时拷贝 host 侧 .cpp 与 .h（如算子自带 tiling 库头文件），并把对
    # ../op_kernel/ 的引用重写到合并后的 ../op_kernel/<op>/ 子目录。
    host_sources = list(host_src_dir.glob("*.cpp")) + list(host_src_dir.glob("*.h"))
    for src in host_sources:
        dst = host_dst_dir / src.name
        shutil.copy2(src, dst)
        text = dst.read_text(encoding="utf-8")
        if INCLUDE_PATTERN.search(text):
            text = INCLUDE_PATTERN.sub(f'#include "../op_kernel/{op}/', text)
            dst.write_text(text, encoding="utf-8")


def _copy_kernel_tree(op: str, entry: OperatorEntry, custom_op: Path) -> None:
    op_src = OPERATORS_DIR / op
    kernel_src = op_src / "op_kernel" / entry.kernel_file
    if not kernel_src.is_file():
        raise FileNotFoundError(f"kernel 入口不存在: {kernel_src}")

    kernel_dst = custom_op / "op_kernel" / op
    kernel_dst.mkdir(parents=True, exist_ok=True)
    for item in (op_src / "op_kernel").iterdir():
        dest = kernel_dst / item.name
        if item.is_dir():
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(item, dest)
        else:
            shutil.copy2(item, dest)
    _copy_common_kernel_headers(kernel_dst)


def _kernel_registration_block(entry: OperatorEntry) -> str:
    return (
        f"npu_op_kernel_sources(ascendc_kernels\n"
        f"    OP_TYPE {entry.op_type}\n"
        f"    KERNEL_FILE {entry.kernel_file}\n"
        f"    KERNEL_DIR {entry.dir_name}/)\n\n"
    )


def write_kernel_cmake(custom_op: Path, selected_ops: list[str]) -> None:
    common_host = _copy_common_host_headers(custom_op)
    if common_host:
        print(f"[merge] 公共 host 头: {', '.join(common_host)}")
    blocks = [KERNEL_CMAKE_HEADER]
    for op in selected_ops:
        entry = lookup_entry(op)
        op_src = OPERATORS_DIR / op
        if not (op_src / "op_host").is_dir() or not (op_src / "op_kernel").is_dir():
            raise FileNotFoundError(f"缺少 operators/{op}/op_host 或 op_kernel")
        print(f"[merge] 合并算子: {op} ({entry.op_type})")
        _copy_host_sources(op, entry, custom_op)
        _copy_kernel_tree(op, entry, custom_op)
        blocks.append(_kernel_registration_block(entry))
    blocks.append(KERNEL_CMAKE_FOOTER)
    (custom_op / "op_kernel" / "CMakeLists.txt").write_text("".join(blocks), encoding="utf-8")


def merge_operators(custom_op: Path, selected_ops: list[str]) -> list[Path]:
    custom_op = custom_op.resolve()
    if not (custom_op / "op_host").is_dir() or not (custom_op / "op_kernel").is_dir():
        raise NotADirectoryError(f"不是有效的 msopgen 工程: {custom_op}")

    if not selected_ops:
        raise ValueError("未指定任何算子")

    print("[merge] 清理 msopgen dummy 占位文件...")
    _remove_dummy_files(custom_op)
    write_kernel_cmake(custom_op, selected_ops)

    host_files = sorted((custom_op / "op_host").glob("*.cpp"))
    print("[merge] 完成。Host 源:")
    for f in host_files:
        print(f"  {f.name}")
    return host_files


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if len(args) < 2:
        print(f"用法: {sys.argv[0]} <CustomOp目录> <op_dir> [op_dir...]", file=sys.stderr)
        return 1
    merge_operators(Path(args[0]), args[1:])
    return 0


if __name__ == "__main__":
    sys.exit(main())
