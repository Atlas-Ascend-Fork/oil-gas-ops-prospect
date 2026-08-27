#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""custom_multi_op_gen — 基于 install.sh + msopgen 的多算子、多机型构建入口。"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

# ``scripts`` resolves because Python puts this script's directory (ascendc/) at
# the head of sys.path; build.py is always invoked as ``python3 <...>/build.py``
# (see build.sh / build_install_ops.sh), never imported as a library.
from scripts.cann_env import run_in_cann_shell
from scripts.config import (
    DEFAULT_PACKAGE_NAME,
    DEFAULT_VENDOR_NAME,
    OUTPUT_DIR,
    ROOT_DIR,
    SUPPORTED_SOC_VERSIONS,
    WORK_ROOT,
    write_dummy_op_ir_json,
)
from scripts.manifest import list_operator_names
from scripts.merge_operators import merge_operators
from scripts.patch_op_host import append_op_host_extra_cmake, patch_op_host_includes
from scripts.patch_package import patch_package_config, renamed_run_filename
from scripts.patch_presets import patch_presets


def _default_jobs() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, NotImplementedError):
        return os.cpu_count() or 4


_EPILOG = """
示例:
  bash build.sh --pkg
  bash build.sh --pkg -v Ascend910B1 --ops complex_mul complex_mul_grad
  bash build.sh -u --ophost
  python3 build.py -v Ascend910B1 --ops complex_mul
  python3 build.py clean
  python3 build.py list-ops
"""


def _add_target_args(parser: argparse.ArgumentParser) -> None:
    """构建目标：芯片型号与算子选择。"""
    parser.add_argument(
        "-v",
        "--soc-version",
        help="msopgen 芯片型号，逗号分隔（同 install.sh -v）",
    )
    parser.add_argument(
        "--ops",
        nargs="*",
        default=None,
        help="算子目录名；可多个或用逗号分隔；默认 all",
    )


def _add_toolchain_args(parser: argparse.ArgumentParser) -> None:
    """工具链：CANN 路径与并行度。"""
    parser.add_argument(
        "-i",
        "--install-path",
        help="CANN 安装路径",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=_default_jobs(),
        help="并行编译线程数",
    )


def _add_package_args(parser: argparse.ArgumentParser) -> None:
    """OPP 包命名（vendor 目录名与 .run 前缀）。"""
    parser.add_argument(
        "--vendor-name",
        default=DEFAULT_VENDOR_NAME,
        help=f"OPP vendor 目录名（默认 {DEFAULT_VENDOR_NAME}）",
    )
    parser.add_argument(
        "--package-name",
        default=DEFAULT_PACKAGE_NAME,
        help=f".run 文件名前缀，替换默认名中的 custom（默认 {DEFAULT_PACKAGE_NAME}，"
        f"如 custom_opp_<os>_<arch>.run → {DEFAULT_PACKAGE_NAME}_opp_<os>_<arch>.run）",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="多算子 OPP 构建（msopgen 脚手架 + 算子合并）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_EPILOG,
    )
    _add_target_args(parser)
    _add_toolchain_args(parser)
    _add_package_args(parser)
    parser.add_argument(
        "command",
        nargs="?",
        default="build",
        choices=("build", "clean", "list-ops", "ut-ophost"),
        help="子命令（默认 build）",
    )
    return parser.parse_args(argv)


def resolve_ops(ops_arg: list[str] | None) -> list[str]:
    if ops_arg is None or len(ops_arg) == 0:
        return list_operator_names()
    if len(ops_arg) == 1 and ops_arg[0] in ("all", ""):
        return list_operator_names()
    result: list[str] = []
    for item in ops_arg:
        result.extend(p.strip() for p in item.split(",") if p.strip())
    if not result:
        raise SystemExit("错误: 未选中任何算子")
    return result


def validate_soc(soc: str) -> None:
    if soc not in SUPPORTED_SOC_VERSIONS:
        supported = " ".join(SUPPORTED_SOC_VERSIONS)
        raise SystemExit(f"[ERROR] SOC_VERSION '{soc}' 不在支持列表: [{supported}]")


def parse_soc_list(soc_version: str) -> list[str]:
    return [s.strip() for s in soc_version.split(",") if s.strip()]


def find_run_package(custom_op: Path) -> Path:
    candidates = sorted(custom_op.glob("build_out/custom_opp_*.run"))
    if not candidates:
        candidates = sorted(custom_op.glob("build_out/*.run"))
    if not candidates:
        raise SystemExit(f"错误: 未找到 OPP 包 {custom_op}/build_out/custom_opp_*.run")
    return candidates[0]


def build_one_soc(
    soc: str,
    selected_ops: list[str],
    *,
    install_path: str | None,
    jobs: int,
    vendor_name: str,
    package_name: str,
) -> Path:
    validate_soc(soc)
    custom_op = WORK_ROOT / f"CustomOp_{soc}"
    out_dir = OUTPUT_DIR / soc
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n========== SOC: {soc} ==========")
    print(f"[1/3] msopgen gen (dummy 脚手架, ai_core-{soc})...")
    if custom_op.exists():
        shutil.rmtree(custom_op)

    dummy_json = write_dummy_op_ir_json()
    run_in_cann_shell(
        f'msopgen gen -i "{dummy_json}" -c "ai_core-{soc}" -lan cpp -out "{custom_op}"',
        install_path=install_path,
    )

    patch_presets(custom_op, soc)
    patch_package_config(custom_op, vendor_name, package_name)

    print(f"[2/3] 合并算子: {' '.join(selected_ops)}...")
    merge_operators(custom_op, selected_ops)
    patch_op_host_includes(custom_op)
    append_op_host_extra_cmake(custom_op, selected_ops)

    print("[3/3] 编译 (CustomOp/build.sh)...")
    run_in_cann_shell(
        f'export MAKEFLAGS="-j{jobs}" && bash build.sh',
        cwd=custom_op,
        install_path=install_path,
    )

    run_pkg = find_run_package(custom_op)
    dest_name = renamed_run_filename(run_pkg.name, package_name)
    dest = out_dir / dest_name
    shutil.copy2(run_pkg, dest)
    print(f"产物: {dest}")
    return dest


def cmd_build(args: argparse.Namespace) -> None:
    if not args.soc_version:
        raise SystemExit("错误: 必须指定 -v/--soc-version（与 install.sh 相同）")

    selected_ops = resolve_ops(args.ops)
    for soc in parse_soc_list(args.soc_version):
        build_one_soc(
            soc,
            selected_ops,
            install_path=args.install_path,
            jobs=args.jobs,
            vendor_name=args.vendor_name,
            package_name=args.package_name,
        )
    print(f"\n全部完成。产物目录: {OUTPUT_DIR}/")


def cmd_clean() -> None:
    for d in (WORK_ROOT, OUTPUT_DIR):
        if d.exists():
            shutil.rmtree(d)
    print(f"已清理 {WORK_ROOT}/ {OUTPUT_DIR}/")


def cmd_list_ops() -> None:
    print("manifest 算子:")
    for name in list_operator_names():
        print(f"  {name}")


def cmd_ut_ophost(args: argparse.Namespace) -> None:
    """编译并运行 ophost 侧单元测试（googletest，无需 NPU 设备）。"""
    ut_src = ROOT_DIR / "tests" / "ophost"
    if not (ut_src / "CMakeLists.txt").is_file():
        raise SystemExit(f"错误: 未找到 ophost UT 工程 {ut_src}/CMakeLists.txt")

    build_dir = WORK_ROOT / "ophost_ut"
    build_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/2] 配置并编译 ophost UT ({build_dir})...")
    run_in_cann_shell(
        f'cmake -S "{ut_src}" -B "{build_dir}" -DCMAKE_BUILD_TYPE=Release'
        f" && cmake --build \"{build_dir}\" -j{args.jobs}",
        install_path=args.install_path,
    )

    print("[2/2] 运行 ophost UT...")
    run_in_cann_shell(
        "ctest --output-on-failure",
        cwd=build_dir,
        install_path=args.install_path,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "clean":
            cmd_clean()
        elif args.command == "list-ops":
            cmd_list_ops()
        elif args.command == "ut-ophost":
            cmd_ut_ophost(args)
        else:
            cmd_build(args)
        return 0
    except subprocess.CalledProcessError as e:
        return e.returncode or 1


if __name__ == "__main__":
    sys.exit(main())
