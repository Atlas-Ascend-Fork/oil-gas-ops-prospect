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

"""构建时拉取 ops-nn v9.0.0，只把 conv3d_v2 的 910B 源码拷进本次 CustomOp。

不编译 ops-nn 工程，也不改它的源码。arch35、Conv3DV2 自己的 kernel 入口和
op 注册不拷。cube 源码和本仓的 Conv3DGelu 在同一次 msopgen 编译里编到一起。
"""

from __future__ import annotations

import shutil
import tarfile
import urllib.request
from pathlib import Path

from scripts.config import OPERATORS_DIR, REPO_ROOT

OPS_NN_URL = "https://raw.gitcode.com/cann/ops-nn/archive/refs/heads/v9.0.0.tar.gz"
OPS_NN_ROOT_NAME = "ops-nn-v9.0.0"
DEPS_DIR = REPO_ROOT / "ascendc" / "_deps"
ARCHIVE_PATH = DEPS_DIR / "ops-nn-v9.0.0.tar.gz"
EXTRACT_ROOT = DEPS_DIR / OPS_NN_ROOT_NAME

# 这些 .cpp 是 tiling 引擎和 API tiling，Conv3DGelu 的 host 要链它们。
# 不编 conv3d_base_tiling / conv3d_tiling / tuning：那是 Conv3DV2 自己的注册入口。
HOST_CPP = (
    "conv3d_api_tiling.cpp",
    "conv3d_api_tiling_algorithm.cpp",
    "conv3d_api_tiling_algorithm_hw_mode.cpp",
    "conv3d_api_tiling_algorithm_pointwise.cpp",
    "conv3d_api_tiling_base.cpp",
    "conv3d_api_tiling_utils.cpp",
    "conv3d_tiling_engine.cpp",
)

KERNEL_SKIP_FILES = {"conv3d_v2.cpp", "conv3d_v2_template.h"}
KERNEL_SKIP_DIRS = {"arch35"}

CMAKE_MARKER = "# [oil_gas_ops_prospect] ops-nn conv3d_v2 sources"


def _conv3d_v2_dir() -> Path:
    return EXTRACT_ROOT / "conv" / "conv3d_v2"


def fetch_ops_nn() -> Path:
    """下载并解压 ops-nn v9.0.0。已解压则直接复用。"""
    src = _conv3d_v2_dir()
    if (src / "op_kernel" / "conv3dv2.h").is_file():
        return src
    DEPS_DIR.mkdir(parents=True, exist_ok=True)
    if not ARCHIVE_PATH.is_file() or ARCHIVE_PATH.stat().st_size < 1024 * 1024:
        print(f"[ops-nn] 下载 {OPS_NN_URL}")
        urllib.request.urlretrieve(OPS_NN_URL, ARCHIVE_PATH)
    print(f"[ops-nn] 解压 {ARCHIVE_PATH.name}")
    if EXTRACT_ROOT.exists():
        shutil.rmtree(EXTRACT_ROOT)
    with tarfile.open(ARCHIVE_PATH, "r:gz") as tar:
        tar.extractall(DEPS_DIR)
    if not (src / "op_kernel" / "conv3dv2.h").is_file():
        raise FileNotFoundError(f"压缩包里没有 conv3d_v2: {src}")
    return src


def _copy_kernel(src_kernel: Path, dst_kernel: Path) -> None:
    for item in src_kernel.iterdir():
        if item.name in KERNEL_SKIP_FILES or item.name in KERNEL_SKIP_DIRS:
            continue
        dest = dst_kernel / item.name
        if item.is_dir():
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(
                item,
                dest,
                ignore=shutil.ignore_patterns(*KERNEL_SKIP_DIRS),
            )
        elif not dest.exists():
            shutil.copy2(item, dest)


def _copy_host_tiling(src_tiling: Path, dst_tiling: Path) -> None:
    if dst_tiling.exists():
        shutil.rmtree(dst_tiling)
    dst_tiling.mkdir(parents=True)
    for name in HOST_CPP:
        src = src_tiling / name
        if not src.is_file():
            raise FileNotFoundError(f"ops-nn 缺少 {src}")
        shutil.copy2(src, dst_tiling / name)
    for header in src_tiling.glob("*.h"):
        shutil.copy2(header, dst_tiling / header.name)


def _link_tiling_data_for_host(kernel_dst: Path) -> None:
    """op_tiling 里的 `#include "../../op_kernel/conv3d_v2_tiling_data.h"` 指到算子目录外一层。"""
    target = kernel_dst / "conv3d_v2_tiling_data.h"
    link = kernel_dst.parent / "conv3d_v2_tiling_data.h"
    if not target.is_file():
        raise FileNotFoundError(f"缺少 {target}")
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(Path(kernel_dst.name) / "conv3d_v2_tiling_data.h")


def _inject_cmake(custom_op: Path) -> None:
    cmake_path = custom_op / "op_host" / "CMakeLists.txt"
    text = cmake_path.read_text(encoding="utf-8")
    if CMAKE_MARKER in text:
        return
    anchor = "aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR} ops_srcs)\n"
    inject = (
        f"{anchor}"
        f"{CMAKE_MARKER}\n"
        "include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/op_tiling)\n"
        "file(GLOB _OILGAS_CONV3D_TILING_SRC ${CMAKE_CURRENT_SOURCE_DIR}/op_tiling/*.cpp)\n"
        "list(APPEND ops_srcs ${_OILGAS_CONV3D_TILING_SRC})\n"
    )
    if anchor not in text:
        raise RuntimeError(f"找不到 aux_source_directory，无法把 ops-nn tiling 编进 host: {cmake_path}")
    cmake_path.write_text(text.replace(anchor, inject, 1), encoding="utf-8")


def stage_conv3d_v2(custom_op: Path) -> None:
    """把未修改的 conv3d_v2 源码放进本次 CustomOp，和 Conv3DGelu 一起编。"""
    src = fetch_ops_nn()
    kernel_dst = custom_op / "op_kernel" / "conv3d_gelu"
    if not kernel_dst.is_dir():
        raise FileNotFoundError(f"先合并本仓 kernel 再拉取 ops-nn: {kernel_dst}")
    _copy_kernel(src / "op_kernel", kernel_dst)
    _copy_host_tiling(src / "op_host" / "op_tiling", custom_op / "op_host" / "op_tiling")
    _link_tiling_data_for_host(kernel_dst)

    log_src = OPERATORS_DIR / "conv3d_gelu" / "op_host" / "log" / "log.h"
    # `#include "log/log.h"` 按头文件所在目录找。tiling 头在 op_tiling/ 下。
    for rel in ("op_host/log/log.h", "op_host/op_tiling/log/log.h"):
        log_dst = custom_op / rel
        log_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(log_src, log_dst)
    _inject_cmake(custom_op)
    print("[ops-nn] conv3d_v2 910B 源码已拷入本次编译，未编 ops-nn 工程")
