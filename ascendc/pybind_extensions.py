#!/usr/bin/env python3
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

"""AscendC 算子 pybind 扩展的统一定义。

约定：`ascendc/pybind/csrc/<op_family>/` 下的 `*.cpp` 编译成扩展模块 `<op_family>_npu`，
新增算子只要建目录、放源码，两个构建入口都会自动带上，无需改本文件。
`csrc/_common/` 只放共享头，不单独编成扩展。

被两处复用：
  - 仓库根 `setup.py`：把扩展打进 `oil_gas_ops_prospect` whl；
  - `ascendc/pybind_setup.py`：就地编译，供 NPU 测试直接 import。

前置条件：OPP 包已安装到 CANN（`opp/vendors/customize/op_api/lib/libcust_opapi.so`
存在），因为扩展直接链接 msopgen 生成的 aclnn 接口。
"""
# pyright: reportMissingImports=false

from __future__ import annotations

import os
from pathlib import Path

import torch
from torch.utils.cpp_extension import CppExtension

ASCENDC_DIR = Path(__file__).resolve().parent
PYBIND_DIR = ASCENDC_DIR / "pybind"
CSRC_DIR = PYBIND_DIR / "csrc"

# 置 1 可在无 CANN / 无 torch_npu 的环境下跳过扩展编译（仅构建纯 Python 部分）。
SKIP_ENV = "OIL_GAS_OPS_SKIP_EXT"

ASCEND_DIR = os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/cann-9.0.0")


def _torch_npu_include_root() -> Path:
    try:
        import torch_npu

        return Path(torch_npu.__file__).parent
    except ImportError:
        return Path(torch.__file__).parent / ".." / "torch_npu"


def _vendor_opapi() -> Path:
    """自定义算子包的 vendor 目录（不同 CANN 版本布局略有差异）。"""
    candidates = [
        Path(ASCEND_DIR) / "opp" / "vendors" / "customize" / "op_api",
        Path(ASCEND_DIR) / "vendors" / "customize" / "op_api",
    ]
    return next(
        (p for p in candidates if (p / "lib" / "libcust_opapi.so").is_file()),
        candidates[0],
    )


def discover() -> list[Path]:
    """返回 ``pybind/csrc/<op_family>/`` 下带 C++ 源码的算子目录，按目录名排序。"""
    if not CSRC_DIR.is_dir():
        return []
    return sorted(
        d
        for d in CSRC_DIR.iterdir()
        if d.is_dir() and not d.name.startswith("_") and any(d.glob("*.cpp"))
    )


def build_extensions(relative_to: Path | None = None) -> list[CppExtension]:
    """构造全部 AscendC pybind 扩展。

    ``relative_to`` 是调用方 setup.py 所在目录；源码路径按它取相对路径，
    否则 setuptools 在打包时会把绝对路径写进产物。
    """
    if os.environ.get(SKIP_ENV, "0") == "1":
        return []

    base = Path(relative_to).resolve() if relative_to is not None else PYBIND_DIR
    torch_npu_root = _torch_npu_include_root()
    vendor_opapi = _vendor_opapi()

    include_dirs = [
        str(CSRC_DIR / "_common"),
        str(torch_npu_root / "include"),
        str(torch_npu_root / "include" / "third_party" / "acl" / "inc"),
        str(torch_npu_root / "include" / "third_party" / "op-plugin"),
        f"{ASCEND_DIR}/include",
        str(vendor_opapi / "include"),
    ]
    library_dirs = [
        f"{ASCEND_DIR}/lib64",
        str(vendor_opapi / "lib"),
        str(torch_npu_root / "lib"),
    ]
    libraries = ["cust_opapi", "ascendcl", "nnopbase", "opapi", "torch_npu"]

    extensions = []
    for op_dir in discover():
        sources = sorted(op_dir.glob("*.cpp"))
        extensions.append(
            CppExtension(
                name=f"{op_dir.name}_npu",
                sources=[str(src.relative_to(base)) for src in sources],
                include_dirs=include_dirs,
                library_dirs=library_dirs,
                libraries=libraries,
                extra_compile_args=["-O2", "-std=c++17"],
            )
        )
    return extensions
