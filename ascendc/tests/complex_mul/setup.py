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

"""编译 ComplexMulOg / ComplexMulGradOg 的 torch_npu pybind 扩展（complex_mul_npu）。

前置条件：OPP 包已安装到 CANN（`opp/vendors/customize/op_api/lib/libcust_opapi.so`
存在），因为扩展直接链接 msopgen 生成的 `aclnnComplexMulOg` / `aclnnComplexMulGradOg`。
"""
# pyright: reportMissingImports=false

from __future__ import annotations

import os
from pathlib import Path

import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

ROOT = Path(__file__).resolve().parent
CPP = ROOT / "cpp" / "bindings.cpp"

ASCEND_DIR = os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/cann-9.0.0")

try:
    import torch_npu

    TORCH_NPU_PATH = Path(torch_npu.__file__).parent
except ImportError:
    TORCH_NPU_PATH = Path(torch.__file__).parent / ".." / "torch_npu"

# 自定义算子包的 vendor 目录（不同 CANN 版本布局略有差异）
_vendor_candidates = [
    Path(ASCEND_DIR) / "opp" / "vendors" / "customize" / "op_api",
    Path(ASCEND_DIR) / "vendors" / "customize" / "op_api",
]
VENDOR_OPAPI = next(
    (p for p in _vendor_candidates if (p / "lib" / "libcust_opapi.so").is_file()),
    _vendor_candidates[0],
)

setup(
    name="complex_mul_npu",
    ext_modules=[
        CppExtension(
            name="complex_mul_npu",
            sources=[str(CPP)],
            include_dirs=[
                str(TORCH_NPU_PATH / "include"),
                str(TORCH_NPU_PATH / "include" / "third_party" / "acl" / "inc"),
                str(TORCH_NPU_PATH / "include" / "third_party" / "op-plugin"),
                f"{ASCEND_DIR}/include",
                str(VENDOR_OPAPI / "include"),
            ],
            library_dirs=[
                f"{ASCEND_DIR}/lib64",
                str(VENDOR_OPAPI / "lib"),
                str(TORCH_NPU_PATH / "lib"),
            ],
            libraries=["cust_opapi", "ascendcl", "nnopbase", "opapi", "torch_npu"],
            extra_compile_args=["-O2", "-std=c++17"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
