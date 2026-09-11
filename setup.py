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

# pyright: reportMissingImports=false

from __future__ import annotations

import os
from pathlib import Path

import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension


ROOT = Path(__file__).resolve().parent
ASCEND_DIR = os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/cann-9.0.0")

try:
    import torch_npu

    TORCH_NPU_PATH = Path(torch_npu.__file__).parent
except ImportError:
    TORCH_NPU_PATH = Path(torch.__file__).parent / ".." / "torch_npu"


def _fused_bias_softmax_extension():
    """pybind 扩展 fused_bias_softmax_npu：直连 aclnnFusedBiasSoftmaxOg / aclnnFusedSoftmaxGradOg。"""
    if os.environ.get("OIL_GAS_OPS_SKIP_EXT", "0") == "1":
        return []

    cpp = ROOT / "ascendc" / "tests" / "fused_bias_softmax" / "cpp" / "bindings.cpp"
    if not cpp.is_file():
        return []

    include_dirs = [
        str(TORCH_NPU_PATH / "include"),
        str(TORCH_NPU_PATH / "include" / "third_party" / "acl" / "inc"),
        str(TORCH_NPU_PATH / "include" / "third_party" / "op-plugin"),
        f"{ASCEND_DIR}/include",
        f"{ASCEND_DIR}/opp/vendors/customize/op_api/include",
    ]
    library_dirs = [
        f"{ASCEND_DIR}/lib64",
        f"{ASCEND_DIR}/opp/vendors/customize/op_api/lib",
        str(TORCH_NPU_PATH / "lib"),
    ]

    return [
        CppExtension(
            name="fused_bias_softmax_npu",
            sources=[str(cpp.relative_to(ROOT))],
            include_dirs=include_dirs,
            library_dirs=library_dirs,
            libraries=["cust_opapi", "ascendcl", "nnopbase", "opapi", "torch_npu"],
            extra_compile_args=["-O2", "-std=c++17"],
        )
    ]


def _complex_mul_extension():
    """pybind 扩展 complex_mul_npu：直连 aclnnComplexMulOg / aclnnComplexMulGradOg。"""
    if os.environ.get("OIL_GAS_OPS_SKIP_EXT", "0") == "1":
        return []

    cpp = ROOT / "ascendc" / "tests" / "complex_mul" / "cpp" / "bindings.cpp"
    if not cpp.is_file():
        return []

    include_dirs = [
        str(TORCH_NPU_PATH / "include"),
        str(TORCH_NPU_PATH / "include" / "third_party" / "acl" / "inc"),
        str(TORCH_NPU_PATH / "include" / "third_party" / "op-plugin"),
        f"{ASCEND_DIR}/include",
        f"{ASCEND_DIR}/opp/vendors/customize/op_api/include",
    ]
    library_dirs = [
        f"{ASCEND_DIR}/lib64",
        f"{ASCEND_DIR}/opp/vendors/customize/op_api/lib",
        str(TORCH_NPU_PATH / "lib"),
    ]
    libraries = ["cust_opapi", "ascendcl", "nnopbase", "opapi", "torch_npu"]

    return [
        CppExtension(
            name="complex_mul_npu",
            sources=[str(cpp.relative_to(ROOT))],
            include_dirs=include_dirs,
            library_dirs=library_dirs,
            libraries=libraries,
            extra_compile_args=["-O2", "-std=c++17"],
        )
    ]


setup(
    name="oil-gas-ops-prospect",
    version="0.1.0",
    description="AscendC and Triton-Ascend custom operators for oil & gas exploration",
    packages=[
        "oil_gas_ops_prospect",
        "oil_gas_ops_prospect.ascendc",
        "oil_gas_ops_prospect.triton",
        "oil_gas_ops_prospect.triton.real_fft",
    ],
    package_dir={
        "oil_gas_ops_prospect": "python/oil_gas_ops_prospect",
        "oil_gas_ops_prospect.ascendc": "python/oil_gas_ops_prospect/ascendc",
        "oil_gas_ops_prospect.triton": "python/oil_gas_ops_prospect/triton",
        "oil_gas_ops_prospect.triton.real_fft": "triton-ascend/real_fft",
    },
    ext_modules=_complex_mul_extension() + _fused_bias_softmax_extension(),
    cmdclass={"build_ext": BuildExtension},
)
