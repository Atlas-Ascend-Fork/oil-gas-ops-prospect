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

# pyright: reportMissingImports=false

from __future__ import annotations

import sys
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension

ROOT = Path(__file__).resolve().parent

# AscendC 算子的 pybind 扩展定义在 ascendc/pybind_extensions.py。
sys.path.insert(0, str(ROOT / "ascendc"))
from pybind_extensions import build_extensions  # noqa: E402

# 源码树不保留 oil_gas_ops_prospect/ 目录。whl 按算子名组装，不暴露后端：
#   oil_gas_ops_prospect                  ← ascendc/pybind/（complex_mul.py、fused_bias_softmax.py）
#   oil_gas_ops_prospect.real_fft         ← triton-ascend/real_fft/
#   oil_gas_ops_prospect.complex_reciprocal ← triton-ascend/complex_reciprocal/
#   oil_gas_ops_prospect.conv3d_custom    ← composite-torch/conv3d_custom/
setup(
    name="oil-gas-ops-prospect",
    version="0.1.0",
    description="AscendC and Triton-Ascend custom operators for oil & gas exploration",
    packages=[
        "oil_gas_ops_prospect",
        "oil_gas_ops_prospect.real_fft",
        "oil_gas_ops_prospect.complex_reciprocal",
        "oil_gas_ops_prospect.conv3d_custom",
    ],
    package_dir={
        "oil_gas_ops_prospect": "ascendc/pybind",
        "oil_gas_ops_prospect.real_fft": "triton-ascend/real_fft",
        "oil_gas_ops_prospect.complex_reciprocal": "triton-ascend/complex_reciprocal",
        "oil_gas_ops_prospect.conv3d_custom": "composite-torch/conv3d_custom",
    },
    include_package_data=False,
    exclude_package_data={
        "oil_gas_ops_prospect": ["csrc*", "csrc/*", "csrc/*/*", "README.md"],
        "oil_gas_ops_prospect.real_fft": ["README.md"],
        "oil_gas_ops_prospect.complex_reciprocal": ["README.md"],
        "oil_gas_ops_prospect.conv3d_custom": ["README.md"],
    },
    ext_modules=build_extensions(relative_to=ROOT),
    cmdclass={"build_ext": BuildExtension},
)
