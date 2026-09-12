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

"""就地编译 AscendC 算子的 pybind 扩展。

    python3 ascendc/pybind_setup.py build_ext --inplace

产物 ``<op_family>_npu*.so`` 落在 ``ascendc/pybind/``，NPU 测试把该目录加进
``sys.path`` 后直接 import。正式交付形态是仓库根 ``setup.py`` 打进 whl。
"""
# pyright: reportMissingImports=false

from __future__ import annotations

import os
import sys
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension

HERE = Path(__file__).resolve().parent
PYBIND = HERE / "pybind"
sys.path.insert(0, str(HERE))

from pybind_extensions import build_extensions  # noqa: E402

# --inplace 把 .so 写到 cwd，切到 pybind/ 以便测试按该目录 import。
os.chdir(PYBIND)

setup(
    name="oil_gas_ops_prospect_ascendc_pybind",
    description="In-place build of the AscendC operator pybind extensions",
    ext_modules=build_extensions(relative_to=PYBIND),
    cmdclass={"build_ext": BuildExtension},
)
