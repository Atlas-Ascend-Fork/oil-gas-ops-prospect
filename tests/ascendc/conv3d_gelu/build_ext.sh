#!/bin/bash
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

# 只编译 conv3d_gelu 的 pybind，不链接其它算子的 aclnn 符号。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ASCEND="${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.0.0}"
export ASCEND_HOME_PATH="${ASCEND}"
export LD_LIBRARY_PATH="${ASCEND}/opp/vendors/customize/op_api/lib:${ASCEND}/lib64:${LD_LIBRARY_PATH:-}"
cd "${ROOT}/ascendc"
python3 - <<'PY'
import os
from pathlib import Path
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension
import torch
import torch_npu

root = Path(".").resolve()
ascend = os.environ["ASCEND_HOME_PATH"]
vendor = Path(ascend) / "opp" / "vendors" / "customize" / "op_api"
tn = Path(torch_npu.__file__).parent
setup(
    name="conv3d_gelu_npu_ext",
    ext_modules=[
        CppExtension(
            name="conv3d_gelu_npu",
            sources=["pybind/csrc/conv3d_gelu/bindings.cpp"],
            include_dirs=[
                str(tn / "include"),
                str(tn / "include" / "third_party" / "acl" / "inc"),
                str(tn / "include" / "third_party" / "op-plugin"),
                f"{ascend}/include",
                str(vendor / "include"),
            ],
            library_dirs=[
                f"{ascend}/lib64",
                str(vendor / "lib"),
                str(tn / "lib"),
                str(Path(torch.__file__).parent / "lib"),
            ],
            libraries=["cust_opapi", "ascendcl", "nnopbase", "opapi", "torch_npu"],
            extra_compile_args=["-O2", "-std=c++17"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    script_args=["build_ext", "--inplace"],
)
PY
echo "extension built under ${ROOT}/ascendc"
