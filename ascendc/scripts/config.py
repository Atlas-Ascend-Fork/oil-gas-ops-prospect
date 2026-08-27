# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""custom_multi_op_gen 公共配置与路径。"""

import json
import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = ROOT_DIR.parent
WORK_ROOT = ROOT_DIR / "_work"
MANIFEST_PATH = ROOT_DIR / "operators" / "manifest.tsv"
OPERATORS_DIR = ROOT_DIR / "operators"

# 所有编译产物统一落到仓库根目录 output/：
#   output/<SOC>/oil_gas_ops_prospect_opp_<os>_<arch>.run   OPP 算子包
#   output/wheels/oil_gas_ops_prospect-*.whl                Python whl
# 可用 OIL_GAS_OPS_OUTPUT_DIR 覆盖（CI 里把产物导到别处时用）。
OUTPUT_DIR = Path(os.environ.get("OIL_GAS_OPS_OUTPUT_DIR") or (REPO_ROOT / "output")).resolve()
WHEEL_DIR = OUTPUT_DIR / "wheels"

# 跨算子共用的 host/kernel 头（不是算子，不在 manifest 里）：
#   _common/op_host/*.h   → 平铺进 CustomOp/op_host/
#   _common/op_kernel/*.h → 拷进每个 CustomOp/op_kernel/<op>/
COMMON_DIR = OPERATORS_DIR / "_common"

DEFAULT_VENDOR_NAME = "customize"
DEFAULT_PACKAGE_NAME = "oil_gas_ops_prospect"

# msopgen 脚手架用占位算子 IR（合并真实算子前会被 dummy 源文件覆盖/删除）
DUMMY_OP_IR = [
    {
        "op": "DummyOp",
        "input_desc": [
            {
                "name": "x",
                "param_type": "required",
                "format": ["ND"],
                "type": ["float16"],
            }
        ],
        "output_desc": [
            {
                "name": "y",
                "param_type": "required",
                "format": ["ND"],
                "type": ["float16"],
            }
        ],
    }
]


def write_dummy_op_ir_json(path: Path | None = None) -> Path:
    """写入 msopgen -i 所需的 JSON，默认放在 _work/ 下（已 gitignore）。"""
    out = path or (WORK_ROOT / ".msopgen_dummy.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(DUMMY_OP_IR, indent=4) + "\n", encoding="utf-8")
    return out


# 与 install.sh 一致的 msopgen SOC 列表
SUPPORTED_SOC_VERSIONS = (
    "Ascend910A",
    "Ascend910B",
    "Ascend310B1",
    "Ascend310B2",
    "Ascend310B3",
    "Ascend310B4",
    "Ascend310P1",
    "Ascend310P3",
    "Ascend910B1",
    "Ascend910B2",
    "Ascend910B3",
    "Ascend910B4",
)

# msopgen 生成值 → 与样例 AddConfig 对齐
SOC_COMPUTE_OVERRIDE = {
    "Ascend910B": "ascend910b",
}

KERNEL_CMAKE_HEADER = """\
# Multi-operator overlay (custom_multi_op_gen) — 覆盖 msopgen 单算子 op_kernel/CMakeLists.txt

if("${CMAKE_BUILD_TYPE}x" STREQUAL "Debugx")
    npu_op_kernel_options(ascendc_kernels ALL OPTIONS -g -O0)
endif()

"""

KERNEL_CMAKE_FOOTER = """\
npu_op_kernel_library(ascendc_kernels
    SRC_BASE ${CMAKE_CURRENT_SOURCE_DIR}
    TILING_LIBRARY cust_optiling
)

npu_op_package_add(${package_name} LIBRARY ascendc_kernels)
"""

DUMMY_HOST_FILES = ("dummy_op.cpp",)
DUMMY_KERNEL_FILES = ("dummy_op.cpp", "dummy_op_tiling.h")
