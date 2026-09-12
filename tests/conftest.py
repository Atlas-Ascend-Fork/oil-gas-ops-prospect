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

"""pytest 收集配置。

`tests/ascendc/*/test_*_npu.py` 名字虽然以 `test_` 开头，但它们是带 argparse 的
CLI 脚本（入口是各目录的 `run_test.sh`），不含 pytest 用例，且要求真实 NPU
设备。把它们排除在收集之外，`pytest tests/` 才能干净地跑完无卡用例。

另外：源码树没有 `oil_gas_ops_prospect/` 目录，收集期先挂上源码映射，
Triton / AscendC 的 Python 用例才能 `import oil_gas_ops_prospect`。
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("local_pkg", _REPO_ROOT / "scripts" / "local_pkg.py")
if _spec is not None and _spec.loader is not None:
    _local_pkg = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_local_pkg)
    _local_pkg.activate(_REPO_ROOT)

collect_ignore = [
    "ascendc/complex_mul/test_complex_mul_npu.py",
    "ascendc/fused_bias_softmax/test_fused_bias_softmax_npu.py",
]
