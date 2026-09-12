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

"""Make the examples runnable from a source checkout without installing the wheel.

``python3 examples/foo.py`` puts ``examples/`` on ``sys.path``，源码树也没有
``oil_gas_ops_prospect/`` 目录（该导入名由 ``setup.py`` 在打包时组装）。导入本模块
会挂上源码映射，使 ``from oil_gas_ops_prospect.complex_mul import complex_mul`` 可用。

必须在任何 ``oil_gas_ops_prospect`` 导入之前执行::

    import _bootstrap  # noqa: F401
    from oil_gas_ops_prospect.real_fft import real_rfft

置 ``OIL_GAS_OPS_WHL_TEST=1`` 时为 no-op，``USING_LOCAL_SOURCES`` 为 False。
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _activate() -> bool:
    spec = importlib.util.spec_from_file_location(
        "local_pkg", REPO_ROOT / "scripts" / "local_pkg.py"
    )
    if spec is None or spec.loader is None:
        return False
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return bool(module.activate(REPO_ROOT))


USING_LOCAL_SOURCES = _activate()
