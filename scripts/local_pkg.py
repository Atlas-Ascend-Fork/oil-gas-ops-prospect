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

"""从源码树解析 ``oil_gas_ops_prospect``，无需仓库根下存在同名目录。

whl 由 ``setup.py`` 的 ``package_dir`` 按算子名组装：

- ``oil_gas_ops_prospect`` → ``ascendc/pybind/``（``complex_mul.py``、``fused_bias_softmax.py``）
- ``oil_gas_ops_prospect.real_fft`` → ``triton-ascend/real_fft/``
- ``oil_gas_ops_prospect.complex_reciprocal`` → ``triton-ascend/complex_reciprocal/``
- ``oil_gas_ops_prospect.<op>`` → ``composite-torch/<op>/``（``_composite_mapping()`` 自动发现）

没装 whl 时仓内没有物理包目录，本模块在 ``sys.meta_path`` 上挂一个 finder，
让 ``from oil_gas_ops_prospect.complex_mul import complex_mul`` 等导入与装包后一致。

置 ``OIL_GAS_OPS_WHL_TEST=1`` 时不激活，避免挡住「验证已安装 whl」的用例。
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

_PKG = "oil_gas_ops_prospect"
_FINDER_ATTR = "_oil_gas_ops_local_finder"


class _SourceTreeFinder:
    """把虚包名映射到源码目录。"""

    def __init__(self, root_pkg: Path, mapping: dict[str, Path]) -> None:
        self.root_pkg = root_pkg
        self.mapping = mapping

    def find_spec(self, fullname: str, path, target=None):  # noqa: ANN001
        loc = self.mapping.get(fullname)
        if loc is not None:
            if loc.is_file() and loc.suffix == ".py":
                return importlib.util.spec_from_file_location(fullname, loc)
            init = loc / "__init__.py"
            if init.is_file():
                return importlib.util.spec_from_file_location(
                    fullname,
                    init,
                    submodule_search_locations=[str(loc)],
                )
            return None

        if fullname == _PKG:
            init = self.root_pkg / "__init__.py"
            if not init.is_file():
                return None
            return importlib.util.spec_from_file_location(
                fullname,
                init,
                submodule_search_locations=[str(self.root_pkg)],
            )

        for prefix, root in self.mapping.items():
            if not fullname.startswith(prefix + ".") or not root.is_dir():
                continue
            rest = fullname[len(prefix) + 1:]
            parts = rest.split(".")
            pkg_init = root.joinpath(*parts) / "__init__.py"
            if pkg_init.is_file():
                return importlib.util.spec_from_file_location(
                    fullname,
                    pkg_init,
                    submodule_search_locations=[str(pkg_init.parent)],
                )
            module_py = root.joinpath(*parts).with_suffix(".py")
            if module_py.is_file():
                return importlib.util.spec_from_file_location(fullname, module_py)
        return None


def _composite_mapping(composite_root: Path) -> dict[str, Path]:
    """自动发现 ``composite-torch/`` 下的算子：子目录含 ``__init__.py``，或顶层单文件 ``.py``。

    ``_`` 前缀的条目（如 ``_common/``）与 ``README.md`` 等非算子文件跳过。
    """
    mapping: dict[str, Path] = {}
    if not composite_root.is_dir():
        return mapping
    for entry in sorted(composite_root.iterdir()):
        if entry.name.startswith("_"):
            continue
        if entry.is_dir():
            if (entry / "__init__.py").is_file():
                mapping[f"{_PKG}.{entry.name}"] = entry
        elif entry.is_file() and entry.suffix == ".py":
            mapping[f"{_PKG}.{entry.stem}"] = entry
    return mapping


def activate(repo_root: Path | None = None) -> bool:
    """把源码树挂到 ``oil_gas_ops_prospect`` 导入名下。已激活则直接返回 True。

    同时处理 torch 的设备后端自动加载（``TORCH_DEVICE_BACKEND_AUTOLOAD``）：
    LLT 等无卡镜像可能装有 torch_npu 但缺 CANN 运行环境，torch 2.7+ 默认开启的
    自动加载会在 ``import torch`` / 访问 ``torch.npu`` 时抛
    ``RuntimeError: Failed to load the backend extension: torch_npu``。因此先
    关闭自动加载（显式置 ``1`` 可覆盖，setdefault 语义），再**显式尽早**导入
    torch_npu：NPU 环境下 autograd 引擎按首次执行时已注册的设备初始化 ready
    queues，torch_npu 晚于首次 autograd 注册会触发 ``device_ready_queues_``
    内部断言；导入失败（无卡 / 缺 CANN）则静默跳过，CPU 用例照常运行。
    whl 验证路径同样受益。
    """
    os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    try:
        import torch_npu  # noqa: F401  尽早注册 NPU 设备（见上）
    except Exception:
        pass
    if os.environ.get("OIL_GAS_OPS_WHL_TEST") == "1":
        return False

    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parent.parent
    pybind = root / "ascendc" / "pybind"
    triton = root / "triton-ascend"
    if not (pybind / "__init__.py").is_file() or not (triton / "real_fft" / "__init__.py").is_file():
        return False

    for existing in sys.meta_path:
        if getattr(existing, _FINDER_ATTR, False):
            return True

    for name in list(sys.modules):
        if name == _PKG or name.startswith(_PKG + "."):
            del sys.modules[name]

    mapping = {
        f"{_PKG}.real_fft": triton / "real_fft",
        f"{_PKG}.complex_reciprocal": triton / "complex_reciprocal",
        f"{_PKG}.interpolate": triton / "interpolate",
        f"{_PKG}.complex_exp": triton / "complex_exp",
    }
    mapping.update(_composite_mapping(root / "composite-torch"))
    finder = _SourceTreeFinder(pybind, mapping)
    setattr(finder, _FINDER_ATTR, True)
    sys.meta_path.insert(0, finder)
    return True
