# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Import helpers for running Triton-Ascend cases from the source tree."""

from __future__ import annotations

import os
import sys
from importlib import import_module
from pathlib import Path


TRITON_SOURCE_ENV = "OIL_GAS_OPS_TRITON_SOURCE"
_LOCAL_MODES = {"local", "source", "sources", "file", "files"}
_WHEEL_MODES = {"whl", "wheel", "package", "installed"}

MODEL_REPO_ENV = "OIL_GAS_OPS_MODEL_REPO"


def _prepend_sys_path(path: Path) -> None:
    path_str = str(path)
    if path_str in sys.path:
        sys.path.remove(path_str)
    sys.path.insert(0, path_str)


def _prepend_package_path(package, path: Path) -> None:
    path_str = str(path)
    existing = list(getattr(package, "__path__", []))
    package.__path__ = [path_str] + [item for item in existing if item != path_str]


def _source_mode() -> str:
    raw_mode = os.environ.get(TRITON_SOURCE_ENV)
    if raw_mode is None or raw_mode.strip() == "":
        return "local"

    raw_mode = raw_mode.strip().lower()
    if raw_mode in _LOCAL_MODES:
        return "local"
    if raw_mode in _WHEEL_MODES:
        return "whl"
    valid = "local|source|file or whl|wheel|installed"
    raise ValueError(f"{TRITON_SOURCE_ENV}={raw_mode!r} is invalid; expected {valid}")


def use_local_triton_sources() -> bool:
    """Use local Triton sources unless the environment explicitly selects whl."""

    if _source_mode() == "whl":
        return False

    triton_root = Path(__file__).resolve().parent
    ops_root = triton_root.parent
    python_root = ops_root / "python"

    _prepend_sys_path(python_root)
    _prepend_sys_path(ops_root)

    oil_gas_ops_prospect = import_module("oil_gas_ops_prospect")
    triton_pkg = import_module("oil_gas_ops_prospect.triton")

    _prepend_package_path(oil_gas_ops_prospect, python_root / "oil_gas_ops_prospect")
    _prepend_package_path(triton_pkg, triton_root)
    return True


def ops_repo_root() -> Path:
    """Root of this operator repository (`oil-gas-ops-prospect`)."""

    return Path(__file__).resolve().parent.parent


def find_model_repo_root() -> Path | None:
    """Root of a downstream model repository, or None when running standalone.

    This operator library is domain-scoped, not project-scoped: it knows nothing
    about any particular model repository layout. A downstream project that wants
    the optional cross-check cases (comparing a fused operator against the model
    code it replaces) opts in explicitly::

        export OIL_GAS_OPS_MODEL_REPO=/path/to/model-repo

    Returns ``None`` when unset, which is the normal standalone case. Callers must
    degrade gracefully (skip the case, or fall back to explicit shapes).
    """

    env_root = os.environ.get(MODEL_REPO_ENV)
    if not env_root or not env_root.strip():
        return None

    root = Path(env_root).expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"{MODEL_REPO_ENV}={env_root!r} is not a directory")
    return root


def require_model_repo_root(reason: str) -> Path:
    """Like :func:`find_model_repo_root`, but raise an actionable error."""

    root = find_model_repo_root()
    if root is None:
        raise SystemExit(
            f"{reason} requires a downstream model repository.\n"
            f"This operator repository ({ops_repo_root()}) is standalone; opt in with\n"
            f"  export {MODEL_REPO_ENV}=/path/to/model-repo\n"
            f"or use the standalone options instead (e.g. --shape / --size)."
        )
    return root
