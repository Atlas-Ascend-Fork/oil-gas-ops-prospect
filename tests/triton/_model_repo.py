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

"""Opt-in access to a downstream model repository, for the cross-check cases."""

from __future__ import annotations

import os
from pathlib import Path

MODEL_REPO_ENV = "OIL_GAS_OPS_MODEL_REPO"


def ops_repo_root() -> Path:
    """Root of this operator repository (`oil-gas-ops-prospect`)."""

    return Path(__file__).resolve().parents[2]


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
