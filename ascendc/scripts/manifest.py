# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""读取 operators/manifest.tsv。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from scripts.config import MANIFEST_PATH


@dataclass(frozen=True)
class OperatorEntry:
    dir_name: str
    op_type: str
    kernel_file: str


def load_manifest(manifest_path: Path | None = None) -> list[OperatorEntry]:
    path = manifest_path or MANIFEST_PATH
    entries: list[OperatorEntry] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            raise ValueError(f"manifest 格式错误（需 3 列 tab 分隔）: {line!r} in {path}")
        entries.append(OperatorEntry(parts[0].strip(), parts[1].strip(), parts[2].strip()))
    if not entries:
        raise ValueError(f"manifest 为空: {path}")
    return entries


def lookup_entry(dir_name: str, manifest_path: Path | None = None) -> OperatorEntry:
    for entry in load_manifest(manifest_path):
        if entry.dir_name == dir_name:
            return entry
    raise KeyError(f"算子 '{dir_name}' 未在 manifest 中注册")


def list_operator_names(manifest_path: Path | None = None) -> list[str]:
    return [e.dir_name for e in load_manifest(manifest_path)]
