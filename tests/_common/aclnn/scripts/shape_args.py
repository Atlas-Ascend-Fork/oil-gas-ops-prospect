#!/usr/bin/python3
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

"""ST gen_data 共用：解析 --shape，写 input/shape.txt。"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence


def parse_ints(text: str) -> tuple[int, ...]:
    cleaned = text.replace("x", ",").replace("X", ",").replace(" ", ",")
    dims = tuple(int(part) for part in cleaned.split(",") if part)
    if not dims or any(d <= 0 for d in dims):
        raise ValueError(f"invalid shape: {text}")
    return dims


def add_shape_argument(parser: argparse.ArgumentParser, default: Sequence[int], extra: bool = False) -> None:
    parser.add_argument("--shape", default=",".join(str(d) for d in default), help="comma-separated dims")
    if extra:
        parser.add_argument("--bias1", choices=("none", "broadcast", "full"), default="none")


def write_shape(path: str | Path, shape: Sequence[int]) -> None:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(" ".join(str(d) for d in shape) + "\n")
