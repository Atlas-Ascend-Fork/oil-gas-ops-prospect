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

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

_SCRIPTS = Path(__file__).resolve().parents[3] / "_common" / "aclnn" / "scripts"
sys.path.insert(0, str(_SCRIPTS))
from shape_args import add_shape_argument, parse_ints, write_shape  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    add_shape_argument(parser, (8, 64))
    args = parser.parse_args()
    shape = parse_ints(args.shape)
    rng = np.random.default_rng(0)
    ur = rng.uniform(-1, 1, shape).astype(np.float32)
    ui = rng.uniform(-1, 1, shape).astype(np.float32)
    vr = rng.uniform(-1, 1, shape).astype(np.float32)
    vi = rng.uniform(-1, 1, shape).astype(np.float32)
    wr = ur * vr - ui * vi
    wi = ur * vi + ui * vr
    os.makedirs("input", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    write_shape("input/shape.txt", shape)
    ur.tofile("input/input_ur.bin")
    ui.tofile("input/input_ui.bin")
    vr.tofile("input/input_vr.bin")
    vi.tofile("input/input_vi.bin")
    wr.tofile("output/golden_wr.bin")
    wi.tofile("output/golden_wi.bin")


if __name__ == "__main__":
    main()
