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

"""ROIAlignRotated ST 数据生成：input/rois + 纯 torch 参考 golden。"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np

_SCRIPTS = Path(__file__).resolve().parents[3] / "_common" / "aclnn" / "scripts"
sys.path.insert(0, str(_SCRIPTS))
from shape_args import add_shape_argument, parse_ints, write_shape  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[4]
_PKG = REPO_ROOT / "scripts" / "local_pkg.py"
_spec = importlib.util.spec_from_file_location("local_pkg", _PKG)
if _spec is not None and _spec.loader is not None:
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    _mod.activate(REPO_ROOT)

import torch  # noqa: E402

from oil_gas_ops_prospect.roi_align_rotated import torch_roi_align_rotated  # noqa: E402


def _parse_floats(text: str) -> list[float]:
    return [float(part) for part in text.replace(" ", ",").split(",") if part]


def make_rois(n: int, h: int, w: int, k: int, angles: list[float], scale: float) -> np.ndarray:
    rows = []
    for i in range(k):
        batch = i % max(n, 1)
        cx = (w - 1) * 0.5 + ((i % 3) - 1) * 2.0
        cy = (h - 1) * 0.5 + ((i % 4) - 2) * 2.0
        rw = max(w * 0.6 + (i % 5), 1.0)
        rh = max(h * 0.55 + (i % 3), 1.0)
        rows.append([batch, cx, cy, rw, rh, angles[i % len(angles)]])
    return np.asarray(rows, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    add_shape_argument(parser, (1, 1, 5, 5))
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--ph", type=int, default=4)
    parser.add_argument("--pw", type=int, default=4)
    parser.add_argument("--sr", type=int, default=0)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--angles", default="0.0")
    args = parser.parse_args()

    shape = parse_ints(args.shape)
    if len(shape) != 4:
        raise ValueError(f"input shape must be 4D, got {shape}")
    n, c, h, w = shape
    angles = _parse_floats(args.angles)
    if not angles:
        raise ValueError("angles must not be empty")

    rng = np.random.default_rng(0)
    inp = rng.standard_normal(shape).astype(np.float32)
    rois = make_rois(n, h, w, args.k, angles, args.scale)

    golden = torch_roi_align_rotated(
        torch.from_numpy(inp), torch.from_numpy(rois), (args.ph, args.pw), args.scale, args.sr
    ).numpy()

    os.makedirs("input", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    write_shape("input/shape.txt", shape)
    write_shape("input/rois_shape.txt", rois.shape)
    Path("input/attrs.txt").write_text(f"{args.ph} {args.pw} {args.sr} {args.scale}\n")
    inp.tofile("input/input.bin")
    rois.tofile("input/rois.bin")
    golden.astype(np.float32).tofile("output/golden.bin")
    print(f"[INFO] gen_data: input={shape} k={args.k} pool={args.ph}x{args.pw} sr={args.sr} scale={args.scale}")


if __name__ == "__main__":
    main()
