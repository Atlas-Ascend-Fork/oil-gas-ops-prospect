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


def softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - x.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return (exp / exp.sum(axis=-1, keepdims=True)).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    add_shape_argument(parser, (2, 4, 64, 64), extra=True)
    args = parser.parse_args()
    shape = parse_ints(args.shape)
    if len(shape) != 4:
        raise ValueError(f"fused_bias_softmax shape must be 4D, got {shape}")
    rng = np.random.default_rng(2)
    x = rng.normal(0.0, 1.0, shape).astype(np.float32)
    logits = x
    os.makedirs("input", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    write_shape("input/shape.txt", shape)
    x.tofile("input/input_x.bin")
    bias_path = Path("input/input_bias1.bin")
    if bias_path.exists():
        bias_path.unlink()
    if args.bias1 == "broadcast":
        bias_shape = (shape[0], shape[1], 1, shape[3])
        bias1 = rng.normal(0.0, 1.0, bias_shape).astype(np.float32)
        bias1.tofile("input/input_bias1.bin")
        logits = x + bias1
    elif args.bias1 == "full":
        bias1 = rng.normal(0.0, 1.0, shape).astype(np.float32)
        bias1.tofile("input/input_bias1.bin")
        logits = x + bias1
    softmax(logits).tofile("output/golden.bin")


if __name__ == "__main__":
    main()
