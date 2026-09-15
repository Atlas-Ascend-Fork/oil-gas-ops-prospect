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
    add_shape_argument(parser, (2, 4, 64, 64, 64))
    args = parser.parse_args()
    dims = parse_ints(args.shape)
    if len(dims) == 4:
        batch, heads, query, key = dims
        value_dim = key
    elif len(dims) == 5:
        batch, heads, query, key, value_dim = dims
    else:
        raise ValueError(f"fused_softmax_grad shape must be B,H,Q,S[,D], got {dims}")
    probs_shape = (batch, heads, query, key)
    go_shape = (batch, heads, query, value_dim)
    value_shape = (batch, heads, key, value_dim)
    rng = np.random.default_rng(3)
    logits = rng.normal(0.0, 1.0, probs_shape).astype(np.float32)
    probs = softmax(logits)
    go = rng.normal(0.0, 1.0, go_shape).astype(np.float32)
    values = rng.normal(0.0, 1.0, value_shape).astype(np.float32)
    # g = go @ V^T → [B,H,Q,D] @ [B,H,D,S] = [B,H,Q,S]
    grad = np.matmul(go, np.swapaxes(values, -1, -2))
    golden = (probs * (grad - (grad * probs).sum(axis=-1, keepdims=True))).astype(np.float32)
    os.makedirs("input", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    write_shape("input/shape.txt", (batch, heads, query, key, value_dim))
    probs.tofile("input/input_probs.bin")
    go.tofile("input/input_go.bin")
    values.tofile("input/input_v.bin")
    golden.tofile("output/golden.bin")


if __name__ == "__main__":
    main()
