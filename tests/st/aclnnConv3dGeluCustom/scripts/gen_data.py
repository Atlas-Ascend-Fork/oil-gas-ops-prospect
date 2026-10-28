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

"""生成 Conv3DGelu ST 输入和 CPU float32 tanh-GELU(conv3d) 真值。"""

from __future__ import annotations

import argparse
import os

import numpy as np

GELU_SCALE = 1.5957691216057308
GELU_CUBE = 0.044715


def parse_ints(text: str, allow_zero: bool = True) -> tuple[int, ...]:
    dims = tuple(int(part) for part in text.replace("x", ",").split(",") if part)
    if not dims or any(dim < 0 for dim in dims) or (not allow_zero and any(dim == 0 for dim in dims)):
        raise ValueError(f"invalid ints: {text}")
    return dims


def conv3d(x: np.ndarray, weight: np.ndarray, bias: np.ndarray | None, stride: tuple[int, int, int],
           padding: tuple[int, int, int], dilation: tuple[int, int, int]) -> np.ndarray:
    """NCDHW conv3d，groups=1。用 float64 累加后再落到 float32。"""
    pd, ph, pw = padding
    if (pd, ph, pw) != (0, 0, 0):
        x = np.pad(x, ((0, 0), (0, 0), (pd, pd), (ph, ph), (pw, pw)))
    n, _, depth, height, width = x.shape
    cout, cin, kd, kh, kw = weight.shape
    sd, sh, sw = stride
    dd, dh, dw = dilation
    od = (depth - dd * (kd - 1) - 1) // sd + 1
    oh = (height - dh * (kh - 1) - 1) // sh + 1
    ow = (width - dw * (kw - 1) - 1) // sw + 1
    y = np.zeros((n, cout, od, oh, ow), dtype=np.float64)
    for oc in range(cout):
        acc = np.zeros((n, od, oh, ow), dtype=np.float64)
        for ic in range(cin):
            for kz in range(kd):
                for ky in range(kh):
                    for kx in range(kw):
                        z0, y0, x0 = kz * dd, ky * dh, kx * dw
                        patch = x[:, ic, z0:z0 + sd * od:sd, y0:y0 + sh * oh:sh, x0:x0 + sw * ow:sw]
                        acc += patch * float(weight[oc, ic, kz, ky, kx])
        if bias is not None:
            acc += float(bias[oc])
        y[:, oc] = acc
    return y.astype(np.float32)


def gelu_tanh(x: np.ndarray) -> np.ndarray:
    cube = x.astype(np.float64)
    inner = GELU_SCALE * (cube + GELU_CUBE * cube ** 3)
    return (cube / (1.0 + np.exp(-inner))).astype(np.float32)


def write_shape(path: str, shape: tuple[int, ...]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(" ".join(str(dim) for dim in shape) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", required=True)
    parser.add_argument("--weight", required=True)
    parser.add_argument("--stride", default="1,1,1")
    parser.add_argument("--padding", default="0,0,0")
    parser.add_argument("--dilation", default="1,1,1")
    parser.add_argument("--bias", type=int, default=1)
    args = parser.parse_args()

    x_shape = parse_ints(args.shape, allow_zero=False)
    w_shape = parse_ints(args.weight, allow_zero=False)
    stride = parse_ints(args.stride)
    padding = parse_ints(args.padding)
    dilation = parse_ints(args.dilation)
    if len(x_shape) != 5 or len(w_shape) != 5 or len(stride) != 3 or len(padding) != 3 or len(dilation) != 3:
        raise ValueError("x/weight must be 5D and stride/padding/dilation must be 3D")
    if w_shape[1] != x_shape[1]:
        raise ValueError("weight Cin must match x Cin")

    rng = np.random.default_rng(7)
    x = rng.normal(0.0, 0.2, x_shape).astype(np.float32)
    weight = rng.normal(0.0, 0.2, w_shape).astype(np.float32)
    bias = rng.normal(0.0, 0.1, (w_shape[0],)).astype(np.float32) if args.bias else None
    y = gelu_tanh(conv3d(x, weight, bias, stride, padding, dilation))

    os.makedirs("input", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    write_shape("input/shape_x.txt", x_shape)
    write_shape("input/shape_w.txt", w_shape)
    write_shape("input/shape_y.txt", tuple(int(dim) for dim in y.shape))
    with open("input/conv_attr.txt", "w", encoding="utf-8") as handle:
        handle.write(" ".join(str(v) for v in (*stride, *padding, *dilation, args.bias)) + "\n")
    x.tofile("input/input_x.bin")
    weight.tofile("input/input_weight.bin")
    bias_path = "input/input_bias.bin"
    if bias is None:
        if os.path.exists(bias_path):
            os.remove(bias_path)
    else:
        bias.tofile(bias_path)
    y.tofile("output/golden.bin")


if __name__ == "__main__":
    main()
