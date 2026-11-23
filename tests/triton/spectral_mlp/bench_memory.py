#!/usr/bin/env python3
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

"""训练单步峰值显存的单路径测量（ref = 原生 einsum + npu_gelu，op = 本算子）。

进程内 `max_memory_allocated` 会被 allocator 复用和常驻张量污染，所以一次只测
一条路径、每条路径单独起进程：

    python3 tests/triton/spectral_mlp/bench_memory.py --mode ref
    python3 tests/triton/spectral_mlp/bench_memory.py --mode op

两条命令各起一个进程，比较输出的 `train_peak_mb`。
"""

from __future__ import annotations

import argparse
import sys

import torch

import _bootstrap  # noqa: F401  (side effect: 源码树里解析 oil_gas_ops_prospect)
from oil_gas_ops_prospect.spectral_mlp import spectral_mlp

from verify_spectral_mlp import _business_shape, _make_inputs, _model_gelu, _reference


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("ref", "op"), required=True)
    ap.add_argument("--img", type=int, default=128)
    ap.add_argument("--modes", type=int, default=64)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--groups", type=int, default=8)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--device-id", type=int, default=0)
    args = ap.parse_args(argv)

    try:
        import torch_npu  # noqa: F401
        device = torch.device(f"npu:{args.device_id}") if torch.npu.is_available() else torch.device("cpu")
    except ImportError:
        device = torch.device("cpu")

    shape = _business_shape(args)
    base = _make_inputs(*shape, device=device)
    leaves = [t.detach().requires_grad_(True) for t in base]

    if args.mode == "ref":
        def step():
            o = _reference(*leaves, act=_model_gelu)
            torch.autograd.grad(o[0].sum() + o[1].sum(), leaves)
    else:
        def step():
            o = spectral_mlp(*leaves)
            torch.autograd.grad(o[0].sum() + o[1].sum(), leaves)

    step()  # JIT 预热
    if device.type == "npu":
        torch.npu.synchronize()
        torch.npu.empty_cache()
        torch.npu.reset_peak_memory_stats()
        step()
        torch.npu.synchronize()
        peak = torch.npu.max_memory_allocated() / (1 << 20)
    else:
        peak = 0.0

    print(f"[memory] mode={args.mode} shape={shape} train_peak_mb={peak:.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
