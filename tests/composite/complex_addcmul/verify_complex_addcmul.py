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

"""complex_addcmul 的 npu_python 套件入口（`--mode all|precision|perf`）。

`tests/suites.json` 登记本脚本后，`bash build.sh -u --precision` /
`--perf` 会用 `--mode <mode>` 调用它。参考实现取 CPU 原生 `torch.addcmul`。

逐 case 判定与退出码绑定：任一 case 超阈值即返回非 0，套件据此判 FAIL。

用法::

    python3 verify_complex_addcmul.py --mode precision
    python3 verify_complex_addcmul.py --mode perf
    python3 verify_complex_addcmul.py --mode all
    python3 verify_complex_addcmul.py --mode precision --device cpu --rtol 1e-5
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import torch

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

_spec = importlib.util.spec_from_file_location(
    "local_pkg", REPO_ROOT / "scripts" / "local_pkg.py"
)
if _spec is not None and _spec.loader is not None:
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    _mod.activate(REPO_ROOT)

from oil_gas_ops_prospect.complex_addcmul import complex_addcmul  # noqa: E402

# (input, tensor1, tensor2) —— 覆盖同 shape、常见广播与 4D 体数据形态。
CASES = [
    ((2, 3), (2, 3), (2, 3)),
    ((2, 3), (1, 3), (2, 3)),
    ((2, 3), (2, 3), (2, 1)),
    ((2, 3), (1, 1), (1, 1)),
    ((3, 1, 5), (3, 4, 5), (1, 4, 1)),
    ((2, 4, 32, 32), (2, 4, 32, 32), (2, 4, 32, 32)),
]
DTYPES = (torch.complex64, torch.complex128)
VALUES = (1.0, 2.5, 1 + 2j)

# perf 模式使用的较大形态（单样本 4D）。
PERF_SHAPE = (1, 16, 128, 128)


def _real_dtype(dtype):
    return torch.float32 if dtype == torch.complex64 else torch.float64


def _make(shape, dtype, device, requires_grad=False):
    rd = _real_dtype(dtype)
    t = torch.complex(torch.randn(shape, dtype=rd), torch.randn(shape, dtype=rd))
    t = t.to(device)
    return t.requires_grad_(requires_grad) if requires_grad else t


def _tol(dtype):
    # 与文档声明的精度契约一致：complex64 ≤ 1e-5、complex128 ≤ 2e-14。
    # 实测上界（6 组 shape × 3 个 value × 40 个 seed，CPU / NPU 各一轮）：
    # complex64 7.63e-6（CPU）/ 4.86e-6（NPU），complex128 1.42e-14 / 1.47e-14；
    # 误差量级约为 eps × |out|（|out| 最大约 3.4e1），故取上述绝对阈值。
    return 1e-5 if dtype == torch.complex64 else 2e-14


def check_precision(device, rtol) -> tuple[list[float], list[str]]:
    """前向 + 反向逐 case 对拍 CPU 原生 torch.addcmul。

    返回 ``(误差列表, 失败描述列表)``；失败列表非空即代表有 case 超阈值。
    """
    worst: list[float] = []
    failures: list[str] = []

    def _record(kind, dtype, shapes, value, err, limit):
        worst.append(err)
        if err > limit:
            failures.append(
                f"{kind} {dtype} shapes={shapes} value={value} "
                f"max|Δ|={err:.3e} > {limit:.3e}"
            )
        print(
            f"[precision] {kind} {dtype} shapes={shapes} value={value} "
            f"max|Δ|={err:.3e} {'ok' if err <= limit else 'FAIL'}"
        )

    for shapes in CASES:
        for dtype in DTYPES:
            for value in VALUES:
                torch.manual_seed(0)
                s1, s2, s3 = shapes
                base = [_make(s, dtype, device) for s in (s1, s2, s3)]
                ref = torch.addcmul(*(t.cpu() for t in base), value=value)
                got = complex_addcmul(*base, value=value)
                if device == "npu":
                    torch.npu.synchronize()
                err = float((got.cpu() - ref).abs().max())
                _record(
                    "fwd", dtype, shapes, value, err,
                    max(_tol(dtype), rtol * float(ref.abs().max())),
                )

                out_shape = torch.broadcast_shapes(s1, s2, s3)
                seed = _make(out_shape, dtype, device)
                cin = [
                    t.cpu().clone().requires_grad_(True) for t in base
                ]
                cout = torch.addcmul(*cin, value=value)
                cgrads = torch.autograd.grad(cout, cin, seed.cpu())
                nin = [t.clone().requires_grad_(True) for t in base]
                nout = complex_addcmul(*nin, value=value)
                ngrads = torch.autograd.grad(nout, nin, seed)
                if device == "npu":
                    torch.npu.synchronize()
                for g, w in zip(ngrads, cgrads):
                    gerr = float((g.cpu() - w).abs().max())
                    _record(
                        "bwd", dtype, shapes, value, gerr,
                        max(_tol(dtype), rtol * float(w.abs().max())),
                    )
    return worst, failures


def run_perf(device):
    """输出机器可读的 [perf] 行（键名与开源报告解析器一致）。

    标杆取同设备等价 torch 表达式 ``input + value * t1 * t2``——NPU 上原生
    ``torch.addcmul`` 对复数直接报错，这条表达式就是调用方在没有本算子时的
    替代写法。本算子把复数乘交给 `complex_mul`（NPU + ascendc 时可落 AscendC），
    但其余乘加仍是多次 torch 小算子，因此加速比接近 1 属预期。
    """
    dtype = torch.complex64
    torch.manual_seed(0)
    args = [_make(PERF_SHAPE, dtype, device) for _ in range(3)]

    def bench(fn, warm=3, iters=10):
        for _ in range(warm):
            fn()
        if device == "npu":
            torch.npu.synchronize()
        start = time.perf_counter()
        for _ in range(iters):
            fn()
        if device == "npu":
            torch.npu.synchronize()
        return (time.perf_counter() - start) / iters * 1e3

    torch_ms = bench(lambda: args[0] + 1.0 * args[1] * args[2])
    custom_ms = bench(lambda: complex_addcmul(*args))
    speedup = torch_ms / custom_ms if custom_ms > 0 else 0.0
    tag = "x".join(str(d) for d in PERF_SHAPE)
    print(
        f"[perf] addcmul fwd [{tag}] torch={torch_ms:.3f}ms "
        f"ascendc={custom_ms:.3f}ms speedup={speedup:.2f}x"
    )


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--mode", choices=("all", "precision", "perf"), default="all"
    )
    p.add_argument(
        "--device", default="npu", help="npu（默认）或 cpu，cpu 用于脚本冒烟"
    )
    p.add_argument(
        "--rtol",
        type=float,
        default=0.0,
        help="相对误差放宽系数；默认 0，即只用文档声明的绝对阈值",
    )
    # tests/suites.json 的 npu_python 入口会给未构建的算子追加 --build-ext。
    # 本算子纯 Python 组合、无自定义扩展，接收但不做任何事，保持协议兼容。
    p.add_argument(
        "--build-ext",
        action="store_true",
        help="兼容 npu_python 套件协议；本算子无自定义扩展，忽略此参数",
    )
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    device = args.device
    if device == "npu":
        if not (hasattr(torch, "npu") and torch.npu.is_available()):
            print("[SKIP] 未检测到 NPU")
            return 1

    if args.mode in ("perf", "all"):
        run_perf(device)
        if args.mode == "perf":
            return 0

    worst, failures = check_precision(device, args.rtol)
    peak = max(worst) if worst else 0.0
    if failures:
        print(
            f"FAIL {len(failures)} 个 case 超过逐 case 阈值，"
            f"最大绝对误差 {peak:.3e}"
        )
        for item in failures[:10]:
            print(f"  - {item}")
        return 1
    print(f"PASS 最大绝对误差 {peak:.3e}（逐 case 阈值均已满足）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
