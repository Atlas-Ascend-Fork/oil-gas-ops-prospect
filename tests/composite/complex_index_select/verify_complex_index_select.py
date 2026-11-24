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

"""complex_index_select 的 npu_python 套件入口（`--mode all|precision|perf`）。

`tests/suites.json` 登记本脚本后，`bash build.sh -u --precision` /
`--perf` 会用 `--mode <mode>` 调用它。参考实现取 CPU 原生
`torch.index_select`（前向逐位一致；反向 NPU 原生对复数报错，参考亦取 CPU）。

逐 case 判定与退出码绑定：任一 case 超阈值即返回非 0，套件据此判 FAIL。

用法::

    python3 verify_complex_index_select.py --mode precision
    python3 verify_complex_index_select.py --mode perf
    python3 verify_complex_index_select.py --mode all
    python3 verify_complex_index_select.py --mode precision --device cpu --rtol 1e-5
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

from oil_gas_ops_prospect.complex_index_select import complex_index_select  # noqa: E402

# (shape, dim, M, dup, index_dtype) —— 覆盖 flat / real-imag 两条路径、
# 重复索引、int32 索引与空索引。
CASES = [
    ((7, 5), 0, 4, False, torch.int64),        # flat：首轴
    ((3, 50, 5), 1, 31, False, torch.int64),    # flat：中间轴
    ((2, 3, 4, 5), 2, 2, False, torch.int64),   # flat：4-D
    ((77, 33), -1, 12, False, torch.int64),     # real/imag：末轴
    ((100,), 0, 37, False, torch.int64),         # real/imag：1-D
    ((7, 5), 0, 12, True, torch.int64),          # 重复索引（反向累加）
    ((3, 50, 5), 1, 20, False, torch.int32),    # int32 索引
    ((7, 5), 0, 0, False, torch.int64),          # 空索引
]
DTYPES = (torch.complex64, torch.complex128)

# perf 模式使用的形态（910B2 / CANN 9.0.0 / torch_npu 2.7.1 实测口径，
# 与 ISSUE #61 验证报告同源；末尾的 1-D 用例走 real/imag 兜底路径，
# 频率体是 FWI 频率选择业务的代表形态，约 3.4GB 源张量）。
PERF_CASES = [
    ((16384, 256), 0, 8192),        # 中块
    ((65536, 1024), 0, 16384),      # 行聚合大块
    ((2**20, 4), 0, 2**18),        # 长条（flat 路径，行宽 32B）
    ((2**20,), 0, 2**17),          # 1-D（real/imag 路径）
    ((4, 64, 80, 144, 144), 1, 16),  # 频率体（dim=1）
]


def _real_dtype(dtype):
    return torch.float32 if dtype == torch.complex64 else torch.float64


def _make(shape, dtype, device, requires_grad=False):
    rd = _real_dtype(dtype)
    t = torch.complex(torch.randn(shape, dtype=rd), torch.randn(shape, dtype=rd))
    t = t.to(device)
    return t.requires_grad_(requires_grad) if requires_grad else t


def _index_for(shape, dim, m, device, dtype, dup=False):
    if dim < 0:
        dim += len(shape)
    n = shape[dim]
    if dup or m > n:
        g = torch.randint(0, n, (m,), dtype=torch.int64)
    else:
        g = torch.randperm(n)[:m]
    return g.to(dtype).to(device)


def _bwd_tol(dtype):
    # 与文档声明的精度契约一致：反向 complex64 ≤ 1e-5、complex128 ≤ 2e-14。
    # 910B2 实测（含重复索引）：反向 max|Δ| 约 2.5e-7（NPU 累加顺序与 CPU
    # 略有差异），阈值留足余量。前向是纯 gather（无算术），逐位一致，阈值恒为 0。
    return 1e-5 if dtype == torch.complex64 else 2e-14


def check_precision(device, rtol) -> tuple[list[float], list[str]]:
    """前向 + 反向逐 case 对拍 CPU 原生 torch.index_select。

    返回 ``(误差列表, 失败描述列表)``；失败列表非空即代表有 case 超阈值。
    """
    worst: list[float] = []
    failures: list[str] = []

    def _record(kind, dtype, shape, dim, m, err, limit):
        worst.append(err)
        if err > limit:
            failures.append(
                f"{kind} {dtype} shape={shape} dim={dim} M={m} "
                f"max|Δ|={err:.3e} > {limit:.3e}"
            )
        print(
            f"[precision] {kind} {dtype} shape={shape} dim={dim} M={m} "
            f"max|Δ|={err:.3e} {'ok' if err <= limit else 'FAIL'}"
        )

    for shape, dim, m, dup, index_dtype in CASES:
        for dtype in DTYPES:
            torch.manual_seed(0)
            z = _make(shape, dtype, device)
            index = _index_for(shape, dim, m, device, index_dtype, dup=dup)
            ref = torch.index_select(z.cpu(), dim, index.cpu())
            got = complex_index_select(z, dim, index)
            if device == "npu":
                torch.npu.synchronize()
            err = (
                float((got.cpu() - ref).abs().max())
                if ref.numel()
                else 0.0
            )
            _record("fwd", dtype, shape, dim, m, err, 0.0)

            d = dim if dim >= 0 else dim + len(shape)
            out_shape = list(shape)
            out_shape[d] = m
            seed = _make(tuple(out_shape), dtype, device)

            cpu_z = z.cpu().clone().requires_grad_(True)
            cpu_out = torch.index_select(cpu_z, dim, index.cpu())
            cpu_g = torch.autograd.grad(cpu_out, cpu_z, seed.cpu())[0]

            dev_z = z.clone().requires_grad_(True)
            dev_out = complex_index_select(dev_z, dim, index)
            dev_g = torch.autograd.grad(dev_out, dev_z, seed)[0]
            if device == "npu":
                torch.npu.synchronize()
            gerr = float((dev_g.cpu() - cpu_g).abs().max())
            _record(
                "bwd", dtype, shape, dim, m, gerr,
                max(_bwd_tol(dtype), rtol * float(cpu_g.abs().max())),
            )
    return worst, failures


def run_perf(device):
    """输出机器可读的 [perf] 行（键名与开源报告解析器一致）。

    - 前向：原生 ``torch.index_select`` 对 complex64 在 NPU 上「可用但派发到
      AiCPU 标量核」，这正是本算子要替换的路径，作为 torch 基准。
    - 反向：NPU 原生对复数直接报错，无可用基准，按 conv3d 的写法记
      ``native=CRASH``，只报告本算子的前向 + 反向耗时；CPU 冒烟时原生可用，
      正常输出对照。
    """
    dtype = torch.complex64

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

    torch.manual_seed(0)
    for shape, dim, m in PERF_CASES:
        z = _make(shape, dtype, device)
        index = _index_for(shape, dim, m, device, torch.int64)
        tag = "x".join(str(s) for s in shape) + f"xM{m}xdim{dim}"

        torch_ms = bench(lambda: torch.index_select(z, dim, index))
        custom_ms = bench(lambda: complex_index_select(z, dim, index))
        speedup = torch_ms / custom_ms if custom_ms > 0 else 0.0
        print(
            f"[perf] index_select fwd [{tag}] torch={torch_ms:.3f}ms "
            f"ascendc={custom_ms:.3f}ms speedup={speedup:.2f}x"
        )

        out_shape = tuple(
            list(shape[:dim]) + [m] + list(shape[dim + 1:])
        )
        seed = _make(out_shape, dtype, device)

        z_g = z.clone().requires_grad_(True)

        def _custom_fwdbwd():
            out = complex_index_select(z_g, dim, index)
            torch.autograd.grad(out, z_g, seed)

        custom_ms = bench(_custom_fwdbwd)

        if device == "cpu":
            n_g = z.clone().requires_grad_(True)

            def _native_fwdbwd():
                out = torch.index_select(n_g, dim, index)
                torch.autograd.grad(out, n_g, seed)

            torch_ms = bench(_native_fwdbwd)
            speedup = torch_ms / custom_ms if custom_ms > 0 else 0.0
            print(
                f"[perf] index_select fwdbwd [{tag}] torch={torch_ms:.3f}ms "
                f"ascendc={custom_ms:.3f}ms speedup={speedup:.2f}x"
            )
            continue

        try:
            n_g = z.clone().requires_grad_(True)
            nout = torch.index_select(n_g, dim, index)
            torch.autograd.grad(nout, n_g, seed)
        except RuntimeError:
            print(
                f"[perf] index_select fwdbwd [{tag}] "
                f"native=CRASH custom={custom_ms:.3f}ms"
            )
            # 同字节量 float32 基线（fwd+bwd）：文档「反向与 float32 基线」
            # 声明的仓内可复现口径。complex 视图末轴翻倍即同字节的实数张量，
            # gather 字节模式与 flat 路径一致；末轴 / 1-D 用例的实数基线无
            # 同构形态，跳过（声明亦只覆盖 flat 形态）。
            if dim < len(shape) - 1:
                real = _real_dtype(dtype)
                f32_shape = tuple(
                    list(shape[:-1]) + [shape[-1] * 2]
                )
                f32_out_shape = tuple(
                    list(f32_shape[:dim]) + [m] + list(f32_shape[dim + 1:])
                )
                f32_g = torch.randn(
                    f32_shape, dtype=real, device=z.device, requires_grad=True
                )
                f32_seed = torch.randn(
                    f32_out_shape, dtype=real, device=z.device
                )

                def _f32_fwdbwd():
                    out = torch.index_select(f32_g, dim, index)
                    torch.autograd.grad(out, f32_g, f32_seed)

                f32_ms = bench(_f32_fwdbwd)
                ratio = f32_ms / custom_ms if custom_ms > 0 else 0.0
                print(
                    f"[perf] index_select fwdbwd-baseline [{tag}] "
                    f"f32={f32_ms:.3f}ms custom={custom_ms:.3f}ms "
                    f"ratio={ratio:.2f}x"
                )
                del f32_g, f32_seed
        else:
            # 上游 aclnnIndexAdd 补齐复数 dtype 后会走到这里：本算子的核心
            # 存在理由消失，需同步更新文档与移除计划。
            raise AssertionError("NPU 原生反向意外可用：请更新本脚本与文档")


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
