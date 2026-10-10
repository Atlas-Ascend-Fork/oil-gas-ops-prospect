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

"""complex_reciprocal 算子的精度与性能验证脚本。

复数倒数 ``1/z`` 在 NPU 上原生走 ``aclnnReciprocal``（Ai_Cpu，慢且
不稳定）。本脚本验证 Triton-Ascend 版：

精度：以 CPU float64 为参考，对比
  - triton stable / naive 两种 kernel 路径
  - torch NPU ``1/z``（aclnnReciprocal 基线）
  - md 方案 A 的 float32 拆解 ``conj(z)/(a^2+b^2)``
  - torch CPU fp32 ``1/z``（fp32 能达到的水平参照）

性能：同步计时（warmup+多次平均），报告耗时、等效带宽（GB/s）与相对基线的加速比；
另含实虚分离融合入口的端到端对比（先构造 complex64 的原始写法 vs 融合算子）与反向 kernel。

用法::

    python3 tests/triton/complex_reciprocal/verify_complex_reciprocal.py            # 默认规模
    python3 tests/triton/complex_reciprocal/verify_complex_reciprocal.py --full     # 追加大规模形状
    python3 tests/triton/complex_reciprocal/verify_complex_reciprocal.py --json out.json
    python3 tests/triton/complex_reciprocal/verify_complex_reciprocal.py --block-sweep
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import _bootstrap  # noqa: F401, E402  (side effect: 源码树里解析 oil_gas_ops_prospect)
from oil_gas_ops_prospect.complex_reciprocal import (  # noqa: E402
    complex_reciprocal,
    triton_complex_reciprocal as recip_impl,
)

try:
    import torch_npu  # noqa: F401
except ImportError:
    torch_npu = None

# 业务大形状（profiler 抓取，16.3M complex64 元素）
PROD_SHAPE = (304, 304, 176)

DEFAULT_SHAPES = [(13,), (1000001,), (64, 64, 64), PROD_SHAPE, (4096, 4096)]
FULL_EXTRA_SHAPES = [(128, 128, 128), (256, 256, 256)]

# 极值幅值：naive 的 a^2+b^2 会溢出，stable 应保持正确
EXTREME_SCALES = (1e-20, 1e20)


def bench_ms(fn, warmup: int = 5, iters: int = 20) -> float:
    """同步计时的平均耗时（ms）。

    不用 ``do_bench_npu``：其对非 Triton 算子（aclnn）会因 profiler 取不到
    kernel 事件返回 inf，且每次测量有 ~3s 的 profiler 解析开销。
    """
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def require_npu():
    if torch_npu is None or not torch.npu.is_available():
        raise SystemExit("需要 NPU 环境（torch_npu + 可用 NPU 设备）")
    torch.npu.set_device(0)


def to_cpu_complex(z: torch.Tensor) -> torch.Tensor:
    return z.detach().cpu().to(torch.complex128)


def to_npu_complex(z: torch.Tensor) -> torch.Tensor:
    flat = z.contiguous().reshape(-1).view(torch.float32).to("npu:0")
    return flat.view(torch.complex64).reshape(z.shape)


def rel_abs_err(got: torch.Tensor, ref: torch.Tensor) -> tuple[float, float]:
    got = to_cpu_complex(got)
    ref = ref.to(torch.complex128)
    diff = (got - ref).abs()
    rel = (diff / ref.abs().clamp_min(1e-300)).max().item()
    return diff.max().item(), rel


def impl_member(name: str):
    """按名取实现模块的成员（``_get_vectorcore_num`` 等下划线前缀的实现细节）。

    直接书写 ``recip_impl._xxx`` 会触发 protected-access 告警，按仓库
    ``kernel_of``（aot_compile_check.py）的先例用 ``getattr`` 按名访问。
    """
    return getattr(recip_impl, name)


def fmt(x: float, prec: str = ".3e") -> str:
    return format(x, prec)


# ------------------------------------------------------------------ #
# 精度
# ------------------------------------------------------------------ #

def _gen_shifted(shape, g: torch.Generator) -> torch.Tensor:
    # 偏移分布的典型取值域：实部 [0,4]-0.25，虚部 -eps
    real = torch.rand(shape, generator=g) * 4.0 - 0.25
    return real - 0.01j


def _gen_scaled_randn(shape, g: torch.Generator, scale: float) -> torch.Tensor:
    return scale * (torch.randn(shape, generator=g) + 1j * torch.randn(shape, generator=g))


# 分布名 -> (生成器, 生成时是否在极值幅值)。“极值分布用小形状即可说明问题”的跳过
# 判断依赖幅值信息，与生成逻辑放在一起。
_INPUT_DISTS = {
    "randn": lambda shape, g: _gen_scaled_randn(shape, g, 1.0),
    "shifted": _gen_shifted,
    "tiny": lambda shape, g: _gen_scaled_randn(shape, g, 1e-20),
    "huge": lambda shape, g: _gen_scaled_randn(shape, g, 1e20),
}
_EXTREME_DISTS = ("tiny", "huge")


def gen_input(shape, dist: str, seed: int) -> torch.Tensor:
    if dist not in _INPUT_DISTS:
        raise ValueError(dist)
    g = torch.Generator().manual_seed(seed)
    return _INPUT_DISTS[dist](shape, g).to(torch.complex64)


def _accuracy_rows(zc: torch.Tensor, z_npu: torch.Tensor) -> list[tuple[str, torch.Tensor]]:
    """待对比的各方法及其前向输出（triton 两种路径 + 三个基线）。"""
    rows = [
        ("triton stable", complex_reciprocal(z_npu, stable=True)),
        ("triton naive", complex_reciprocal(z_npu, stable=False)),
        ("torch NPU 1/z", 1 / z_npu),
    ]
    a, b = z_npu.real, z_npu.imag
    rows.append(("方案A 拆解", torch.complex(a, -b) / (a * a + b * b)))
    rows.append(("torch CPU fp32", 1 / zc))
    return rows


def _report_accuracy_case(shape, dist: str, report) -> None:
    """单个 (shape, 分布) 用例：算 float64 参考并逐方法打印/记录误差。"""
    zc = gen_input(shape, dist, seed=42)
    ref = 1 / zc.to(torch.complex128)
    z_npu = to_npu_complex(zc)
    shape_str = "x".join(str(s) for s in shape)
    if len(shape_str) > 20:
        shape_str = shape_str[:17] + "..."
    for name, out in _accuracy_rows(zc, z_npu):
        max_abs, max_rel = rel_abs_err(out, ref)
        print(f"{shape_str:<22}{dist:<7}{name:<18}{fmt(max_abs):>12}{fmt(max_rel):>12}")
        report["accuracy"].append(
            {"shape": list(shape), "dist": dist, "method": name,
             "max_abs": max_abs, "max_rel": max_rel}
        )


def run_accuracy_section(shapes, report) -> None:
    print("=" * 78)
    print("一、前向精度（参考：CPU float64）")
    print("=" * 78)
    header = f"{'shape':<22}{'分布':<7}{'方法':<18}{'max_abs':>12}{'max_rel':>12}"
    print(header)
    print("-" * len(header))
    for shape in shapes:
        for dist in _INPUT_DISTS:
            if dist in _EXTREME_DISTS and torch.tensor(shape).prod() > 2_000_000:
                continue  # 极值分布用小形状即可说明问题
            _report_accuracy_case(shape, dist, report)


def run_backward_section(report) -> None:
    print()
    print("=" * 78)
    print("二、反向精度（grad_z = -g*conj(r^2)，对照 torch autograd / float64）")
    print("=" * 78)
    n = 100003
    g = torch.Generator().manual_seed(7)
    zc = (torch.randn(n, generator=g) + 1j * torch.randn(n, generator=g)).to(torch.complex64)
    u, v = torch.randn(n, generator=g), torch.randn(n, generator=g)
    gout = torch.complex(u, v)

    z_ref = zc.clone().requires_grad_(True)
    g_torch = torch.autograd.grad(1 / z_ref, z_ref, grad_outputs=gout)[0]

    z64 = zc.to(torch.complex128).requires_grad_(True)
    g64 = torch.autograd.grad(1 / z64, z64, grad_outputs=gout.to(torch.complex128))[0]

    z_npu = to_npu_complex(zc).requires_grad_(True)
    out = complex_reciprocal(z_npu)
    gz = torch.autograd.grad(out, z_npu, grad_outputs=gout.to("npu:0"))[0]

    _, rel_torch = rel_abs_err(gz, g_torch)
    _, rel_fp64 = rel_abs_err(gz, g64)
    _, rel_torch_own = rel_abs_err(g_torch, g64)
    print(f"  vs torch fp32 autograd(对 z): rel={fmt(rel_torch)}   （同路径逐位接近）")
    print(f"  vs float64 参考:              rel={fmt(rel_fp64)}")
    print(f"  torch fp32 autograd 自身 vs float64: rel={fmt(rel_torch_own)}")
    print("  （vs float64 的 1e-3 量级误差来自 float32 消 catastrophic cancellation，")
    print("    与 torch 自身水平相当，非算子缺陷）")
    report["backward"] = {
        "vs_torch_fp32": rel_torch,
        "vs_float64": rel_fp64,
        "torch_fp32_vs_float64": rel_torch_own,
    }


# ------------------------------------------------------------------ #
# 性能
# ------------------------------------------------------------------ #

def run_perf_section(shapes, report) -> None:
    print()
    print("=" * 78)
    print("三、前向性能（同步计时，含等效带宽与加速比）")
    print("=" * 78)
    print("  计时口径：warmup 5 次 + 20 次调用的墙钟平均（含 host 侧派发/launch 开销，")
    print("  贴近真实训练循环；aclnn 多算子路径的 host 开销远大于单 kernel 路径）。")
    print("  注：极小形状下 Triton kernel 的 python launch 开销 (~0.3ms) 会超过 torch，")
    print("  属预期；元素数 >= 1M 时 Triton 全面占优。")
    header = (f"{'shape':<22}{'方法':<18}{'耗时(ms)':>12}{'GB/s':>9}{'加速比':>10}")
    print(header)
    print("-" * len(header))
    for shape in shapes:
        n = int(torch.tensor(shape).prod())
        if n == 0:
            continue
        zc = gen_input(shape, "randn", seed=99)
        z = to_npu_complex(zc)
        a, b = z.real, z.imag
        bytes_moved = n * 8 * 2  # 读 8B + 写 8B 每元素

        def torch_recip():
            return 1 / z

        def plan_a():
            return torch.complex(a, -b) / (a * a + b * b)

        def triton_stable():
            return complex_reciprocal(z, stable=True)

        def triton_naive():
            return complex_reciprocal(z, stable=False)

        results = {}
        for name, fn in (("torch 1/z", torch_recip), ("方案A 拆解", plan_a),
                         ("triton stable", triton_stable), ("triton naive", triton_naive)):
            results[name] = bench_ms(fn)
        base = results.get("torch 1/z", float("nan"))
        ms_stable = results.get("triton stable", float("nan"))
        ms_plan_a = results.get("方案A 拆解", float("nan"))
        shape_str = "x".join(str(s) for s in shape)
        if len(shape_str) > 20:
            shape_str = shape_str[:17] + "..."
        for name, ms in results.items():
            gbps = bytes_moved / (ms * 1e6)
            print(f"{shape_str:<22}{name:<18}{ms:>12.4f}{gbps:>9.1f}{base / ms:>9.1f}x")
        report["perf_forward"].append(
            {"shape": list(shape), "n": n, "ms": results,
             "gbps_triton_stable": bytes_moved / (ms_stable * 1e6),
             "speedup_vs_torch": base / ms_stable,
             "speedup_vs_plan_a": ms_plan_a / ms_stable}
        )
        print()


def run_e2e_fused_section(report) -> None:
    print("=" * 78)
    print("四、端到端融合（1 / (a - i*c)：实部张量 + 常数虚部）")
    print("=" * 78)
    torch.manual_seed(11)
    a = torch.rand(PROD_SHAPE) * 4.0 - 0.25
    c = 1e-2
    a_npu = a.to("npu:0")
    ref = 1 / (a.to(torch.complex128) - 1j * c)

    def baseline():
        return 1 / (a_npu - 1j * c)

    def fused():
        return complex_reciprocal(a_npu, -c)

    out = fused()
    torch.npu.synchronize()
    max_abs, max_rel = rel_abs_err(out, ref)
    ms_base = bench_ms(baseline)
    ms_fused = bench_ms(fused)
    n = int(torch.tensor(PROD_SHAPE).prod())
    print(f"  形状 {PROD_SHAPE}（{n} 元素）")
    print(f"  原始写法  1/(a - i*c): {ms_base:.4f} ms")
    print(f"  融合算子  complex_reciprocal(xr, xi): {ms_fused:.4f} ms  加速比 {ms_base / ms_fused:.1f}x")
    print(f"  融合算子 vs float64 参考: max_abs={fmt(max_abs)}  max_rel={fmt(max_rel)}")
    report["e2e_fused"] = {
        "shape": list(PROD_SHAPE), "n": n,
        "baseline_ms": ms_base, "fused_ms": ms_fused,
        "speedup": ms_base / ms_fused, "max_abs": max_abs, "max_rel": max_rel,
    }


def run_backward_perf_section(report) -> None:
    print()
    print("=" * 78)
    print("五、反向 kernel 性能（读 r+g 共 16B/元素，写 8B/元素）")
    print("=" * 78)
    n = int(torch.tensor(PROD_SHAPE).prod())
    z = to_npu_complex(gen_input(PROD_SHAPE, "randn", seed=13))
    g = to_npu_complex(gen_input(PROD_SHAPE, "randn", seed=14))
    z_req = z.requires_grad_(True)

    def fwd_bwd():
        out = complex_reciprocal(z_req)
        return torch.autograd.grad(out, z_req, grad_outputs=g, retain_graph=True)[0]

    ms = bench_ms(fwd_bwd)
    ms_fwd = bench_ms(lambda: complex_reciprocal(z))
    bytes_total = n * 8 * 2 + n * 8 * 3  # 前向 16B + 反向 24B 每元素
    print(f"  前向: {ms_fwd:.4f} ms   前向+反向: {ms:.4f} ms  （合计 {bytes_total / (ms * 1e6):.1f} GB/s）")
    report["perf_backward"] = {"fwd_ms": ms_fwd, "fwd_bwd_ms": ms, "n": n}


def run_block_sweep(report) -> None:
    print()
    print("=" * 78)
    print("六、block_size 扫描（业务大形状，stable 路径）")
    print("=" * 78)
    z = to_npu_complex(gen_input(PROD_SHAPE, "randn", seed=15))
    for block in (64, 128, 256, 512, 1024, 2048):
        ms = bench_ms(lambda: complex_reciprocal(z, block_size=block))
        gbps = z.numel() * 16 / (ms * 1e6)
        print(f"  block_size={block:<6d} {ms:.4f} ms  {gbps:.1f} GB/s")
        report["block_sweep"].append({"block": block, "ms": ms, "gbps": gbps})
    print("  （>2048 会触发 Vector Core UB 运行时溢出，host 侧已钳制）")


def _validate_precision(report: dict, rtol: float) -> list[str]:
    """按阈值校验精度结果，返回失败项列表（stable 路径 vs float64 参考）。"""
    failures: list[str] = []
    for row in report["accuracy"]:
        if row["method"] == "triton stable" and row["max_rel"] > rtol:
            failures.append(
                f"前向 shape={row['shape']} dist={row['dist']} "
                f"max_rel={row['max_rel']:.3e} > {rtol:.0e}")
    bwd = report["backward"]
    if bwd.get("vs_float64", 0.0) > rtol:
        failures.append(f"反向 vs float64 rel={bwd['vs_float64']:.3e} > {rtol:.0e}")
    return failures


def _print_perf_lines(report: dict) -> None:
    """输出机器可读的 [perf] 行（键名与开源报告解析器一致）。"""
    for row in report["perf_forward"]:
        ms = row["ms"]
        tag = "x".join(str(s) for s in row["shape"])
        torch_ms, stable_ms = ms.get("torch 1/z"), ms.get("triton stable")
        if torch_ms and stable_ms:
            print(f"[perf] recip fwd [{tag}] torch={torch_ms:.3f}ms "
                  f"ascendc={stable_ms:.3f}ms speedup={torch_ms / stable_ms:.2f}x")
    e2e = report["e2e_fused"]
    if e2e:
        print(f"[perf] e2e fused [{e2e['n']}elem] torch={e2e['baseline_ms']:.3f}ms "
              f"ascendc={e2e['fused_ms']:.3f}ms speedup={e2e['speedup']:.2f}x")


def main() -> int:
    ap = argparse.ArgumentParser(description="complex_reciprocal 精度与性能验证")
    ap.add_argument("--mode", choices=("all", "precision", "perf"), default="all",
                    help="all=精度+性能（默认，手动用法不变）；precision/perf 供"
                         " npu_python 套件调用（tests/suites.json）")
    ap.add_argument("--build-ext", action="store_true",
                    help="npu_python 套件协议兼容参数；本算子无 C++ 扩展，忽略")
    ap.add_argument("--rtol", type=float, default=1e-5,
                    help="precision 模式阈值：stable 前向 / 反向 vs float64 的最大相对误差")
    ap.add_argument("--full", action="store_true", help="追加大规模形状")
    ap.add_argument("--json", default="", help="结果写入该 JSON 文件")
    ap.add_argument("--block-sweep", action="store_true", help="运行 block_size 扫描")
    args = ap.parse_args()

    require_npu()

    versions = {
        "torch": torch.__version__,
        "torch_npu": getattr(torch_npu, "__version__", "?"),
        "triton": __import__("triton").__version__,
    }
    print(f"torch={versions['torch']}  torch_npu={versions['torch_npu']}  "
          f"triton={versions['triton']}  device={torch.npu.get_device_name(0)}")
    print(f"vector cores = {impl_member('_get_vectorcore_num')()}  "
          f"default block = {impl_member('_resolve_block')(None)}")

    shapes = list(DEFAULT_SHAPES)
    if args.full:
        shapes += FULL_EXTRA_SHAPES

    report: dict = {
        "versions": versions,
        "accuracy": [], "backward": {}, "perf_forward": [],
        "e2e_fused": {}, "perf_backward": {}, "block_sweep": [],
    }
    if args.mode in ("all", "precision"):
        run_accuracy_section(shapes, report)
        run_backward_section(report)
    if args.mode in ("all", "perf"):
        run_perf_section(shapes, report)
        run_e2e_fused_section(report)
        run_backward_perf_section(report)
        if args.block_sweep:
            run_block_sweep(report)

    failures = _validate_precision(report, args.rtol) if args.mode in ("all", "precision") else []
    if args.mode in ("all", "perf"):
        _print_perf_lines(report)

    if args.json:
        Path(args.json).write_text(json.dumps(report, ensure_ascii=False, indent=2))
        print(f"\n结果已写入 {args.json}")
    if failures:
        for f in failures:
            print(f"[FAIL] {f}")
        print("FAIL")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
