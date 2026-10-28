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

"""complex_exp 精度与性能验证（NPU）。

复数指数 ``e^z`` 在 NPU 上原生走 ``aclnnExp``（Ai_Cpu，慢 2~3 个数量级）。
本脚本验证 Triton-Ascend 版：

精度：以 CPU float64 为参考，对比 triton kernel 与 torch NPU ``z.exp()``。
性能：同步计时（warmup + 多次平均），报告耗时、等效带宽与相对基线的加速比，
另含反向 kernel。

用法::

    python3 tests/triton/complex_exp/verify_complex_exp.py            # 默认规模
    python3 tests/triton/complex_exp/verify_complex_exp.py --mode precision --rtol 1e-5
    python3 tests/triton/complex_exp/verify_complex_exp.py --json out.json
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

import _bootstrap  # noqa: F401, E402
from oil_gas_ops_prospect.complex_exp import complex_exp  # noqa: E402

try:
    import torch_npu  # noqa: F401
except ImportError:
    torch_npu = None

# 业务大形状（profiler 抓取，16.3M complex64 元素）
PROD_SHAPE = (304, 304, 176)
DEFAULT_SHAPES = [(13,), (1000001,), (64, 64, 64), PROD_SHAPE, (4096, 4096)]


def bench_ms(fn, warmup: int = 5, iters: int = 20) -> float:
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
    flat = z.detach().contiguous().reshape(-1).view(torch.float32).to("npu")
    return flat.view(torch.complex64).reshape(z.shape)


def rel_abs_err(got: torch.Tensor, ref: torch.Tensor) -> tuple[float, float]:
    got = to_cpu_complex(got)
    ref = ref.to(torch.complex128)
    diff = (got - ref).abs()
    rel = (diff / ref.abs().clamp_min(1e-30)).max().item()
    return diff.max().item(), rel


def fmt(x: float, prec: str = ".3e") -> str:
    return format(x, prec)


def gen_input(shape, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(shape, generator=g) + 1j * torch.randn(shape, generator=g)).to(torch.complex64)


def run_accuracy_section(shapes, report) -> None:
    print("=" * 78)
    print("一、前向精度（参考：CPU float64）")
    print("=" * 78)
    print(f"{'shape':<22}{'方法':<18}{'max_abs':>12}{'max_rel':>12}")
    print("-" * 78)
    for shape in shapes:
        z = gen_input(shape, seed=1)
        z_npu = to_npu_complex(z)
        ref = torch.exp(z.to(torch.complex128))
        shape_str = "x".join(str(s) for s in shape)
        if len(shape_str) > 20:
            shape_str = shape_str[:17] + "..."
        for name, out in (("triton", complex_exp(z_npu)),
                          ("torch NPU exp", z_npu.exp())):
            max_abs, max_rel = rel_abs_err(out, ref)
            print(f"{shape_str:<22}{name:<18}{fmt(max_abs):>12}{fmt(max_rel):>12}")
            report["accuracy"].append(
                {"shape": list(shape), "method": name, "max_abs": max_abs, "max_rel": max_rel}
            )
        print()


def run_backward_section(report) -> None:
    print("=" * 78)
    print("二、反向精度（grad_z = g * conj(exp(z))，对照 torch autograd / float64）")
    print("=" * 78)
    z = gen_input((4096, 4096), seed=2)
    g = gen_input((4096, 4096), seed=3)

    z_g = to_npu_complex(z).requires_grad_(True)
    g_npu = to_npu_complex(g)
    w = complex_exp(z_g)
    (w.real * g_npu.real + w.imag * g_npu.imag).sum().backward()

    z_r = z.clone().requires_grad_(True)
    w_r = torch.exp(z_r)
    (w_r.real * g.real + w_r.imag * g.imag).sum().backward()

    rel_torch = rel_abs_err(z_g.grad, z_r.grad.to(torch.complex64))[1]
    ref64 = torch.exp(z.to(torch.complex128))
    grad64 = g.to(torch.complex128) * ref64.conj()
    rel_fp64 = ((to_cpu_complex(z_g.grad) - grad64).abs() / grad64.abs().clamp_min(1e-30)).max().item()
    print(f"  vs torch fp32 autograd(对 z): rel={fmt(rel_torch)}")
    print(f"  vs float64 参考:              rel={fmt(rel_fp64)}")

    # 二阶导：create_graph=True 后反向走 torch 组合式，结果需与 float64 对齐。
    z2 = gen_input((513, 257), seed=7)
    g2 = gen_input((513, 257), seed=8)
    h2 = gen_input((513, 257), seed=9)
    z2_g = to_npu_complex(z2).requires_grad_(True)
    gz, = torch.autograd.grad(complex_exp(z2_g), z2_g, grad_outputs=to_npu_complex(g2), create_graph=True)
    gg, = torch.autograd.grad((gz * to_npu_complex(h2).conj()).real.sum(), z2_g)

    z2_64 = z2.to(torch.complex128).requires_grad_(True)
    gz64, = torch.autograd.grad(torch.exp(z2_64), z2_64,
                                grad_outputs=g2.to(torch.complex128), create_graph=True)
    gg64, = torch.autograd.grad((gz64 * h2.to(torch.complex128).conj()).real.sum(), z2_64)
    rel_gg = ((to_cpu_complex(gg) - gg64).abs() / gg64.abs().clamp_min(1e-30)).max().item()
    print(f"  二阶导 vs float64 参考:        rel={fmt(rel_gg)}")
    report["backward"] = {"vs_torch_fp32": rel_torch, "vs_float64": rel_fp64, "second_order_vs_float64": rel_gg}


def run_perf_section(shapes, report) -> None:
    print("=" * 78)
    print("三、前向性能（同步计时，含等效带宽与加速比）")
    print("=" * 78)
    print("  计时口径：warmup 5 次 + 20 次调用的墙钟平均（含 host 侧派发/launch 开销）。")
    print(f"{'shape':<22}{'方法':<18}{'耗时(ms)':>12}{'GB/s':>9}{'加速比':>9}")
    print("-" * 78)
    for shape in shapes:
        z = to_npu_complex(gen_input(shape, seed=4))
        n = z.numel()
        bytes_moved = n * 16  # 读 8B + 写 8B 每复数元素

        results = {}
        for name, fn in (
            ("torch exp", lambda t=z: t.exp()),
            ("triton", lambda t=z: complex_exp(t)),
        ):
            results[name] = bench_ms(fn)
        base = results["torch exp"]
        for name, ms in results.items():
            gbps = bytes_moved / (ms * 1e6)
            print(f"{'x'.join(str(s) for s in shape):<22}{name:<18}{ms:>12.4f}{gbps:>9.1f}{base / ms:>9.1f}x")
        report["perf_forward"].append(
            {"shape": list(shape), "n": n, "ms": results,
             "gbps_triton": bytes_moved / (results["triton"] * 1e6),
             "speedup_vs_torch": base / results["triton"]}
        )
        print()


def run_backward_perf_section(report) -> None:
    print("=" * 78)
    print("四、反向 kernel 性能（读 w+g 共 16B/元素，写 8B/元素）")
    print("=" * 78)
    z = to_npu_complex(gen_input(PROD_SHAPE, seed=5))
    g = to_npu_complex(gen_input(PROD_SHAPE, seed=6))
    z_req = z.requires_grad_(True)

    def fwd_bwd():
        out = complex_exp(z_req)
        return torch.autograd.grad(out, z_req, grad_outputs=g, retain_graph=True)[0]

    ms = bench_ms(fwd_bwd)
    ms_fwd = bench_ms(lambda: complex_exp(z))
    n = z.numel()
    bytes_total = n * 8 * 2 + n * 8 * 3  # 前向 16B + 反向 24B 每元素
    print(f"  前向: {ms_fwd:.4f} ms   前向+反向: {ms:.4f} ms  （合计 {bytes_total / (ms * 1e6):.1f} GB/s）")
    report["perf_backward"] = {"fwd_ms": ms_fwd, "fwd_bwd_ms": ms, "n": n}


def _validate_precision(report: dict, rtol: float) -> list[str]:
    """按阈值校验精度结果，返回失败项列表（triton 路径 vs float64 参考）。"""
    failures: list[str] = []
    for row in report["accuracy"]:
        if row["method"] == "triton" and row["max_rel"] > rtol:
            failures.append(
                f"前向 shape={row['shape']} max_rel={row['max_rel']:.3e} > {rtol:.0e}")
    bwd = report["backward"]
    if bwd.get("vs_float64", 0.0) > rtol * 10:
        failures.append(f"反向 vs float64 rel={bwd['vs_float64']:.3e} > {rtol * 10:.0e}")
    if bwd.get("second_order_vs_float64", 0.0) > rtol * 10:
        failures.append(f"二阶导 vs float64 rel={bwd['second_order_vs_float64']:.3e} > {rtol * 10:.0e}")
    return failures


def _print_perf_lines(report: dict) -> None:
    """输出机器可读的 [perf] 行（键名与开源报告解析器一致）。"""
    for row in report["perf_forward"]:
        tag = "x".join(str(s) for s in row["shape"])
        torch_ms = row["ms"].get("torch exp")
        triton_ms = row["ms"].get("triton")
        if torch_ms and triton_ms:
            print(f"[perf] exp fwd [{tag}] torch={torch_ms:.3f}ms "
                  f"ascendc={triton_ms:.3f}ms speedup={torch_ms / triton_ms:.2f}x")


def _environment() -> dict:
    """记录本次运行的软硬件环境，便于报告溯源。"""
    import glob
    import platform

    env = {
        "host": platform.node(),
        "device": torch.npu.get_device_name(0),
        "torch": torch.__version__,
        "torch_npu": getattr(torch_npu, "__version__", "?"),
        "triton": __import__("triton").__version__,
    }
    # CANN 版本只在装好 toolkit 的镜像里有；目录名形如 cann-9.0.0。
    for pattern in (
        "/usr/local/Ascend/ascend-toolkit/latest/*/ascend_toolkit_install.info",
        "/usr/local/Ascend/cann-*/version.info",
        "/usr/local/Ascend/cann-*/*/ascend_toolkit_install.info",
    ):
        for path in sorted(glob.glob(pattern)):
            try:
                with open(path, encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        if "version" in line.lower():
                            _, _, value = line.partition("=")
                            value = value.strip() or line.strip()
                            if value:
                                env["cann"] = value
                                return env
            except OSError:
                continue
    return env


def main() -> int:
    ap = argparse.ArgumentParser(description="complex_exp 精度与性能验证")
    ap.add_argument("--mode", choices=("all", "precision", "perf"), default="all",
                    help="all=精度+性能（默认，手动用法不变）；precision/perf 供"
                         " npu_python 套件调用（tests/suites.json）")
    ap.add_argument("--build-ext", action="store_true",
                    help="npu_python 套件协议兼容参数；本算子无 C++ 扩展，忽略")
    ap.add_argument("--rtol", type=float, default=1e-5,
                    help="precision 模式阈值：前向 / 反向 vs float64 的最大相对误差")
    ap.add_argument("--json", default="", help="结果写入该 JSON 文件")
    args = ap.parse_args()

    require_npu()

    print(f"torch={torch.__version__}  torch_npu={getattr(torch_npu, '__version__', '?')}  "
          f"triton={__import__('triton').__version__}  device={torch.npu.get_device_name(0)}")

    report: dict = {
        "operator": "complex_exp",
        "date": time.strftime("%Y-%m-%d"),
        "environment": _environment(),
        "tolerance": {
            "forward_vs_float64_rel": args.rtol,
            "backward_vs_float64_rel": args.rtol * 10,
            "second_order_vs_float64_rel": args.rtol * 10,
        },
        "accuracy": [],
        "backward": {},
        "perf_forward": [],
        "perf_backward": {},
    }
    if args.mode in ("all", "precision"):
        run_accuracy_section(DEFAULT_SHAPES, report)
        run_backward_section(report)
    if args.mode in ("all", "perf"):
        run_perf_section(DEFAULT_SHAPES, report)
        run_backward_perf_section(report)

    failures = _validate_precision(report, args.rtol) if args.mode in ("all", "precision") else []
    if args.mode in ("all", "perf"):
        _print_perf_lines(report)

    if args.json:
        report["failures"] = failures
        report["status"] = "FAIL" if failures else "PASS"
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
