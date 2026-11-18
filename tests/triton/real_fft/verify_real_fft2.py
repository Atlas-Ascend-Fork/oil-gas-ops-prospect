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

"""real_rfft2 / real_irfft2 精度与性能验证（NPU）。

精度：float64 CPU ``torch.fft`` 为参考，对比 NPU 上的 ``real_rfft2`` /
``real_irfft2``（分离式 + 一维 Triton kernel），另含 autograd。

性能：同步计时（warmup + 多次平均），对拍 ``torch.fft.rfft2`` / ``irfft2``。
收益取决于行数 ``numel/(h*w)`` 而非 rank：行数少时本算子明显更慢，约 256 行起反超
（原因见 triton-ascend/real_fft/README.md）。下面的 shape 列表同时覆盖两档区间。

用法::

    python3 tests/triton/real_fft/verify_real_fft2.py
    python3 tests/triton/real_fft/verify_real_fft2.py --mode precision --rtol 1e-5
    python3 tests/triton/real_fft/verify_real_fft2.py --json out.json
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
from oil_gas_ops_prospect.real_fft import real_irfft2, real_rfft2  # noqa: E402

try:
    import torch_npu  # noqa: F401
except ImportError:  # pragma: no cover - CPU 环境直接退出
    torch_npu = None

# (lead..., h, w)：2 的幂与非 2 的幂，覆盖 strided kernel 的 320 边界两侧，
# 以及「低秩（原生更快）→ 高秩（组合实现反超）」两种性能区间。
SHAPES = [
    (2, 32, 32),
    (1, 64, 64),
    (2, 3, 128, 128),
    (1, 64, 288),
    (1, 576, 576),
    (1, 160, 288, 288),
    (1, 2, 32, 288, 288),
    (1, 2, 4, 32, 288, 288),
    (1, 2, 4, 32, 576, 576),
]

DIMS = [(-2, -1), (-1, -2)]


def require_npu():
    if torch_npu is None or not (hasattr(torch, "npu") and torch.npu.is_available()):
        raise SystemExit("NPU is not available; this script needs torch_npu and a real device")


def rel_err(actual: torch.Tensor, expect: torch.Tensor) -> float:
    denom = expect.abs().max().clamp_min(1e-12)
    return ((actual - expect).abs().max() / denom).item()


def bench_ms(fn, warmup: int = 5, iters: int = 20) -> float:
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def run_accuracy(shapes, dims, rtol: float, report: dict) -> list[str]:
    failures: list[str] = []
    for shape in shapes:
        for dim in dims:
            torch.manual_seed(0)
            x_cpu = torch.randn(shape, dtype=torch.float64)
            x_npu = x_cpu.float().npu()

            ref = torch.fft.rfft2(x_cpu, dim=dim, norm="backward")
            xr, xi = real_rfft2(x_npu, dim=dim, norm="backward")
            ef = max(rel_err(xr.double().cpu(), ref.real), rel_err(xi.double().cpu(), ref.imag))

            y = real_irfft2(xr, xi, s=(shape[dim[0]], shape[dim[1]]), dim=dim, norm="backward")
            ei = rel_err(y.double().cpu(), x_cpu)

            ortho_ref = torch.fft.rfft2(x_cpu, dim=dim, norm="ortho")
            o_r, o_i = real_rfft2(x_npu, dim=dim, norm="ortho")
            eo = max(
                rel_err(o_r.double().cpu(), ortho_ref.real),
                rel_err(o_i.double().cpu(), ortho_ref.imag),
            )

            entry = {
                "shape": list(shape),
                "dim": list(dim),
                "fwd_backward_norm_rel": ef,
                "inv_backward_norm_rel": ei,
                "fwd_ortho_norm_rel": eo,
            }
            report["accuracy"].append(entry)
            print(f"  {tuple(shape)} dim={dim} fwd={ef:.3e} inv={ei:.3e} ortho={eo:.3e}")
            if max(ef, ei, eo) > rtol:
                failures.append(f"{tuple(shape)} dim={dim} rel={max(ef, ei, eo):.3e} > {rtol}")

    # autograd：对拍 float64 参考的梯度
    for shape in [(2, 16, 16), (1, 32, 48)]:
        torch.manual_seed(1)
        x_cpu = torch.randn(shape, dtype=torch.float64, requires_grad=True)
        ref = torch.fft.rfft2(x_cpu, dim=(-2, -1), norm="backward")
        w = torch.randn(ref.real.shape, dtype=torch.float64)
        g_cpu, = torch.autograd.grad((ref.real * w).sum(), x_cpu)

        x_npu = x_cpu.detach().float().npu().requires_grad_(True)
        xr, xi = real_rfft2(x_npu, dim=(-2, -1), norm="backward")
        g_npu, = torch.autograd.grad((xr * w.float().npu()).sum(), x_npu)
        e = rel_err(g_npu.double().cpu(), g_cpu)
        report["backward"].setdefault("cases", []).append({"shape": list(shape), "grad_rel": e})
        print(f"  grad {tuple(shape)} rel={e:.3e}")
        if e > rtol * 10:
            failures.append(f"grad {tuple(shape)} rel={e:.3e} > {rtol * 10:.3e}")
    return failures


def run_perf(shapes, dims, report: dict) -> None:
    for shape in shapes:
        for dim in dims:
            torch.manual_seed(2)
            x = torch.randn(shape, dtype=torch.float32).npu()
            s = (shape[dim[0]], shape[dim[1]])
            try:
                t_torch = bench_ms(lambda: torch.fft.rfft2(x, s=s, dim=dim))
            except Exception as exc:  # noqa: BLE001 - torch 在该 shape 上可能不支持
                print(f"  {tuple(shape)} dim={dim} torch rfft2 unavailable: {exc}")
                t_torch = float("nan")
            t_og = bench_ms(lambda: real_rfft2(x, s=s, dim=dim))

            xr, xi = real_rfft2(x, s=s, dim=dim)
            try:
                t_torch_inv = bench_ms(lambda: torch.fft.irfft2(torch.complex(xr, xi), s=s, dim=dim))
            except Exception:
                t_torch_inv = float("nan")
            t_og_inv = bench_ms(lambda: real_irfft2(xr, xi, s=s, dim=dim))

            entry = {
                "shape": list(shape),
                "dim": list(dim),
                "rfft2_torch_ms": t_torch,
                "rfft2_ogops_ms": t_og,
                "rfft2_speedup": (t_torch / t_og) if t_og else float("nan"),
                "irfft2_torch_ms": t_torch_inv,
                "irfft2_ogops_ms": t_og_inv,
                "irfft2_speedup": (t_torch_inv / t_og_inv) if t_og_inv else float("nan"),
            }
            report["perf"].append(entry)
            print(
                f"  {tuple(shape)} dim={dim} "
                f"rfft2 torch={t_torch:.2f}ms ogops={t_og:.2f}ms x{entry['rfft2_speedup']:.2f} | "
                f"irfft2 torch={t_torch_inv:.2f}ms ogops={t_og_inv:.2f}ms x{entry['irfft2_speedup']:.2f}"
            )


def main() -> int:
    ap = argparse.ArgumentParser(description="real_rfft2 / real_irfft2 精度与性能验证")
    ap.add_argument("--mode", choices=("all", "precision", "perf"), default="all",
                    help="all=精度+性能（默认）；precision/perf 供 npu_python 套件调用")
    ap.add_argument("--build-ext", action="store_true",
                    help="npu_python 套件协议兼容参数；本算子无 C++ 扩展，忽略")
    ap.add_argument("--rtol", type=float, default=1e-5, help="precision 模式的最大相对误差阈值")
    ap.add_argument("--json", default="", help="结果写入该 JSON 文件")
    args = ap.parse_args()

    require_npu()
    print(f"torch={torch.__version__}  torch_npu={getattr(torch_npu, '__version__', '?')}  "
          f"device={torch.npu.get_device_name(0)}")

    report: dict = {
        "operator": "real_fft2",
        "date": time.strftime("%Y-%m-%d"),
        "tolerance": {"rel": args.rtol, "grad_rel": args.rtol * 10},
        "accuracy": [],
        "backward": {},
        "perf": [],
    }

    failures: list[str] = []
    if args.mode in ("all", "precision"):
        failures = run_accuracy(SHAPES, DIMS, args.rtol, report)
    if args.mode in ("all", "perf"):
        run_perf(SHAPES, DIMS, report)
        print(
            "\n注意：组合实现的 kernel 数多于原生，收益取决于行数 numel/(h*w) 而非 rank；"
            "行数 ≲64 时更慢，约 256 行起反超。详见 triton-ascend/real_fft/README.md。"
        )

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
