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

"""interpolate 的精度与性能验证。

用法::

    python3 tests/triton/interpolate/verify_interpolate.py --mode all
    python3 tests/triton/interpolate/verify_interpolate.py --mode precision
    python3 tests/triton/interpolate/verify_interpolate.py --mode perf --full
    python3 tests/triton/interpolate/verify_interpolate.py --mode all --json out.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import _bootstrap  # noqa: F401, E402
from oil_gas_ops_prospect.interpolate import interpolate  # noqa: E402
from oil_gas_ops_prospect.interpolate import triton_interpolate as ti  # noqa: E402

try:
    import torch_npu  # noqa: F401
except ImportError:
    torch_npu = None

TOL = 1e-5

# The operator's default Cube threshold is 128 MiB. Keep the verification
# shapes at or above that threshold; smaller shapes are covered by the native
# fallback unit test instead of forcing them through the Cube path.
DEFAULT_3D = [
    (1, 32, 33, 33, 33, 128, 128, 128),
    (1, 32, 41, 41, 41, 160, 160, 160),
]
FULL_3D = [
    (1, 256, 33, 33, 33, 128, 128, 128),
    (1, 256, 41, 41, 41, 160, 160, 160),
]
DEFAULT_2D = [
    (1, 128, 65, 65, 256, 256),
]
# Backward benchmarks are intentionally limited to manageable output sizes.
# C=256 forward cases are still covered; their gradients would need several GB
# of extra workspace and are not run by default.
BACKWARD_3D = [
    (1, 32, 33, 33, 33, 128, 128, 128),
    (1, 32, 41, 41, 41, 160, 160, 160),
    (1, 64, 33, 33, 33, 128, 128, 128),
    (1, 64, 41, 41, 41, 160, 160, 160),
]


def require_npu():
    if torch_npu is None or not torch.npu.is_available():
        raise SystemExit("需要 NPU 环境（torch_npu + 可用 NPU 设备）")
    torch.npu.set_device(0)


def bench_ms(fn, warmup: int = 3, iters: int = 10) -> float:
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def _grad_only(y, x, grad_out, retain_graph=True):
    return torch.autograd.grad(y, x, grad_outputs=grad_out, retain_graph=retain_graph)


def _fwd_bwd_custom(x, size, mode, grad_out):
    y = interpolate(x, size=size, mode=mode, align_corners=False)
    return torch.autograd.grad(y, x, grad_outputs=grad_out)


def _fwd_bwd_native(x, size, mode, grad_out):
    y = F.interpolate(x, size=size, mode=mode, align_corners=False)
    return torch.autograd.grad(y, x, grad_outputs=grad_out)


def _shape_label(mode, item):
    if mode == "bilinear":
        _, c, h, w, ho, wo = item
        return f"C={c} {h}x{w}->{ho}x{wo}"
    _, c, d, h, w, do, ho, wo = item
    return f"C={c} {d}x{h}x{w}->{do}x{ho}x{wo}"


def parse_shapes(args):
    shapes = []
    for item in DEFAULT_3D + (FULL_3D if args.full else []):
        shapes.append(("trilinear", item))
    for item in DEFAULT_2D:
        shapes.append(("bilinear", item))
    return shapes


def run_precision(shapes, tol):
    results = []
    failed = 0
    print("== precision ==")
    print(
        f"{'mode':10s} {'shape':30s} {'max_abs':>10s} {'max_rel':>10s} {'status':>6s}"
    )
    for mode, item in shapes:
        if mode == "bilinear":
            b, c, h, w, ho, wo = item
            size = (ho, wo)
            x = torch.randn(b, c, h, w, device="npu", dtype=torch.float32)
        else:
            b, c, d, h, w, do, ho, wo = item
            size = (do, ho, wo)
            x = torch.randn(b, c, d, h, w, device="npu", dtype=torch.float32)
        got = interpolate(x, size=size, mode=mode, align_corners=False)
        ref = F.interpolate(x, size=size, mode=mode, align_corners=False)
        diff = (got - ref).abs()
        max_abs = diff.max().item()
        max_rel = (diff / ref.abs().clamp_min(1e-6)).max().item()
        ok = max_abs <= tol
        failed += 0 if ok else 1
        label = (
            f"C={c} {d}x{h}x{w}->{do}x{ho}x{wo}"
            if mode == "trilinear"
            else f"C={c} {h}x{w}->{ho}x{wo}"
        )
        print(
            f"{mode:10s} {label:30s} {max_abs:10.3e} {max_rel:10.3e} {'PASS' if ok else 'FAIL':>6s}"
        )
        results.append(
            {
                "mode": mode,
                "shape": item,
                "max_abs": max_abs,
                "max_rel": max_rel,
                "status": "PASS" if ok else "FAIL",
            }
        )
        del x, got, ref, diff
        torch.npu.empty_cache()
    print(f"precision: {len(results) - failed}/{len(results)} pass, tol={tol:g}")
    return results, failed


def run_perf(shapes):
    results = []
    print("== performance ==")
    print(
        f"{'mode':10s} {'shape':30s} {'triton':>10s} {'native':>10s} {'speedup':>8s} {'GB/s':>8s}"
    )
    for mode, item in shapes:
        # 4D bilinear is native-only and has no Cube performance result.
        if mode != "trilinear":
            continue
        b, c, d, h, w, do, ho, wo = item
        size = (do, ho, wo)
        x = torch.randn(b, c, d, h, w, device="npu", dtype=torch.float32)
        got = interpolate(x, size=size, mode=mode, align_corners=False)
        out_bytes = got.numel() * got.element_size()
        del got
        torch.npu.empty_cache()

        def custom():
            return interpolate(x, size=size, mode=mode, align_corners=False)

        def native():
            return F.interpolate(x, size=size, mode=mode, align_corners=False)

        t_custom = bench_ms(custom)
        t_native = bench_ms(native)
        speedup = t_native / t_custom if t_custom > 0 else 0.0
        gbs = out_bytes / (t_custom * 1e6) if t_custom > 0 else 0.0
        label = (
            f"C={c} {d}x{h}x{w}->{do}x{ho}x{wo}"
            if mode == "trilinear"
            else f"C={c} {h}x{w}->{ho}x{wo}"
        )
        print(
            f"{mode:10s} {label:30s} {t_custom:9.2f}ms {t_native:9.2f}ms "
            f"{speedup:7.2f}x {gbs:7.0f}"
        )
        print(
            f"[perf] interpolate fwd {label} torch={t_native:.4f}ms "
            f"ascendc={t_custom:.4f}ms speedup={speedup:.2f}x"
        )
        results.append(
            {
                "mode": mode,
                "shape": item,
                "triton_ms": t_custom,
                "native_ms": t_native,
                "speedup": speedup,
                "out_gbs": gbs,
            }
        )
        # Do not `del x` here: ruff's flow analysis treats it as making the
        # variable unbound for the closures above (F821), and the input tensor
        # is small compared with the output.  It is overwritten next iteration.
        torch.npu.empty_cache()
    return results


def run_grad_precision(shapes, tol):
    results = []
    failed = 0
    print("== gradient precision ==")
    print(f"{'mode':10s} {'shape':30s} {'grad_max_abs':>12s} {'status':>6s}")
    for item in shapes:
        _, c, d, h, w, do, ho, wo = item
        size = (do, ho, wo)
        x = torch.randn(
            1, c, d, h, w, device="npu", dtype=torch.float32, requires_grad=True
        )
        y = interpolate(x, size=size, mode="trilinear", align_corners=False)
        grad_out = torch.randn_like(y)
        grad_custom = torch.autograd.grad(y, x, grad_outputs=grad_out)[0]

        x_ref = x.detach().clone().requires_grad_(True)
        y_ref = F.interpolate(x_ref, size=size, mode="trilinear", align_corners=False)
        grad_ref = torch.autograd.grad(y_ref, x_ref, grad_outputs=grad_out)[0]

        max_abs = (grad_custom - grad_ref).abs().max().item()
        ok = max_abs <= tol
        failed += 0 if ok else 1
        label = _shape_label("trilinear", item)
        print(
            f"{'trilinear':10s} {label:30s} {max_abs:12.3e} {'PASS' if ok else 'FAIL':>6s}"
        )
        results.append(
            {
                "mode": "trilinear",
                "shape": item,
                "grad_max_abs": max_abs,
                "status": "PASS" if ok else "FAIL",
            }
        )
        del x, y, grad_out, grad_custom, x_ref, y_ref, grad_ref
        torch.npu.empty_cache()
    print(
        f"gradient precision: {len(results) - failed}/{len(results)} pass, tol={tol:g}"
    )
    return results, failed


def run_backward_perf(shapes):
    results = []
    print("== backward performance ==")
    print(
        f"{'shape':30s} {'bwd triton':>11s} {'bwd native':>11s} {'bwd x':>7s} "
        f"{'fwd+bwd triton':>14s} {'fwd+bwd native':>14s} {'fb x':>7s}"
    )
    for item in shapes:
        _, c, d, h, w, do, ho, wo = item
        size = (do, ho, wo)
        x = torch.randn(
            1, c, d, h, w, device="npu", dtype=torch.float32, requires_grad=True
        )
        y = interpolate(x, size=size, mode="trilinear", align_corners=False)
        grad_out = torch.randn_like(y)

        t_bwd_custom = bench_ms(lambda: _grad_only(y, x, grad_out))
        x_ref = x.detach().clone().requires_grad_(True)
        y_ref = F.interpolate(x_ref, size=size, mode="trilinear", align_corners=False)
        t_bwd_native = bench_ms(lambda: _grad_only(y_ref, x_ref, grad_out))
        bwd_speedup = t_bwd_native / t_bwd_custom if t_bwd_custom > 0 else 0.0

        # Drop retained graphs before measuring a fresh forward+backward.
        y = None
        y_ref = None
        torch.npu.empty_cache()

        t_fb_custom = bench_ms(lambda: _fwd_bwd_custom(x, size, "trilinear", grad_out))
        t_fb_native = bench_ms(
            lambda: _fwd_bwd_native(x_ref, size, "trilinear", grad_out)
        )
        fb_speedup = t_fb_native / t_fb_custom if t_fb_custom > 0 else 0.0

        label = _shape_label("trilinear", item)
        print(
            f"{label:30s} {t_bwd_custom:10.2f}ms {t_bwd_native:10.2f}ms "
            f"{bwd_speedup:6.2f}x {t_fb_custom:13.2f}ms {t_fb_native:13.2f}ms "
            f"{fb_speedup:6.2f}x"
        )
        print(
            f"[perf] interpolate bwd {label} torch={t_bwd_native:.4f}ms "
            f"ascendc={t_bwd_custom:.4f}ms speedup={bwd_speedup:.2f}x"
        )
        print(
            f"[perf] interpolate fwd+bwd {label} torch={t_fb_native:.4f}ms "
            f"ascendc={t_fb_custom:.4f}ms speedup={fb_speedup:.2f}x"
        )
        results.append(
            {
                "mode": "trilinear",
                "shape": item,
                "bwd_triton_ms": t_bwd_custom,
                "bwd_native_ms": t_bwd_native,
                "bwd_speedup": bwd_speedup,
                "fwd_bwd_triton_ms": t_fb_custom,
                "fwd_bwd_native_ms": t_fb_native,
                "fwd_bwd_speedup": fb_speedup,
            }
        )
        x = None
        x_ref = None
        grad_out = None
        torch.npu.empty_cache()
    return results


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("all", "precision", "perf"), default="all")
    p.add_argument("--full", action="store_true", help="include C=256 large shapes")
    p.add_argument("--json", default=None, help="write results to JSON")
    p.add_argument("--tol", type=float, default=TOL)
    p.add_argument(
        "--impl",
        choices=("torch", "triton"),
        default="triton",
        help="triton = Cube GEMM (default); torch = native F.interpolate",
    )
    p.add_argument(
        "--min-bytes",
        type=int,
        default=0,
        help="test-only output threshold override (default: 0 for this harness)",
    )
    p.add_argument(
        "--chunk-bytes",
        type=int,
        default=None,
        help="test-only per-launch output budget override",
    )
    p.add_argument(
        "--allow-fallback",
        action="store_true",
        help="allow the operator to fall back instead of requiring the Cube path",
    )
    # The shared NPU runner appends --build-ext for operators that ship a
    # pybind extension; interpolate is pure Python/Triton, so accept and ignore.
    p.add_argument("--build-ext", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args(argv)

    tol = args.tol
    require_npu()
    os.environ["OIL_GAS_OPS_INTERPOLATE_IMPL"] = args.impl
    # Test-only knobs are process-local; they are not public environment vars.
    ti._set_test_options(
        require_custom=not args.allow_fallback and args.impl == "triton",
        min_bytes=args.min_bytes,
        chunk_bytes=args.chunk_bytes,
    )
    print(f"impl={args.impl} min_bytes={args.min_bytes}")

    shapes = parse_shapes(args)
    payload = {"shapes": len(shapes), "tol": tol}
    failed = 0
    if args.mode in ("all", "precision"):
        precision, failed = run_precision(shapes, tol)
        payload["precision"] = precision
        grad_precision, grad_failed = run_grad_precision(BACKWARD_3D, tol)
        payload["grad_precision"] = grad_precision
        failed += grad_failed
    if args.mode in ("all", "perf"):
        payload["perf"] = run_perf(shapes)
        payload["backward_perf"] = run_backward_perf(BACKWARD_3D)
    if args.json:
        Path(args.json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote {args.json}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
