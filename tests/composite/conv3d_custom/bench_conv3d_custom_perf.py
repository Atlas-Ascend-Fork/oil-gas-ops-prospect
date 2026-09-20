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

"""conv3d_custom 性能微基准与 NPU 精度对拍：自定义路径 vs 原生 nn.Conv3d。

形态与实际训练提取的 11 种调用模式对齐：

- NPU：按实际训练尺寸（160~296 空间维度）对比，float32；
- CPU：空间维度缩至 --spatial（默认 20），仅用于无卡环境的脚本冒烟，
  通道数保持不变。

模式 4/5（大 kernel depthwise，kernel 7/9 + groups=7 + 大空间）在 NPU 上
原生 backward 会因 L1 buffer 超限崩溃：脚本对原生侧异常做捕获并标注，
自定义侧正常完成。

两种用法：

- 手动基准（默认 bench 模式）：前向与 fwd+bwd 分开计时，输出
  ``[perf] ... torch=..ms ascendc=..ms speedup=..x``（键名与仓内其他 bench
  及开源报告解析器一致）；
- npu_python 套件（`tests/suites.json` 登记，`bash build.sh -u --precision
  / --perf` 触发）：``--mode precision`` 为 NPU 实际训练尺寸 float32 的
  自定义前/反向 vs 原生 L2 相对误差对拍（阈值 ``--rtol``，默认 1e-5），
  ``--mode perf`` 为真机计时。

用法::

    python3 tests/composite/conv3d_custom/bench_conv3d_custom_perf.py            # 手动基准，自动选设备
    python3 tests/composite/conv3d_custom/bench_conv3d_custom_perf.py --mode precision --rtol 1e-5
    python3 tests/composite/conv3d_custom/bench_conv3d_custom_perf.py --device cpu --spatial 24
    python3 tests/composite/conv3d_custom/bench_conv3d_custom_perf.py --patterns train04,train05
"""

from __future__ import annotations

import argparse
import os
import time

import torch
from torch import nn

import _bootstrap  # noqa: F401  (side effect: 源码树里解析 oil_gas_ops_prospect)
from _common import rel_err
from oil_gas_ops_prospect.conv3d_custom import Conv3d

# 实际训练提取的 11 种调用模式（NPU 实际训练尺寸）。
# 用例字段依次为：id、input_shape、weight_shape、S、P、G、bias、padding_mode。
PATTERNS = [
    ("train01_smooth_1x1", (1, 1, 160, 288, 288), (32, 1, 1, 1, 1), 1, 0, 1, True, "zeros"),
    ("train02_proj_1x1", (1, 32, 160, 288, 288), (1, 32, 1, 1, 1), 1, 0, 1, True, "zeros"),
    ("train03_k5_replicate", (1, 7, 164, 292, 292), (7, 7, 5, 5, 5), 1, 0, 1, True, "replicate"),
    ("train04_dw_k7_g7", (1, 7, 166, 294, 294), (7, 1, 7, 7, 7), 1, 0, 7, True, "replicate"),
    ("train05_dw_k9_g7", (1, 7, 168, 296, 296), (7, 1, 9, 9, 9), 1, 0, 7, True, "replicate"),
    ("train06_k3_replicate", (1, 7, 162, 290, 290), (7, 7, 3, 3, 3), 1, 0, 1, True, "replicate"),
    ("train07_stride4_k8_nobias", (1, 7, 168, 296, 296), (19, 7, 8, 8, 8), 4, 0, 1, False, "replicate"),
    ("train08_patch_embed", (1, 19, 41, 73, 73), (512, 19, 1, 1, 1), 1, 0, 1, True, "zeros"),
    ("train09_up_1x1", (1, 512, 41, 73, 73), (1024, 512, 1, 1, 1), 1, 0, 1, True, "zeros"),
    ("train10_down_1x1", (1, 1024, 41, 73, 73), (512, 1024, 1, 1, 1), 1, 0, 1, True, "zeros"),
    ("train11_dw_smooth_g2", (1, 2, 162, 290, 290), (2, 1, 3, 3, 3), 1, 0, 2, True, "replicate"),
]


def _sync() -> None:
    if hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.synchronize()


def bench(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    _sync()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    _sync()
    return (time.perf_counter() - t0) / iters * 1e3


def report(label: str, kind: str, native_ms: float | None, custom_ms: float) -> None:
    if native_ms is None:
        print(f"[perf] {label} {kind} native=CRASH custom={custom_ms:.3f}ms")
        return
    speedup = native_ms / custom_ms if custom_ms > 0 else 0.0
    print(f"[perf] {label} {kind} torch={native_ms:.3f}ms ascendc={custom_ms:.3f}ms speedup={speedup:.2f}x")


def select_patterns(selection: str) -> list:
    """按 id 或 id 前缀选择模式（train01 可匹配 train01_smooth_1x1）。"""
    if selection == "all":
        return list(PATTERNS)
    tokens = [t for t in selection.split(",") if t]
    selected = [
        spec
        for spec in PATTERNS
        if any(spec[0] == t or spec[0].startswith(t + "_") for t in tokens)
    ]
    return selected


def shrink_spatial(shape: tuple[int, ...], limit: int) -> tuple[int, ...]:
    return (*shape[:2], *(min(d, limit) for d in shape[2:]))


def build_pair(spec, device, dtype):
    """按模式构造 (原生 nn.Conv3d, 自定义 Conv3d) 与共享输入。"""
    _, in_shape, w_shape, s, p, g, has_bias, padding_mode = spec
    cin, cout, kernel = in_shape[1], w_shape[0], tuple(w_shape[2:])
    ref = nn.Conv3d(
        cin, cout, kernel, stride=s, padding=p, groups=g, bias=has_bias,
        padding_mode=padding_mode,
    ).to(device=device, dtype=dtype)
    cus = Conv3d(
        cin, cout, kernel, stride=s, padding=p, groups=g, bias=has_bias,
        padding_mode=padding_mode,
    ).to(device=device, dtype=dtype)
    cus.load_state_dict(ref.state_dict())
    cus.use_custom = True
    x = torch.randn(in_shape, device=device, dtype=dtype)
    return ref, cus, x


def run_precision(spec, device, rtol) -> bool:
    """NPU 实际训练尺寸的 float32 精度对拍：自定义前/反向 vs 原生，L2 相对误差。

    模式 4/5（大 kernel depthwise）原生 backward 会因 L1 buffer 超限崩溃：
    该情形下前向仍逐位对拍，自定义反向要求能跑通且梯度有限。
    """
    pid = spec[0]
    ref, cus, x = build_pair(spec, device, torch.float32)
    x_ref = x.detach().clone().requires_grad_(True)
    x_cus = x.detach().clone().requires_grad_(True)
    cus_params = [x_cus, cus.weight] + ([cus.bias] if cus.bias is not None else [])
    ref_params = [x_ref, ref.weight] + ([ref.bias] if ref.bias is not None else [])

    out_ref = ref(x_ref)
    out_cus = cus(x_cus)
    gout = torch.ones_like(out_ref)

    errs = {"fwd": rel_err(out_cus, out_ref)}
    try:
        grads_ref = torch.autograd.grad(out_ref, ref_params, gout)
    except RuntimeError as exc:
        grads_cus = torch.autograd.grad(out_cus, cus_params, gout)
        finite = all(torch.isfinite(t).all().item() for t in grads_cus if t is not None)
        if not finite or rel_err(out_cus, out_ref) > rtol:
            print(f"[precision] {pid} FAIL: 原生反向崩溃（{type(exc).__name__}）且自定义侧异常")
            return False
        print(f"[precision] {pid} PASS: 原生反向崩溃（{type(exc).__name__}，L1 超限），"
              f"自定义反向正常、梯度有限，fwd_rel={errs['fwd']:.2e}")
        return True

    grads_cus = torch.autograd.grad(out_cus, cus_params, gout)
    for key, cus_g, ref_g in zip(("gi", "gw", "gb"), grads_cus, grads_ref):
        if cus_g is not None:
            errs[key] = rel_err(cus_g, ref_g)

    ok = all(v <= rtol for v in errs.values())
    detail = " ".join(f"{k}={v:.2e}" for k, v in errs.items())
    print(f"[precision] {pid} {'PASS' if ok else 'FAIL'}: {detail}")
    return ok


def run_pattern(spec, device, dtype, warmup, iters, spatial_limit):
    pid, in_shape, _, _, _, _, _, _ = spec
    if device.type == "cpu":
        spec = (*spec[:1], shrink_spatial(spec[1], spatial_limit), *spec[2:])
        label = f"{pid}[spatial<={spatial_limit}]"
    else:
        label = f"{pid}[{in_shape[2]}x{in_shape[3]}x{in_shape[4]}]"

    ref, cus, x = build_pair(spec, device, dtype)
    x_ref = x.detach().clone().requires_grad_(True)
    x_cus = x.detach().clone().requires_grad_(True)

    def native_fwd():
        with torch.no_grad():
            ref(x)

    def custom_fwd():
        with torch.no_grad():
            cus(x)

    def native_fb():
        torch.autograd.grad(ref(x_ref).sum(), x_ref)

    def custom_fb():
        torch.autograd.grad(cus(x_cus).sum(), x_cus)

    report(label, "fwd", bench(native_fwd, warmup, iters), bench(custom_fwd, warmup, iters))

    try:
        native_fb_ms = bench(native_fb, warmup, iters)
    except Exception as exc:  # noqa: BLE001  原生 NPU backward 对大 kernel depthwise 崩溃
        print(f"[perf] {label} fwdbwd native CRASH: {type(exc).__name__}（自定义侧不受影响）")
        native_fb_ms = None
    report(label, "fwdbwd", native_fb_ms, bench(custom_fb, warmup, iters))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("bench", "precision", "perf", "all"), default="bench",
                   help="bench=手动基准（本脚本原有行为）；precision/perf 供 npu_python "
                        "套件调用（NPU 实际训练尺寸）；all=precision+perf")
    p.add_argument("--build-ext", action="store_true",
                   help="npu_python 套件协议兼容参数；composite 无 C++ 扩展，忽略")
    p.add_argument("--rtol", type=float, default=1e-5,
                   help="precision 模式的 L2 相对误差阈值")
    p.add_argument("--device", choices=("auto", "cpu", "npu"), default="auto",
                   help="仅 bench 模式生效；precision/perf 固定 NPU")
    p.add_argument("--dtype", choices=("float32", "float64"), default="float32",
                   help="仅 bench 模式生效；precision 固定 float32")
    p.add_argument("--spatial", type=int, default=20,
                   help="CPU 模式下空间维度上限（NPU 模式按实际训练尺寸，不缩放）")
    p.add_argument("--warmup", type=int, default=None, help="默认：npu=5 / cpu=2")
    p.add_argument("--iters", type=int, default=None, help="默认：npu=20 / cpu=5")
    p.add_argument("--patterns", default="all", help="逗号分隔的模式 id，默认 all")
    return p.parse_args(argv)


def resolve_device(name: str) -> torch.device:
    npu_available = hasattr(torch, "npu") and torch.npu.is_available()
    if name == "cpu" or (name == "auto" and not npu_available):
        return torch.device("cpu")
    return torch.device("npu")


def npu_ready() -> bool:
    return hasattr(torch, "npu") and torch.npu.is_available()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.mode != "bench":
        if not npu_ready():
            print(f"[SKIP] 未检测到 NPU，--mode {args.mode} 需要真机环境")
            return 1
        device = torch.device("npu")
        torch.npu.set_device(int(os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "0").split(",")[0] or 0))
    else:
        if args.device == "npu" and not npu_ready():
            print("[SKIP] 未检测到 NPU")
            return 1
        device = resolve_device(args.device)
        if device.type == "npu":
            torch.npu.set_device(int(os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "0").split(",")[0] or 0))

    selected = select_patterns(args.patterns)
    if not selected:
        available = ", ".join(spec[0] for spec in PATTERNS)
        print(f"[ERROR] 未匹配到模式: {args.patterns}（可用: {available}）")
        return 2

    rc = 0
    if args.mode in ("precision", "all"):
        print(f"== conv3d_custom precision (npu, float32, real shapes, rtol={args.rtol}) ==")
        if not all(run_precision(spec, device, args.rtol) for spec in selected):
            rc = 1

    if args.mode in ("perf", "all") or args.mode == "bench":
        warmup = args.warmup if args.warmup is not None else (5 if device.type == "npu" else 2)
        iters = args.iters if args.iters is not None else (20 if device.type == "npu" else 5)
        dtype = torch.float32 if args.dtype == "float32" else torch.float64
        print(f"== conv3d_custom perf (device={device.type}, dtype={args.dtype}, "
              f"warmup {warmup}, iter {iters}, ms/call) ==")
        for spec in selected:
            run_pattern(spec, device, dtype, warmup, iters, args.spatial)

    if rc == 0:
        print("PASS")
    else:
        print("FAIL")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
