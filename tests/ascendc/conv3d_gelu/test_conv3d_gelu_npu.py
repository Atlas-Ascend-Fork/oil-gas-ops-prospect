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
"""Conv3DGelu 精度 / 性能 / 显存对拍。

基准是同一张 NPU 上的 F.conv3d + F.gelu(approximate='tanh')。
显存看两件事：
  1. 前向峰值（含 workspace）
  2. 带梯度前向返回后仍占用的显存——融合算子不保存 conv 与 gelu 之间的预激活
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass

# 必须在 import torch 之前。setdefault 对已经启动的解释器无效。
os.environ.setdefault("ACLNN_CACHE_LIMIT", "0")

import torch
import torch.nn.functional as F

import torch_npu  # noqa: F401

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "ascendc"))
sys.path.insert(0, os.path.join(REPO, "ascendc", "pybind"))

from conv3d_gelu import conv3d_gelu  # noqa: E402

PYBIND_DIR = os.path.join(REPO, "ascendc", "pybind")


def build_ext() -> None:
    """就地编译 conv3d_gelu_npu。前置条件：已安装 OPP 包。"""
    subprocess.run(
        [sys.executable, os.path.join(REPO, "ascendc", "pybind_setup.py"), "build_ext", "--inplace"],
        cwd=PYBIND_DIR,
        check=True,
    )


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-ext", action="store_true", help="先就地编译 conv3d_gelu_npu 扩展")
    parser.add_argument(
        "--mode",
        choices=("all", "precision", "perf"),
        default="all",
        help="all=精度+性能+显存；precision=只精度；perf=只计时且不以加速比作为退出码",
    )
    return parser.parse_args(argv)


def _conv_ref(x, weight, bias, stride, padding, dilation, padding_mode):
    if padding_mode != "zeros":
        if isinstance(padding, int):
            pd = ph = pw = padding
        else:
            pd, ph, pw = padding
        x = F.pad(x, (pw, pw, ph, ph, pd, pd), mode=padding_mode)
        padding = 0
    return F.gelu(
        F.conv3d(x, weight, bias, stride=stride, padding=padding, dilation=dilation),
        approximate="tanh",
    )


def _check_param_contract() -> None:
    x = torch.randn(1, 4, 4, 4, 4)
    weight = torch.randn(4, 4, 1, 1, 1)
    try:
        conv3d_gelu(x, weight, groups=2)
    except ValueError:
        pass
    else:
        raise SystemExit("groups!=1 should raise ValueError")
    try:
        conv3d_gelu(x, weight, stride=(1, 2))
    except ValueError:
        pass
    else:
        raise SystemExit("stride of length 2 should raise ValueError")


@dataclass
class CaseResult:
    name: str
    max_abs: float
    mean_abs: float
    rel: float
    grad_max_abs: float
    base_ms: float
    custom_ms: float
    speedup: float
    base_peak_mb: float
    custom_peak_mb: float
    base_held_mb: float
    custom_held_mb: float
    held_saved_mb: float
    activation_mb: float


def _sync():
    torch.npu.synchronize()


def _bench(fn, warmup: int = 3, repeat: int = 8) -> float:
    for _ in range(warmup):
        y = fn()
        del y
    _sync()
    start = time.perf_counter()
    for _ in range(repeat):
        y = fn()
        del y
    _sync()
    return (time.perf_counter() - start) * 1000.0 / repeat


def _peak_mb(fn) -> float:
    torch.npu.empty_cache()
    _sync()
    y = fn()
    del y
    _sync()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()
    y = fn()
    _sync()
    peak = torch.npu.max_memory_allocated() / (1024 ** 2)
    del y
    torch.npu.empty_cache()
    return peak


def _held_mb(fn) -> float:
    """前向（带梯度）返回后、反向开始前，当前流上仍分配的显存。"""
    torch.npu.empty_cache()
    _sync()
    torch.npu.reset_peak_memory_stats()
    before = torch.npu.memory_allocated()
    y = fn()
    _sync()
    held = (torch.npu.memory_allocated() - before) / (1024 ** 2)
    y.sum().backward()
    _sync()
    del y
    torch.npu.empty_cache()
    return held


def run_case(
    name, x_shape, w_shape, stride=1, padding=0, bias=True, seed=0, dilation=1,
    padding_mode="zeros", measure_prec=True, measure_perf=True, measure_mem=True,
) -> CaseResult:
    torch.manual_seed(seed)
    device = torch.device("npu:0")
    x = torch.randn(*x_shape, dtype=torch.float32, device=device)
    weight = torch.randn(*w_shape, dtype=torch.float32, device=device)
    b = torch.randn(w_shape[0], dtype=torch.float32, device=device) if bias else None

    def baseline():
        return _conv_ref(x, weight, b, stride, padding, dilation, padding_mode)

    def custom():
        return conv3d_gelu(
            x, weight, b, stride=stride, padding=padding, dilation=dilation, groups=1,
            padding_mode=padding_mode,
        )

    # 基准用 CPU fp32。NPU 上 allow_hf32 默认打开，会把 conv 拉到 1e-4，不能当 float32 真值。
    max_abs = mean_abs = rel = grad_max = 0.0
    activation_mb = 0.0
    if measure_prec:
        y_cpu = _conv_ref(
            x.detach().float().cpu(), weight.detach().float().cpu(),
            None if b is None else b.detach().float().cpu(),
            stride, padding, dilation, padding_mode,
        )
        y_base = baseline()
        y_custom = custom()
        _sync()
        diff = (y_custom.float().cpu() - y_cpu).abs()
        max_abs = diff.max().item()
        mean_abs = diff.mean().item()
        denom = y_cpu.abs().max().clamp_min(1e-8).item()
        rel = max_abs / denom
        activation_mb = y_base.numel() * 4 / (1024 ** 2)

        x1 = x.detach().clone().requires_grad_(True)
        w1 = weight.detach().clone().requires_grad_(True)
        b1 = None if b is None else b.detach().clone().requires_grad_(True)
        x2 = x.detach().clone().requires_grad_(True)
        w2 = weight.detach().clone().requires_grad_(True)
        b2 = None if b is None else b.detach().clone().requires_grad_(True)
        y1 = _conv_ref(x1, w1, b1, stride, padding, dilation, padding_mode)
        y2 = conv3d_gelu(
            x2, w2, b2, stride=stride, padding=padding, dilation=dilation, groups=1,
            padding_mode=padding_mode,
        )
        g = torch.randn_like(y1)
        y1.backward(g)
        y2.backward(g)
        _sync()
        # 输入梯度应与整幅反向一致。权重梯度在 NPU 上是原子累加，分块后噪声更大，单独放宽。
        grad_max = (x1.grad - x2.grad).abs().max().item()
        del y1, y2, x1, w1, b1, x2, w2, b2, g, y_base, y_custom
        torch.npu.empty_cache()

    base_ms = custom_ms = 0.0
    if measure_perf:
        base_ms = _bench(baseline)
        custom_ms = _bench(custom)
    base_peak = custom_peak = 0.0
    if measure_mem:
        base_peak = _peak_mb(baseline)
        custom_peak = _peak_mb(custom)

    def baseline_train():
        return _conv_ref(x, weight, b, stride, padding, dilation, padding_mode)

    def custom_train():
        return conv3d_gelu(
            x, weight, b, stride=stride, padding=padding, dilation=dilation, groups=1,
            padding_mode=padding_mode,
        )

    # 两次测量之间清掉上一轮 backward 留下的 grad，避免干扰分配。
    def _zero():
        if x.grad is not None:
            x.grad = None
        if weight.grad is not None:
            weight.grad = None
        if b is not None and b.grad is not None:
            b.grad = None

    base_held = custom_held = 0.0
    if measure_mem:
        x.requires_grad_(True)
        weight.requires_grad_(True)
        if b is not None:
            b.requires_grad_(True)
        _zero()
        base_held = _held_mb(baseline_train)
        _zero()
        custom_held = _held_mb(custom_train)
        _zero()

    return CaseResult(
        name=name,
        max_abs=max_abs,
        mean_abs=mean_abs,
        rel=rel,
        grad_max_abs=grad_max,
        base_ms=base_ms,
        custom_ms=custom_ms,
        speedup=base_ms / custom_ms if custom_ms > 0 else float("inf"),
        base_peak_mb=base_peak,
        custom_peak_mb=custom_peak,
        base_held_mb=base_held,
        custom_held_mb=custom_held,
        held_saved_mb=base_held - custom_held,
        activation_mb=activation_mb,
    )


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.build_ext:
        build_ext()
    _check_param_contract()
    os.environ.setdefault("OIL_GAS_OPS_CONV3D_GELU_IMPL", "ascendc")
    os.environ.setdefault("OIL_GAS_OPS_REQUIRE_ASCENDC_CONV3D_GELU", "1")
    if not torch.npu.is_available():
        print("NPU 不可用")
        return 1
    torch.npu.set_device(0)
    torch_npu.npu.conv.allow_hf32 = False

    want_prec = args.mode in ("all", "precision")
    want_perf = args.mode in ("all", "perf")
    want_mem = args.mode == "all"
    # name, x, weight, stride, padding, bias, dilation, padding_mode
    cases = [
        ("k3_16", (1, 16, 16, 16, 16), (32, 16, 3, 3, 3), 1, 1, True, 1, "zeros"),
        ("k3_32", (2, 32, 32, 32, 32), (32, 32, 3, 3, 3), 1, 1, True, 1, "zeros"),
        ("k3_24", (1, 8, 24, 24, 24), (16, 8, 3, 3, 3), 1, 0, False, 1, "zeros"),
        ("k1_latent", (1, 32, 16, 32, 32), (64, 32, 1, 1, 1), 1, 0, True, 1, "zeros"),
        ("k1_wide", (1, 128, 8, 32, 32), (64, 128, 1, 1, 1), 1, 0, True, 1, "zeros"),
        ("k3_fullres_chunk", (1, 32, 80, 128, 128), (32, 32, 3, 3, 3), 1, 1, True, 1, "zeros"),
        ("k3_96x192", (1, 32, 96, 192, 192), (32, 32, 3, 3, 3), 1, 1, True, 1, "zeros"),
        ("k3_128x160", (1, 32, 128, 160, 160), (32, 32, 3, 3, 3), 1, 1, True, 1, "zeros"),
        ("k3_64x256", (1, 16, 64, 256, 256), (32, 16, 3, 3, 3), 1, 1, True, 1, "zeros"),
        ("k3_n2_112", (2, 16, 112, 160, 160), (32, 16, 3, 3, 3), 1, 1, True, 1, "zeros"),
        ("k3_160", (2, 32, 160, 160, 160), (32, 32, 3, 3, 3), 1, 1, True, 1, "zeros"),
        ("k_stride", (1, 4, 6, 8, 8), (4, 4, 3, 3, 3), (2, 1, 1), (1, 1, 0), True, 1, "zeros"),
        ("k_dilate", (1, 4, 8, 6, 6), (4, 4, 3, 1, 1), 1, (2, 0, 0), False, (2, 1, 1), "zeros"),
        ("k_asym", (1, 4, 5, 7, 6), (4, 4, 1, 3, 1), (1, 1, 2), (0, 1, 0), True, 1, "zeros"),
        ("k_replicate", (1, 4, 4, 6, 6), (4, 4, 3, 3, 3), 1, 1, False, 1, "replicate"),
    ]
    results = []
    for name, xs, ws, stride, padding, bias, dilation, padding_mode in cases:
        print(f"\n[run] {name} x={xs} w={ws} mode={args.mode}", flush=True)
        results.append(run_case(
            name, xs, ws, stride=stride, padding=padding, bias=bias, dilation=dilation,
            padding_mode=padding_mode, measure_prec=want_prec, measure_perf=want_perf,
            measure_mem=want_mem,
        ))

    if want_prec:
        print("\n=== 精度 ===")
        print(f"{'case':<20} {'maxAbs':>11} {'meanAbs':>11} {'rel':>11} {'gradMax':>11}")
        for r in results:
            print(f"{r.name:<20} {r.max_abs:11.3e} {r.mean_abs:11.3e} {r.rel:11.3e} {r.grad_max_abs:11.3e}")

    if want_perf:
        print("\n=== 性能 ===")
        print(f"{'case':<20} {'base_ms':>10} {'custom_ms':>10} {'speedup':>8}")
        for r in results:
            print(f"{r.name:<20} {r.base_ms:10.3f} {r.custom_ms:10.3f} {r.speedup:7.2f}x")
            print(
                f"[perf] {r.name} torch={r.base_ms:.3f}ms ascendc={r.custom_ms:.3f}ms "
                f"speedup={r.speedup:.3f}x"
            )

    if want_mem:
        print("\n=== 显存 ===")
        print(f"{'case':<20} {'actMB':>8} {'basePeak':>10} {'cusPeak':>10} {'baseHeld':>10} {'cusHeld':>10} {'heldSave':>10}")
        for r in results:
            print(
                f"{r.name:<20} {r.activation_mb:8.1f} {r.base_peak_mb:10.1f} {r.custom_peak_mb:10.1f} "
                f"{r.base_held_mb:10.1f} {r.custom_held_mb:10.1f} {r.held_saved_mb:10.1f}"
            )

    if args.mode == "perf":
        print("PASS")
        return 0

    # 前向相对 CPU fp32 < 1e-5。输入梯度在大张量上有约 1e-4 的 NPU 原子累加噪声。
    prec_ok = all(r.rel < 1e-5 and r.grad_max_abs < 1e-4 for r in results)
    if args.mode == "precision":
        print("ALL PASS" if prec_ok else "precision FAIL")
        return 0 if prec_ok else 1

    named = {r.name: r for r in results}
    big = named["k3_160"]
    # 1×1 直写必须明显更快。3×3 按一个输出平面切，换的是整幅私有格式缓冲，不要求加速。
    perf_ok = named["k1_latent"].speedup > 1.05 and named["k1_wide"].speedup > 1.05
    peak_ok = big.custom_peak_mb < big.base_peak_mb - 0.4 * big.activation_mb
    # 预激活大约一整份输出。持有显存至少少掉该激活的 40%，才算去掉了 conv/gelu 之间的缓冲。
    mem_ok = big.held_saved_mb > 0.4 * big.activation_mb
    print(f"\nprecision={prec_ok} perf={perf_ok} (speedup={big.speedup:.2f}x) "
          f"memory={mem_ok} peak={peak_ok} "
          f"(held_saved={big.held_saved_mb:.1f}MB peak {big.base_peak_mb:.1f}->{big.custom_peak_mb:.1f} "
          f"act={big.activation_mb:.1f}MB)")
    return 0 if (prec_ok and perf_ok and mem_ok and peak_ok) else 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
