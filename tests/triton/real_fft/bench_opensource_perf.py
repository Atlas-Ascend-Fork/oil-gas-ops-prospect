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

"""real_rfft / real_irfft 微基准，形态与开源报告 5.3 对齐。"""

from __future__ import annotations

import os
import time

import torch

import _bootstrap  # noqa: F401  (side effect: 源码树里解析 oil_gas_ops_prospect)
from oil_gas_ops_prospect.real_fft import real_irfft, real_rfft


def _sync() -> None:
    if hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.synchronize()


def bench(fn, n_warm: int = 5, n_iter: int = 20) -> float:
    for _ in range(n_warm):
        fn()
    _sync()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        fn()
    _sync()
    return (time.perf_counter() - t0) / n_iter * 1e3


def report(label: str, torch_ms: float, op_ms: float) -> None:
    speedup = torch_ms / op_ms if op_ms > 0 else 0.0
    print(f"[perf] {label} torch={torch_ms:.3f}ms ascendc={op_ms:.3f}ms speedup={speedup:.2f}x")


def _axis(dim: int, rank: int) -> int:
    return dim if dim >= 0 else rank + dim


def bench_pair(shape: tuple[int, ...], dim: int, tag: str) -> None:
    x = torch.randn(*shape, device="npu", dtype=torch.float32)
    axis = _axis(dim, x.ndim)
    n = x.shape[axis]
    freq_shape = (*x.shape[:axis], n // 2 + 1, *x.shape[axis + 1:])
    xr = torch.randn(*freq_shape, device="npu", dtype=torch.float32)
    xi = torch.randn(*freq_shape, device="npu", dtype=torch.float32)
    x_grad = x.detach().clone().requires_grad_(True)

    def fused_rfft():
        with torch.no_grad():
            real_rfft(x, dim=axis, norm="ortho")

    def torch_rfft():
        with torch.no_grad():
            torch.fft.rfft(x, dim=axis, norm="ortho")

    def fused_irfft():
        with torch.no_grad():
            real_irfft(xr, xi, n=n, dim=axis, norm="ortho")

    def torch_irfft():
        with torch.no_grad():
            torch.fft.irfft(torch.complex(xr, xi), n=n, dim=axis, norm="ortho")

    def fused_fwdbwd():
        out_r, out_i = real_rfft(x_grad, dim=axis, norm="ortho")
        torch.autograd.grad(out_r.sum() + out_i.sum(), x_grad)

    def torch_fwdbwd():
        spec = torch.fft.rfft(x_grad, dim=axis, norm="ortho")
        torch.autograd.grad(spec.real.sum() + spec.imag.sum(), x_grad)

    report(f"rfft {tag}", bench(torch_rfft), bench(fused_rfft))
    report(f"irfft {tag}", bench(torch_irfft), bench(fused_irfft))
    if tag.startswith("5d "):
        report(f"fwd+bwd {tag}", bench(torch_fwdbwd), bench(fused_fwdbwd))


def main() -> int:
    try:
        import torch_npu  # noqa: F401
    except ImportError:
        print("[SKIP] torch_npu 不可用")
        return 1
    if not (hasattr(torch, "npu") and torch.npu.is_available()):
        print("[SKIP] 未检测到 NPU")
        return 1

    torch.npu.set_device(int(os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "0").split(",")[0] or 0))
    print("== fft opensource perf (ms/call, warmup 5, iter 20) ==")
    bench_pair((8, 256), -1, "[8,256] 末轴")
    bench_pair((32, 512), -1, "[32,512] 末轴")
    bench_pair((128, 128, 128), -1, "128³ 末轴")
    bench_pair((128, 128, 128), 0, "128³ 非末轴")
    bench_pair((1, 35, 128, 128, 128), -1, "5d W")
    bench_pair((1, 35, 128, 128, 128), -2, "5d H")
    bench_pair((1, 35, 128, 128, 128), -3, "5d D")
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
