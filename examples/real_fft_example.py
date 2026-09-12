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

"""real_rfft / real_irfft 调用示例。

有 NPU 时走 Triton kernel，无 NPU 时自动回退到 torch.fft，因此在开发容器里也能跑通。

    python3 examples/real_fft_example.py

未安装 whl 时自动回落到仓内源码，无需设置 PYTHONPATH。
"""

from __future__ import annotations

import sys

import torch

import _bootstrap  # noqa: F401  (side effect: resolve the package from a source checkout)
from oil_gas_ops_prospect.real_fft import real_irfft, real_rfft

BATCH = 8
N = 256
DIM = -1


def pick_device() -> str:
    try:
        import torch_npu  # noqa: F401
    except ImportError:
        return "cpu"
    return "npu" if torch.npu.is_available() else "cpu"


def rel_err(actual: torch.Tensor, expect: torch.Tensor) -> float:
    denom = expect.abs().max().clamp_min(1e-12)
    return ((actual - expect).abs().max() / denom).item()


def main() -> int:
    device = pick_device()
    print(f"device={device} shape=({BATCH}, {N}) dim={DIM} norm='ortho'")

    torch.manual_seed(0)
    x = torch.randn(BATCH, N, device=device)

    # ---- 正向：实数 -> 半谱实部/虚部 ----
    xr, xi = real_rfft(x, dim=DIM)
    print(f"[rfft] {tuple(x.shape)} -> real{tuple(xr.shape)} imag{tuple(xi.shape)}")
    assert xr.shape[-1] == N // 2 + 1, "频率轴长度应为 n // 2 + 1"

    # 与 torch.fft 对拍
    ref = torch.fft.rfft(x.float(), dim=DIM, norm="ortho")
    print(f"[rfft] vs torch.fft rel_err real={rel_err(xr, ref.real):.3e} imag={rel_err(xi, ref.imag):.3e}")

    # ---- 反向：半谱 -> 实数，往返重建 ----
    y = real_irfft(xr, xi, n=N, dim=DIM)
    print(f"[irfft] round-trip rel_err={rel_err(y, x):.3e}")

    # ---- autograd ----
    xg = x.clone().requires_grad_(True)
    gr, gi = real_rfft(xg, dim=DIM)
    (gr.pow(2).sum() + gi.pow(2).sum()).backward()
    print(f"[autograd] grad shape={tuple(xg.grad.shape)} finite={bool(torch.isfinite(xg.grad).all())}")

    # ---- 非法入参 ----
    try:
        real_rfft(x, dim=DIM, norm="backward")
    except ValueError as e:
        print(f"[check] norm='backward' 被正确拒绝: {e}")
    else:
        print("[check] FAILED: norm='backward' 未报错")
        return 1

    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
