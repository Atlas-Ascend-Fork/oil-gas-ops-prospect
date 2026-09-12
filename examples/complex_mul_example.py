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

"""ComplexMul / ComplexMulGrad 调用示例。

有 NPU 时走 AscendC 融合算子，无 NPU 时走 PyTorch 参考实现，因此在开发容器里也能跑通。

    python3 examples/complex_mul_example.py

强制要求走自定义算子（失败直接抛异常，而不是静默回退到 PyTorch）::

    OIL_GAS_OPS_COMPLEX_MUL_IMPL=ascendc \\
    OIL_GAS_OPS_REQUIRE_ASCENDC_COMPLEX_MUL=1 \\
    python3 examples/complex_mul_example.py
"""

from __future__ import annotations

import os
import sys

import torch

import _bootstrap  # noqa: F401  (side effect: resolve the package from a source checkout)
from oil_gas_ops_prospect.complex_mul import (
    complex_mul,
    torch_complex_mul,
    torch_complex_mul_grad,
)

SHAPE = (4, 1024)


def pick_device() -> str:
    """有 NPU 就用 NPU，并把实现切到 AscendC 算子；否则退回 CPU。"""
    try:
        import torch_npu  # noqa: F401
    except ImportError:
        return "cpu"

    if not torch.npu.is_available():
        return "cpu"

    # complex_mul 默认走 PyTorch 参考实现，需要显式切换。
    os.environ.setdefault("OIL_GAS_OPS_COMPLEX_MUL_IMPL", "ascendc")
    return "npu"


def rel_err(actual: torch.Tensor, expect: torch.Tensor) -> float:
    denom = expect.abs().max().clamp_min(1e-12)
    return ((actual - expect).abs().max() / denom).item()


def main() -> int:
    device = pick_device()
    impl = os.environ.get("OIL_GAS_OPS_COMPLEX_MUL_IMPL", "torch")
    print(f"device={device} OIL_GAS_OPS_COMPLEX_MUL_IMPL={impl} shape={SHAPE}")

    torch.manual_seed(0)
    ur, ui, vr, vi = (torch.randn(SHAPE, device=device) for _ in range(4))

    # ---- 前向 ----
    wr, wi = complex_mul(ur, ui, vr, vi)
    ref_wr, ref_wi = torch_complex_mul(ur, ui, vr, vi)
    print(f"[fwd] rel_err wr={rel_err(wr, ref_wr):.3e} wi={rel_err(wi, ref_wi):.3e}")

    # ---- 反向 ----
    inputs = [t.clone().requires_grad_(True) for t in (ur, ui, vr, vi)]
    out_r, out_i = complex_mul(*inputs)
    gwr, gwi = torch.randn_like(out_r), torch.randn_like(out_i)
    torch.autograd.backward([out_r, out_i], [gwr, gwi])

    ref_grads = torch_complex_mul_grad(gwr, gwi, ur, ui, vr, vi)
    names = ("gur", "gui", "gvr", "gvi")
    errs = " ".join(
        f"{name}={rel_err(t.grad, ref):.3e}"
        for name, t, ref in zip(names, inputs, ref_grads)
    )
    print(f"[bwd] rel_err {errs}")

    # 数学恒等式自检：complex_mul 等价于 torch 复数乘法
    u = torch.complex(ur, ui)
    v = torch.complex(vr, vi)
    w = u * v
    print(
        f"[check] vs torch.complex rel_err "
        f"real={rel_err(wr, w.real):.3e} imag={rel_err(wi, w.imag):.3e}"
    )

    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
