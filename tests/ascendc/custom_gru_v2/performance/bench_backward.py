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

"""Performance benchmark: CustomGRU v2 backward.

Measures backward (gradient computation) latency.

Fixes from review:
- P1-5: Added torch.npu.synchronize() before timing
- P1-新-3: Ratio assertion only on NPU (on CPU fallback == reference).
  On NPU: assert ratio < 1.0 (custom should be faster).
  On CPU: skip ratio assertion.
- Round-3: Fix retain_graph issue — recreate leaf tensors each iteration
  to avoid "backward through graph a second time" error.
"""
from __future__ import annotations

import time

import torch

from conftest import maybe_npu_sync, HAS_TORCH_NPU


def _bench_backward(B, T, H, F_in, warmup=3, repeats=10):
    from oil_gas_ops_prospect.custom_gru_v2 import custom_gru_v2

    def _make_inputs():
        torch.manual_seed(42)
        x = torch.randn(B, T, F_in, requires_grad=True)
        wi = torch.randn(3 * H, F_in, requires_grad=True) * 0.1
        wh = torch.randn(3 * H, H, requires_grad=True) * 0.1
        bi = torch.randn(3 * H, requires_grad=True) * 0.1
        bh = torch.randn(3 * H, requires_grad=True) * 0.1
        return x, wi, wh, bi, bh

    for _ in range(warmup):
        x, wi, wh, bi, bh = _make_inputs()
        y, h = custom_gru_v2(x, wi, wh, bi, bh, H, False)
        loss = y.sum() + h.sum()
        loss.backward()
    maybe_npu_sync()

    times = []
    for _ in range(repeats):
        x, wi, wh, bi, bh = _make_inputs()
        maybe_npu_sync()
        t0 = time.perf_counter()
        y, h = custom_gru_v2(x, wi, wh, bi, bh, H, False)
        loss = y.sum() + h.sum()
        loss.backward()
        maybe_npu_sync()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    avg_ms = sum(times) / len(times)
    return {"avg_ms": avg_ms, "min_ms": min(times)}


def test_perf_bwd_001_single():
    result = _bench_backward(1, 100, 24, 3)
    print(f"\n[PERF-BWD-001] B=1,T=100,H=24: avg={result['avg_ms']:.2f}ms")
    assert result["avg_ms"] > 0


def test_perf_bwd_002_batch():
    result = _bench_backward(32, 100, 24, 3)
    print(f"\n[PERF-BWD-002] B=32,T=100,H=24: avg={result['avg_ms']:.2f}ms")
    assert result["avg_ms"] > 0


def test_perf_bwd_003_vs_pure_ratio():
    from oil_gas_ops_prospect.custom_gru_v2 import _torch_gru_forward
    B, T, H, F_in = 1, 100, 8, 3

    def _make_inputs():
        torch.manual_seed(42)
        x = torch.randn(B, T, F_in, requires_grad=True)
        wi = torch.randn(3 * H, F_in, requires_grad=True) * 0.1
        wh = torch.randn(3 * H, H, requires_grad=True) * 0.1
        bi = torch.randn(3 * H, requires_grad=True) * 0.1
        bh = torch.randn(3 * H, requires_grad=True) * 0.1
        return x, wi, wh, bi, bh

    for _ in range(3):
        x, wi, wh, bi, bh = _make_inputs()
        y, h = _torch_gru_forward(x, wi, wh, bi, bh, H, False)
        (y.sum() + h.sum()).backward()
    maybe_npu_sync()
    times_pure = []
    for _ in range(10):
        x, wi, wh, bi, bh = _make_inputs()
        maybe_npu_sync()
        t0 = time.perf_counter()
        y, h = _torch_gru_forward(x, wi, wh, bi, bh, H, False)
        (y.sum() + h.sum()).backward()
        maybe_npu_sync()
        t1 = time.perf_counter()
        times_pure.append((t1 - t0) * 1000)
    pure_ms = sum(times_pure) / len(times_pure)
    custom_result = _bench_backward(B, T, H, F_in, repeats=5)
    ratio = custom_result["avg_ms"] / pure_ms if pure_ms > 0 else float("inf")
    print(f"\n[PERF-BWD-003] custom={custom_result['avg_ms']:.2f}ms vs pure={pure_ms:.2f}ms, "
          f"ratio={ratio:.2f}x")
    if HAS_TORCH_NPU and torch.npu.is_available():
        try:
            from oil_gas_ops_prospect.custom_gru_v2 import _load_extension
            _load_extension()
            has_custom_kernel = True
        except (RuntimeError, ImportError):
            has_custom_kernel = False
    else:
        has_custom_kernel = False

    if has_custom_kernel:
        assert ratio < 1.0, f"Custom backward is {ratio:.1f}x slower — NPU should be faster"
    else:
        print("[PERF-BWD-003] Fallback mode (no compiled kernel) — ratio assertion skipped (fallback == reference)")
