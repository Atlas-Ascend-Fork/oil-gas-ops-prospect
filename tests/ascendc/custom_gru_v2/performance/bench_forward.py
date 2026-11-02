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

"""Performance benchmark: CustomGRU v2 forward.

Measures forward latency and throughput.

Fixes from review:
- P1-5: Added torch.npu.synchronize() before timing (async NPU)
- P1-新-3: Ratio assertion only meaningful on NPU (on CPU the fallback IS
  the reference implementation, so ratio ≈ 1.0 with zero information).
  On NPU: assert ratio < 1.0 (custom kernel should be faster than Python loop).
  On CPU: skip ratio assertion.
  Throughput renamed to steps/s (was misleadingly counting B*T as "samples").
"""
from __future__ import annotations

import time

import torch

from conftest import maybe_npu_sync, HAS_TORCH_NPU


def _bench_forward(B, T, H, F_in, warmup=2, repeats=5):
    from oil_gas_ops_prospect.custom_gru_v2 import custom_gru_v2
    torch.manual_seed(42)
    x = torch.randn(B, T, F_in)
    wi = torch.randn(3 * H, F_in) * 0.1
    wh = torch.randn(3 * H, H) * 0.1
    bi = torch.randn(3 * H) * 0.1
    bh = torch.randn(3 * H) * 0.1

    for _ in range(warmup):
        y, h = custom_gru_v2(x, wi, wh, bi, bh, H, False)
    maybe_npu_sync()

    times = []
    for _ in range(repeats):
        maybe_npu_sync()
        t0 = time.perf_counter()
        y, h = custom_gru_v2(x, wi, wh, bi, bh, H, False)
        maybe_npu_sync()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    avg_ms = sum(times) / len(times)
    min_ms = min(times)
    throughput = (B * T) / (avg_ms / 1000) if avg_ms > 0 else 0
    return {"avg_ms": avg_ms, "min_ms": min_ms, "throughput": throughput, "B": B, "T": T}


def test_perf_fwd_001_single_long_seq():
    result = _bench_forward(1, 1000, 24, 3)
    print(f"\n[PERF-FWD-001] B=1,T=1000,H=24: avg={result['avg_ms']:.2f}ms, "
          f"min={result['min_ms']:.2f}ms, throughput={result['throughput']:.0f} steps/s")
    assert result["avg_ms"] > 0


def test_perf_fwd_002_batch():
    result = _bench_forward(32, 100, 24, 3)
    print(f"\n[PERF-FWD-002] B=32,T=100,H=24: avg={result['avg_ms']:.2f}ms, "
          f"min={result['min_ms']:.2f}ms, throughput={result['throughput']:.0f} steps/s")
    assert result["avg_ms"] > 0


def test_perf_fwd_003_throughput():
    result = _bench_forward(1, 1000, 24, 3)
    print(f"\n[PERF-FWD-003] throughput={result['throughput']:.0f} steps/s (B*T per second)")
    assert result["throughput"] > 0


def test_perf_fwd_004_vs_puregru_ratio():
    from oil_gas_ops_prospect.custom_gru_v2 import _torch_gru_forward
    B, T, H, F_in = 1, 100, 8, 3
    torch.manual_seed(42)
    x = torch.randn(B, T, F_in)
    wi = torch.randn(3 * H, F_in) * 0.1
    wh = torch.randn(3 * H, H) * 0.1
    bi = torch.randn(3 * H) * 0.1
    bh = torch.randn(3 * H) * 0.1

    for _ in range(3):
        _torch_gru_forward(x, wi, wh, bi, bh, H, False)
    maybe_npu_sync()
    times = []
    for _ in range(10):
        maybe_npu_sync()
        t0 = time.perf_counter()
        _torch_gru_forward(x, wi, wh, bi, bh, H, False)
        maybe_npu_sync()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    pure_ms = sum(times) / len(times)

    custom_result = _bench_forward(B, T, H, F_in, repeats=10)
    ratio = custom_result["avg_ms"] / pure_ms if pure_ms > 0 else float("inf")
    print(f"\n[PERF-FWD-004] custom={custom_result['avg_ms']:.2f}ms vs pure={pure_ms:.2f}ms, "
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
        assert ratio < 1.0, f"Custom is {ratio:.1f}x slower than pure — NPU kernel should be faster"
    else:
        print("[PERF-FWD-004] Fallback mode (no compiled kernel) — ratio assertion skipped (fallback == reference)")
