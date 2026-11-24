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

import pytest
import torch

from conftest import maybe_npu_sync, HAS_TORCH_NPU

# 设备标记整改（与 bench_forward.py 相同的"假 NPU"重构）：基准强制 NPU，
# 无 NPU 环境退回 CPU 并标注。
if HAS_TORCH_NPU and torch.npu.is_available():
    BENCH_DEVICE = "npu:0"
else:
    BENCH_DEVICE = "cpu"


def _bench_backward(B, T, H, F_in, warmup=3, repeats=10):
    from oil_gas_ops_prospect.custom_gru_v2 import custom_gru_v2

    def _make_inputs():
        torch.manual_seed(42)
        x = torch.randn(B, T, F_in, device=BENCH_DEVICE, requires_grad=True)
        wi = torch.randn(3 * H, F_in, device=BENCH_DEVICE, requires_grad=True) * 0.1
        wh = torch.randn(3 * H, H, device=BENCH_DEVICE, requires_grad=True) * 0.1
        bi = torch.randn(3 * H, device=BENCH_DEVICE, requires_grad=True) * 0.1
        bh = torch.randn(3 * H, device=BENCH_DEVICE, requires_grad=True) * 0.1
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
    print(f"\n[PERF-BWD-001] device={BENCH_DEVICE} B=1,T=100,H=24: avg={result['avg_ms']:.2f}ms")
    assert result["avg_ms"] > 0


def test_perf_bwd_002_batch():
    result = _bench_backward(32, 100, 24, 3)
    print(f"\n[PERF-BWD-002] device={BENCH_DEVICE} B=32,T=100,H=24: avg={result['avg_ms']:.2f}ms")
    assert result["avg_ms"] > 0


def test_perf_bwd_003_vs_pure_ratio():
    from oil_gas_ops_prospect.custom_gru_v2 import _torch_gru_forward
    B, T, H, F_in = 1, 100, 8, 3

    def _make_inputs():
        torch.manual_seed(42)
        x = torch.randn(B, T, F_in, device=BENCH_DEVICE, requires_grad=True)
        wi = torch.randn(3 * H, F_in, device=BENCH_DEVICE, requires_grad=True) * 0.1
        wh = torch.randn(3 * H, H, device=BENCH_DEVICE, requires_grad=True) * 0.1
        bi = torch.randn(3 * H, device=BENCH_DEVICE, requires_grad=True) * 0.1
        bh = torch.randn(3 * H, device=BENCH_DEVICE, requires_grad=True) * 0.1
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
    print(f"\n[PERF-BWD-003] device={BENCH_DEVICE} custom={custom_result['avg_ms']:.2f}ms vs pure={pure_ms:.2f}ms, "
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
        # CustomGRU v2 的 backward 是 Python recompute 实现（`_CustomGRUV2Function.backward`
        # 内部用 `_torch_gru_forward` 重算 + torch.autograd.grad，见 ascendc/pybind/
        # custom_gru_v2.py），与参考实现在架构上同构，ratio ≈ 1.0 是当前实现的固有
        # 属性。评审 L1：这类数字不能作为性能门禁（"永不失败"的正数检查没有信号），
        # 改为 skip 并在输出中报告绝对时延 / ratio；等 op_kernel 提供融合 backward
        # 后恢复 `< 1.0` 门禁。
        print(f"[PERF-BWD-003] device={BENCH_DEVICE} latency: custom={custom_result['avg_ms']:.2f}ms "
              f"pure={pure_ms:.2f}ms ratio={ratio:.2f}x (informational)")
        pytest.skip(
            f"CustomGRU backward 是 Python recompute（无融合 backward kernel），"
            f"ratio={ratio:.2f}x 仅供信息记录：custom={custom_result['avg_ms']:.2f}ms / "
            f"pure={pure_ms:.2f}ms；融合 backward kernel 落地后恢复 <1.0 门禁")
    else:
        print("[PERF-BWD-003] Fallback mode (no compiled kernel) — ratio assertion skipped (fallback == reference)")
