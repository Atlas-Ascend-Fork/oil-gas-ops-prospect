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

"""Unit tests: CustomGRU v2 forward precision.

Validates custom kernel output against PyTorch GRUCell reference.
All tests have CPU fallback path; NPU path activates when torch_npu is available.

Fixes from review:
- P0-6: bidirectional reference now flips time dim (flip(1)), not batch dim (flip(0))
- P1-3: bidirectional uses independent forward/reverse weights (offset into weight tensors)
- P0-1: num_layers removed from API; no multi-layer test
- P0-2: h0 removed from API; initial hidden state is always zero
- P1-2: bias=False path tested with zero bias fallback
"""
from __future__ import annotations

import torch

from conftest import npu_only, requires_compiled_kernel, make_gru_inputs as _make_gru_inputs, torch_gru_ref as _torch_gru_ref


def _run_forward(B, T, H, F_in, seed=42, bidir=False, bias=True, threshold=0.01, device="cpu"):
    from oil_gas_ops_prospect.custom_gru_v2 import custom_gru_v2
    x, wi, wh, bi, bh = _make_gru_inputs(B, T, H, F_in, seed=seed, device=device, bidir=bidir, bias=bias)
    y_ref, h_ref = _torch_gru_ref(x, wi, wh, bi, bh, H, bidir=bidir)

    wi_flat = wi.reshape(-1, F_in) if bidir else wi[0]
    wh_flat = wh.reshape(-1, H) if bidir else wh[0]
    bi_flat = bi.reshape(-1) if bidir else bi[0]
    bh_flat = bh.reshape(-1) if bidir else bh[0]

    y_custom, h_custom = custom_gru_v2(x, wi_flat, wh_flat, bi_flat, bh_flat, H, bidir)
    assert y_custom.shape == y_ref.shape, f"y shape {y_custom.shape} != {y_ref.shape}"
    assert h_custom.shape == h_ref.shape, f"h shape {h_custom.shape} != {h_ref.shape}"
    max_diff_y = (y_custom - y_ref).abs().max().item()
    max_diff_h = (h_custom - h_ref).abs().max().item()
    assert max_diff_y <= threshold, f"y max_diff={max_diff_y} > {threshold}"
    assert max_diff_h <= threshold, f"h max_diff={max_diff_h} > {threshold}"
    return max_diff_y, max_diff_h


@requires_compiled_kernel()
def test_ut_fwd_001_minimal_shape():
    _run_forward(1, 1, 1, 1)


@requires_compiled_kernel()
def test_ut_fwd_002_standard_shape():
    _run_forward(1, 100, 24, 3)


@requires_compiled_kernel()
def test_ut_fwd_003_long_sequence():
    max_diff, _ = _run_forward(1, 1000, 24, 3, threshold=0.003)
    assert max_diff <= 0.003


@requires_compiled_kernel()
def test_ut_fwd_004_batch32():
    _run_forward(32, 100, 24, 3)


@requires_compiled_kernel()
def test_ut_fwd_005_aligned_h8():
    _run_forward(1, 50, 8, 3)


@requires_compiled_kernel()
def test_ut_fwd_006_unaligned_h7():
    _run_forward(1, 50, 7, 3)


@requires_compiled_kernel()
def test_ut_fwd_007_bidirectional():
    _run_forward(1, 50, 8, 3, bidir=True)


@requires_compiled_kernel()
def test_ut_fwd_008_zero_input():
    from oil_gas_ops_prospect.custom_gru_v2 import custom_gru_v2
    H, F_in = 8, 3
    x = torch.zeros(1, 10, F_in)
    wi = torch.randn(3 * H, F_in) * 0.1
    wh = torch.randn(3 * H, H) * 0.1
    bi = torch.randn(3 * H) * 0.1
    bh = torch.randn(3 * H) * 0.1
    y, h = custom_gru_v2(x, wi, wh, bi, bh, H, False)
    assert not torch.isnan(y).any()
    assert not torch.isnan(h).any()
    assert not torch.isinf(y).any()


@requires_compiled_kernel()
def test_ut_fwd_009_reproducible_seed():
    diff1, _ = _run_forward(4, 100, 16, 8, seed=42)
    diff2, _ = _run_forward(4, 100, 16, 8, seed=42)
    assert abs(diff1 - diff2) < 1e-10


def test_ut_fwd_010_cpu_fallback():
    from oil_gas_ops_prospect.custom_gru_v2 import custom_gru_v2
    x, wi, wh, bi, bh = _make_gru_inputs(1, 10, 8, 3, seed=11)
    y_ref, h_ref = _torch_gru_ref(x, wi, wh, bi, bh, 8)
    y, h = custom_gru_v2(x, wi[0], wh[0], bi[0], bh[0], 8, False)
    assert torch.allclose(y, y_ref, rtol=1e-6, atol=1e-6)
    assert torch.allclose(h, h_ref, rtol=1e-6, atol=1e-6)


@requires_compiled_kernel()
def test_ut_fwd_011_bias_false():
    from oil_gas_ops_prospect.custom_gru_v2 import custom_gru_v2
    H, F_in = 8, 3
    x = torch.randn(1, 10, F_in)
    wi = torch.randn(3 * H, F_in) * 0.1
    wh = torch.randn(3 * H, H) * 0.1
    zero_bias = torch.zeros(3 * H)
    y_ref, h_ref = _torch_gru_ref(x, wi.unsqueeze(0), wh.unsqueeze(0), zero_bias.unsqueeze(0), zero_bias.unsqueeze(0), H)
    y, h = custom_gru_v2(x, wi, wh, zero_bias, zero_bias, H, False)
    assert torch.allclose(y, y_ref, rtol=1e-6, atol=1e-6)


@requires_compiled_kernel()
def test_ut_fwd_012_bidirectional_batch():
    _run_forward(4, 50, 8, 3, bidir=True)


@npu_only()
def test_ut_fwd_013_npu_forward():
    _run_forward(1, 100, 24, 3, device="npu:0", threshold=0.5)
