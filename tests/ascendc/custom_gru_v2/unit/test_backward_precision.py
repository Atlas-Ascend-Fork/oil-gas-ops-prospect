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

"""Unit tests: CustomGRU v2 backward gradient precision.

Fixes from review:
- P0-5: Replaced self-referential max_diff==0.0 with:
  1. torch.autograd.gradcheck (double precision) for true gradient verification
  2. atol/rtol-based comparison instead of exact match
- CPU fallback path: validates autograd wiring + shapes
- NPU path (when available): validates against reference with tolerance
"""
from __future__ import annotations

import torch

from conftest import npu_only, requires_compiled_kernel


def _make_gru_inputs(B, T, H, F_in, seed=42, device="cpu", dtype=torch.float32):
    torch.manual_seed(seed)
    x = torch.randn(B, T, F_in, device=device, dtype=dtype)
    weight_ih = torch.randn(3 * H, F_in, device=device, dtype=dtype) * 0.1
    weight_hh = torch.randn(3 * H, H, device=device, dtype=dtype) * 0.1
    bias_ih = torch.randn(3 * H, device=device, dtype=dtype) * 0.1
    bias_hh = torch.randn(3 * H, device=device, dtype=dtype) * 0.1
    return x, weight_ih, weight_hh, bias_ih, bias_hh


def _torch_gru_ref(x, weight_ih, weight_hh, bias_ih, bias_hh, H):
    B, T, _ = x.shape
    h = torch.zeros(B, H, dtype=x.dtype, device=x.device)
    outputs = []
    for t in range(T):
        xt = x[:, t, :]
        gi = torch.nn.functional.linear(xt, weight_ih, bias_ih)
        gh = torch.nn.functional.linear(h, weight_hh, bias_hh)
        r = torch.sigmoid(gi[:, :H] + gh[:, :H])
        z = torch.sigmoid(gi[:, H:2*H] + gh[:, H:2*H])
        n = torch.tanh(gi[:, 2*H:] + r * gh[:, 2*H:])
        h = (1 - z) * n + z * h
        outputs.append(h)
    y = torch.stack(outputs, dim=1).unsqueeze(0)
    return y, h.unsqueeze(0)


def test_ut_bwd_001_gradcheck_double_precision():
    """True gradient verification using torch.autograd.gradcheck in double precision.

    gradcheck numerically verifies that the analytical backward matches finite differences.
    This replaces the self-referential 'custom == custom' assertion from the original.
    """
    from oil_gas_ops_prospect.custom_gru_v2 import custom_gru_v2

    B, T, H, F_in = 1, 5, 4, 3
    x, wi, wh, bi, bh = _make_gru_inputs(B, T, H, F_in, seed=42, dtype=torch.float64)

    leaves = [t.detach().clone().requires_grad_(True) for t in (x, wi, wh, bi, bh)]

    def func(x, wi, wh, bi, bh):
        y, h = custom_gru_v2(x, wi, wh, bi, bh, H, False)
        return y.sum() + h.sum()

    passed = torch.autograd.gradcheck(func, leaves, eps=1e-6, atol=1e-4, rtol=1e-3)
    assert passed, "gradcheck failed: analytical backward does not match numerical"


def test_ut_bwd_002_gradcheck_batch():
    from oil_gas_ops_prospect.custom_gru_v2 import custom_gru_v2

    B, T, H, F_in = 2, 5, 4, 3
    x, wi, wh, bi, bh = _make_gru_inputs(B, T, H, F_in, seed=99, dtype=torch.float64)
    leaves = [t.detach().clone().requires_grad_(True) for t in (x, wi, wh, bi, bh)]

    def func(x, wi, wh, bi, bh):
        y, h = custom_gru_v2(x, wi, wh, bi, bh, H, False)
        return y.sum() + h.sum()

    passed = torch.autograd.gradcheck(func, leaves, eps=1e-6, atol=1e-4, rtol=1e-3)
    assert passed


@requires_compiled_kernel()
def test_ut_bwd_003_gradient_atol_rtol():
    """Compare custom backward gradients against reference with engineering tolerance.

    Uses atol=1e-4, rtol=1e-3 (not exact 0.0) as recommended by review.
    On CPU this validates autograd wiring; on NPU it validates kernel correctness.
    """
    from oil_gas_ops_prospect.custom_gru_v2 import custom_gru_v2

    B, T, H, F_in = 1, 10, 8, 3
    x, wi, wh, bi, bh = _make_gru_inputs(B, T, H, F_in, seed=7)

    custom_leaves = [t.detach().clone().requires_grad_(True) for t in (x, wi, wh, bi, bh)]
    y_c, h_c = custom_gru_v2(*custom_leaves, H, False)
    grad_y = torch.randn_like(y_c)
    grad_h = torch.randn_like(h_c)
    custom_grads = torch.autograd.grad((y_c, h_c), custom_leaves, (grad_y, grad_h), allow_unused=True)

    ref_leaves = [t.detach().clone().requires_grad_(True) for t in (x, wi, wh, bi, bh)]
    y_r, h_r = _torch_gru_ref(*ref_leaves, H)
    ref_grads = torch.autograd.grad((y_r, h_r), ref_leaves, (grad_y, grad_h), allow_unused=True)

    for i, (cg, rg) in enumerate(zip(custom_grads, ref_grads)):
        name = ["x", "wi", "wh", "bi", "bh"][i]
        if cg is None and rg is None:
            continue
        assert cg is not None and rg is not None, f"grad {name}: custom={cg is not None}, ref={rg is not None}"
        assert torch.allclose(cg, rg, atol=1e-4, rtol=1e-3), \
            f"grad {name}: max_diff={(cg - rg).abs().max().item()}"


def test_ut_bwd_004_weight_grad_nonzero():
    from oil_gas_ops_prospect.custom_gru_v2 import custom_gru_v2
    x, wi, wh, bi, bh = _make_gru_inputs(1, 10, 8, 3, seed=7)
    leaves = [t.detach().clone().requires_grad_(True) for t in (x, wi, wh, bi, bh)]
    y, h = custom_gru_v2(*leaves, 8, False)
    y.sum().backward()
    assert leaves[1].grad is not None and leaves[1].grad.abs().sum() > 0
    assert leaves[2].grad is not None and leaves[2].grad.abs().sum() > 0


def test_ut_bwd_005_bias_grad_nonzero():
    from oil_gas_ops_prospect.custom_gru_v2 import custom_gru_v2
    x, wi, wh, bi, bh = _make_gru_inputs(1, 10, 8, 3, seed=7)
    leaves = [t.detach().clone().requires_grad_(True) for t in (x, wi, wh, bi, bh)]
    y, h = custom_gru_v2(*leaves, 8, False)
    y.sum().backward()
    assert leaves[3].grad is not None and leaves[3].grad.abs().sum() > 0
    assert leaves[4].grad is not None and leaves[4].grad.abs().sum() > 0


def test_ut_bwd_006_cpu_fallback_backward():
    from oil_gas_ops_prospect.custom_gru_v2 import custom_gru_v2
    x, wi, wh, bi, bh = _make_gru_inputs(1, 10, 8, 3, seed=99)
    leaves = [t.detach().clone().requires_grad_(True) for t in (x, wi, wh, bi, bh)]
    y, h = custom_gru_v2(*leaves, 8, False)
    y.sum().backward()
    for i, name in enumerate(["x", "wi", "wh", "bi", "bh"]):
        assert leaves[i].grad is not None, f"{name} grad is None"


@npu_only()
def test_ut_bwd_007_npu_gradient_tolerance():
    """NPU gradient verification with engineering tolerance (not exact 0.0).

    Different hardware/compilation may produce slightly different floating point results.
    Uses atol=1e-3, rtol=1e-2 for NPU path.
    """
    from oil_gas_ops_prospect.custom_gru_v2 import custom_gru_v2
    from conftest import get_device

    device = get_device()
    B, T, H, F_in = 1, 10, 8, 3
    x, wi, wh, bi, bh = _make_gru_inputs(B, T, H, F_in, seed=42, device=device)

    custom_leaves = [t.detach().clone().requires_grad_(True) for t in (x, wi, wh, bi, bh)]
    y_c, h_c = custom_gru_v2(*custom_leaves, H, False)
    grad_y = torch.randn_like(y_c)
    grad_h = torch.randn_like(h_c)
    custom_grads = torch.autograd.grad((y_c, h_c), custom_leaves, (grad_y, grad_h))

    ref_leaves = [t.detach().clone().requires_grad_(True) for t in (x, wi, wh, bi, bh)]
    y_r, h_r = _torch_gru_ref(*ref_leaves, H)
    ref_grads = torch.autograd.grad((y_r, h_r), ref_leaves, (grad_y, grad_h))

    for i, (cg, rg) in enumerate(zip(custom_grads, ref_grads)):
        name = ["x", "wi", "wh", "bi", "bh"][i]
        assert torch.allclose(cg, rg, atol=1e-3, rtol=1e-2), \
            f"NPU grad {name}: max_diff={(cg - rg).abs().max().item()}"
