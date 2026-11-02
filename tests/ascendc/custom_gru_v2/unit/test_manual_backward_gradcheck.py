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

"""Manual GRU backward gradient verification via gradcheck.

Implements the single-step GRU forward + manual backward with the CORRECTED
dh_prev formula (W_hn^T @ (dgate_n * r), not (W_hn^T @ dgate_n) * r) and
verifies via torch.autograd.gradcheck (float64, small shape).

This test was added in response to the second-round expert review (P0-新):
the original documented formula had dh_prev's third backprop path with
matrix-multiply and element-wise-multiply in the wrong order.

Reference formula (corrected):
  dh_prev = dh_raw * z
          + W_hr^T @ dgate_r + W_hz^T @ dgate_z
          + W_hn^T @ (dgate_n * r)          ← corrected: was (W_hn^T @ dgate_n) * r
"""
from __future__ import annotations

import torch


def _gru_single_step_forward(x, h_prev, wi, wh, bi, bh, H):
    gi = torch.nn.functional.linear(x, wi, bi)
    gh = torch.nn.functional.linear(h_prev, wh, bh)
    r = torch.sigmoid(gi[:, :H] + gh[:, :H])
    z = torch.sigmoid(gi[:, H:2*H] + gh[:, H:2*H])
    n = torch.tanh(gi[:, 2*H:] + r * gh[:, 2*H:])
    h_new = (1 - z) * n + z * h_prev
    return h_new, r, z, n, gi, gh


def _gru_manual_backward(x, h_prev, wi, wh, bi, bh, H, grad_h_new):
    """Manual single-step GRU backward with corrected dh_prev formula.

    Returns gradients: dx, dh_prev, dwi, dwh, dbi, dbh

    Key correction (expert review P0-新):
    The n-gate path through h_prev is modulated by r:
      gh_n = h_prev @ wh[2H:3H, :].t() + bh[2H:3H]
      n = tanh(gi_n + r * gh_n)
    So the gradient from dgate_n to dh_prev is:
      (dgate_n * r) @ wh[2H:3H, :]
    NOT (dgate_n @ wh[2H:3H, :]) and NOT ((W_hn^T @ dgate_n) * r).

    Full dh_prev:
      dh_prev = dh_raw * z                          # direct path
              + dgate_r @ wh[:H, :]                 # through r gate
              + dgate_z @ wh[H:2H, :]               # through z gate
              + (dgate_n * r) @ wh[2H:3H, :]        # through n gate (r-modulated)
    """
    h_new, r, z, n, gi, gh = _gru_single_step_forward(x, h_prev, wi, wh, bi, bh, H)

    dh_raw = grad_h_new
    dn = dh_raw * (1 - z)
    dgate_n = dn * (1 - n * n)
    dr = dgate_n * gh[:, 2*H:3*H]
    dgate_r = dr * r * (1 - r)
    dz = dh_raw * (h_prev - n)
    dgate_z = dz * z * (1 - z)

    dgi = torch.cat([dgate_r, dgate_z, dgate_n], dim=1)
    dgh = torch.cat([dgate_r, dgate_z, dgate_n * r], dim=1)

    dx = torch.nn.functional.linear(dgi, wi.t())
    dwi = dgi.t() @ x
    dbi = dgi.sum(dim=0)

    dh_prev = dh_raw * z + dgh @ wh
    dwh = dgh.t() @ h_prev
    dbh = dgh.sum(dim=0)

    return dx, dh_prev, dwi, dwh, dbi, dbh


class _ManualGRUStepFunction(torch.autograd.Function):
    """Wraps manual single-step GRU for gradcheck.

    The backward() implements the corrected formula with
    dh_prev = ... + W_hn^T @ (dgate_n * r)
    instead of the erroneous
    dh_prev = ... + (W_hn^T @ dgate_n) * r
    """

    @staticmethod
    def forward(ctx, x, h_prev, wi, wh, bi, bh, H):
        h_new, r, z, n, gi, gh = _gru_single_step_forward(
            x, h_prev, wi, wh, bi, bh, H
        )
        ctx.save_for_backward(x, h_prev, wi, wh, bi, bh, r, z, n, gh)
        ctx.H = H
        return h_new

    @staticmethod
    def backward(ctx, grad_h_new):
        x, h_prev, wi, wh, bi, bh, r, z, n, gh = ctx.saved_tensors
        H = ctx.H

        dh_raw = grad_h_new
        dn = dh_raw * (1 - z)
        dgate_n = dn * (1 - n * n)
        dr = dgate_n * gh[:, 2*H:3*H]
        dgate_r = dr * r * (1 - r)
        dz = dh_raw * (h_prev - n)
        dgate_z = dz * z * (1 - z)

        dgi = torch.cat([dgate_r, dgate_z, dgate_n], dim=1)
        dgh = torch.cat([dgate_r, dgate_z, dgate_n * r], dim=1)

        dx = torch.nn.functional.linear(dgi, wi.t())
        dwi = dgi.t() @ x
        dbi = dgi.sum(dim=0)

        dh_prev = dh_raw * z + dgh @ wh
        dwh = dgh.t() @ h_prev
        dbh = dgh.sum(dim=0)

        return dx, dh_prev, dwi, dwh, dbi, dbh, None


def test_ut_bwd_008_manual_backward_gradcheck():
    """Verify CORRECTED manual backward formula via gradcheck (float64).

    This test directly validates that the corrected dh_prev formula
    (W_hn^T @ (dgate_n * r)) produces gradients matching finite differences.
    """
    B, H, F_in = 2, 4, 3
    torch.manual_seed(42)
    x = torch.randn(B, F_in, dtype=torch.float64, requires_grad=True)
    h_prev = torch.randn(B, H, dtype=torch.float64, requires_grad=True)
    wi = torch.randn(3 * H, F_in, dtype=torch.float64, requires_grad=True) * 0.1
    wh = torch.randn(3 * H, H, dtype=torch.float64, requires_grad=True) * 0.1
    bi = torch.randn(3 * H, dtype=torch.float64, requires_grad=True) * 0.1
    bh = torch.randn(3 * H, dtype=torch.float64, requires_grad=True) * 0.1

    leaves = [x, h_prev, wi, wh, bi, bh]

    def func(x, h_prev, wi, wh, bi, bh):
        return _ManualGRUStepFunction.apply(x, h_prev, wi, wh, bi, bh, H).sum()

    passed = torch.autograd.gradcheck(func, leaves, eps=1e-6, atol=1e-4, rtol=1e-3)
    assert passed, "gradcheck failed: corrected manual backward does not match numerical"


def test_ut_bwd_009_manual_backward_wrong_formula_fails():
    """Verify that the WRONG v1 formula FAILS gradcheck.

    This tests the EXACT erroneous formula from the v1 response document that
    the expert identified as P0-新:

      dgh = [dgate_r, dgate_z, dgate_n]                    # unmodulated
      dh_prev = dh_raw * z + dgh @ wh
                + (dgate_n @ wh[2H:3H, :]) * r             # ORDER SWAP error

    The error is twofold:
    1. dgh uses unmodulated dgate_n (should be dgate_n * r)
    2. dh_prev adds (dgate_n @ wh[2H:3H,:]) * r — matrix multiply FIRST,
       then element-wise multiply by r — which is (W_hn^T @ dgate_n) ⊙ r
       instead of the correct W_hn^T @ (dgate_n ⊙ r).
    """

    class _WrongGRUStepFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, h_prev, wi, wh, bi, bh, H):
            h_new, r, z, n, gi, gh = _gru_single_step_forward(
                x, h_prev, wi, wh, bi, bh, H
            )
            ctx.save_for_backward(x, h_prev, wi, wh, bi, bh, r, z, n, gh)
            ctx.H = H
            return h_new

        @staticmethod
        def backward(ctx, grad_h_new):
            x, h_prev, wi, wh, bi, bh, r, z, n, gh = ctx.saved_tensors
            H = ctx.H

            dh_raw = grad_h_new
            dn = dh_raw * (1 - z)
            dgate_n = dn * (1 - n * n)
            dr = dgate_n * gh[:, 2*H:3*H]
            dgate_r = dr * r * (1 - r)
            dz = dh_raw * (h_prev - n)
            dgate_z = dz * z * (1 - z)

            dgi = torch.cat([dgate_r, dgate_z, dgate_n], dim=1)

            dx = torch.nn.functional.linear(dgi, wi.t())
            dwi = dgi.t() @ x
            dbi = dgi.sum(dim=0)

            # WRONG (v1 formula): unmodulated dgh + order-swapped extra term
            dgh_wrong = torch.cat([dgate_r, dgate_z, dgate_n], dim=1)
            dh_prev = (
                dh_raw * z
                + dgh_wrong @ wh
                + (dgate_n @ wh[2*H:3*H, :]) * r
            )
            dwh = dgh_wrong.t() @ h_prev
            dbh = dgh_wrong.sum(dim=0)

            return dx, dh_prev, dwi, dwh, dbi, dbh, None

    B, H, F_in = 2, 4, 3
    torch.manual_seed(42)
    x = torch.randn(B, F_in, dtype=torch.float64, requires_grad=True)
    h_prev = torch.randn(B, H, dtype=torch.float64, requires_grad=True)
    wi = torch.randn(3 * H, F_in, dtype=torch.float64, requires_grad=True) * 0.1
    wh = torch.randn(3 * H, H, dtype=torch.float64, requires_grad=True) * 0.1
    bi = torch.randn(3 * H, dtype=torch.float64, requires_grad=True) * 0.1
    bh = torch.randn(3 * H, dtype=torch.float64, requires_grad=True) * 0.1

    leaves = [x, h_prev, wi, wh, bi, bh]

    def func(x, h_prev, wi, wh, bi, bh):
        return _WrongGRUStepFunction.apply(x, h_prev, wi, wh, bi, bh, H).sum()

    try:
        torch.autograd.gradcheck(func, leaves, eps=1e-6, atol=1e-4, rtol=1e-3)
        assert False, "gradcheck should have FAILED for the wrong formula but passed"
    except RuntimeError:
        pass


def test_ut_bwd_010_manual_backward_multi_step():
    """Verify corrected manual backward over multiple time steps (T=3).

    Chains single-step manual backward through time and checks against
    PyTorch autograd. This validates the time-dependency path in dh_prev.
    """
    B, T, H, F_in = 2, 3, 4, 3
    torch.manual_seed(77)
    x = torch.randn(B, T, F_in, dtype=torch.float64, requires_grad=True)
    wi = torch.randn(3 * H, F_in, dtype=torch.float64, requires_grad=True) * 0.1
    wh = torch.randn(3 * H, H, dtype=torch.float64, requires_grad=True) * 0.1
    bi = torch.randn(3 * H, dtype=torch.float64, requires_grad=True) * 0.1
    bh = torch.randn(3 * H, dtype=torch.float64, requires_grad=True) * 0.1

    leaves = [x, wi, wh, bi, bh]

    def func(x, wi, wh, bi, bh):
        outputs = []
        h = torch.zeros(B, H, dtype=torch.float64)
        for t in range(T):
            h = _ManualGRUStepFunction.apply(x[:, t, :], h, wi, wh, bi, bh, H)
            outputs.append(h.sum())
        return sum(outputs)

    passed = torch.autograd.gradcheck(func, leaves, eps=1e-6, atol=1e-4, rtol=1e-3)
    assert passed, "gradcheck failed: multi-step manual backward incorrect"
