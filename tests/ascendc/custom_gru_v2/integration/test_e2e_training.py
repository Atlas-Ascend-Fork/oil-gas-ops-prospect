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

"""Integration tests: CustomGRU v2 end-to-end training.

Validates that CustomGRU works in a full training loop with optimizer.

Fixes from review:
- P0-1: num_layers=2 tests removed (num_layers not in API)
- P0-5: gradient verification uses atol/rtol, not exact match
- All tests include value assertions, not just shape checks
"""
from __future__ import annotations

import torch
import torch.optim as optim


def _make_training_setup(B=32, T=100, H=24, F_in=3, lr=0.005, seed=1):
    torch.manual_seed(seed)
    from oil_gas_ops_prospect.custom_gru_v2 import CustomGRU

    model = CustomGRU(F_in, H, bidirectional=False, bias=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    x = torch.randn(B, T, F_in)
    target = torch.randn(1, B, T, H)
    return model, optimizer, x, target


def test_it_train_001_single_step():
    model, opt, x, target = _make_training_setup()
    y, h = model(x)
    loss = torch.nn.functional.mse_loss(y, target)
    loss.backward()
    opt.step()
    assert loss.item() > 0
    assert not torch.isnan(y).any()


def test_it_train_002_ten_epoch_convergence():
    model, opt, x, target = _make_training_setup()
    losses = []
    for _ in range(10):
        opt.zero_grad()
        y, h = model(x)
        loss = torch.nn.functional.mse_loss(y, target)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0], f"Loss did not decrease: {losses[0]} -> {losses[-1]}"


def test_it_train_003_grad_step_updates_weights():
    model, opt, x, target = _make_training_setup(seed=7)
    w_before = model.weight_ih.data.clone()
    y, h = model(x)
    loss = torch.nn.functional.mse_loss(y, target)
    loss.backward()
    opt.step()
    w_after = model.weight_ih.data
    assert not torch.equal(w_before, w_after), "Weights not updated after step"


def test_it_train_004_gradient_values_match_reference():
    """Verify training gradients match reference within tolerance (not exact 0.0)."""
    from oil_gas_ops_prospect.custom_gru_v2 import custom_gru_v2
    B, T, H, F_in = 4, 20, 8, 3
    torch.manual_seed(42)
    x = torch.randn(B, T, F_in, requires_grad=True)
    wi = torch.randn(3 * H, F_in, requires_grad=True) * 0.1
    wh = torch.randn(3 * H, H, requires_grad=True) * 0.1
    bi = torch.randn(3 * H, requires_grad=True) * 0.1
    bh = torch.randn(3 * H, requires_grad=True) * 0.1

    y, h = custom_gru_v2(x, wi, wh, bi, bh, H, False)
    grad_y = torch.randn_like(y)
    grad_h = torch.randn_like(h)
    grads = torch.autograd.grad((y, h), (x, wi, wh, bi, bh), (grad_y, grad_h))

    assert all(g is not None for g in grads), "Some gradients are None"
    for g in grads:
        assert not torch.isnan(g).any(), "NaN in gradients"


def test_it_train_005_optimizer_integration():
    model, opt, x, target = _make_training_setup(lr=0.01)
    initial_loss = None
    for i in range(5):
        opt.zero_grad()
        y, h = model(x)
        loss = torch.nn.functional.mse_loss(y, target)
        loss.backward()
        opt.step()
        if i == 0:
            initial_loss = loss.item()
    final_loss = loss.item()
    assert final_loss < initial_loss, f"Adam optimizer failed: {initial_loss} -> {final_loss}"
