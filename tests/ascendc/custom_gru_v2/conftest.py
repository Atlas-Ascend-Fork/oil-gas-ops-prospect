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

from __future__ import annotations

import pytest
import torch


def _try_import_torch_npu():
    try:
        import torch_npu  # noqa: F401
        return True
    except ImportError:
        return False


HAS_TORCH_NPU = _try_import_torch_npu()


def _has_npu_hw():
    return HAS_TORCH_NPU and torch.npu.is_available()


def _has_compiled_kernel():
    """Check if the compiled NPU extension (custom_gru_v2_npu) is loadable.

    On CPU or when the extension is not installed, returns False.
    Tests that compare the custom kernel against a reference should skip
    when this returns False — otherwise they compare the PyTorch fallback
    against another PyTorch implementation (self-proving, P1-2).
    """
    if not _has_npu_hw():
        return False
    try:
        import importlib
        importlib.import_module("custom_gru_v2_npu")
        return True
    except (ImportError, RuntimeError, OSError):
        return False


def npu_only():
    return pytest.mark.skipif(
        not _has_npu_hw(),
        reason="requires NPU hardware (not just torch_npu import)",
    )


def npu_hardware_only():
    return pytest.mark.skipif(
        not (HAS_TORCH_NPU and torch.npu.is_available()),
        reason="requires NPU hardware (not just torch_npu import)",
    )


def requires_compiled_kernel():
    return pytest.mark.skipif(
        not _has_compiled_kernel(),
        reason="requires compiled NPU kernel — CPU fallback is same as reference (P1-2)",
    )


def maybe_npu_sync():
    if HAS_TORCH_NPU and torch.npu.is_available():
        import torch_npu  # noqa: F401
        torch.npu.synchronize()


def get_device():
    if HAS_TORCH_NPU and torch.npu.is_available():
        return "npu:0"
    return "cpu"


def make_gru_inputs(B, T, H, F_in, seed=42, device="cpu", bidir=False, bias=True):
    torch.manual_seed(seed)
    num_dirs = 2 if bidir else 1
    x = torch.randn(B, T, F_in, device=device)
    weight_ih = torch.randn(num_dirs, 3 * H, F_in, device=device) * 0.1
    weight_hh = torch.randn(num_dirs, 3 * H, H, device=device) * 0.1
    if bias:
        bias_ih = torch.randn(num_dirs, 3 * H, device=device) * 0.1
        bias_hh = torch.randn(num_dirs, 3 * H, device=device) * 0.1
    else:
        bias_ih = torch.zeros(num_dirs, 3 * H, device=device)
        bias_hh = torch.zeros(num_dirs, 3 * H, device=device)
    return x, weight_ih, weight_hh, bias_ih, bias_hh


def torch_gru_ref(x, weight_ih, weight_hh, bias_ih, bias_hh, H, bidir=False):
    """Reference GRU using PyTorch ops, with correct bidirectional semantics.

    Matches _torch_gru_forward convention:
    - Direction 0 processes x[:, 0, 1, ..., T-1] (forward)
    - Direction 1 processes x[:, T-1, T-2, ..., 0] (backward)
    - Output y[0, :, t, :] = forward hidden state at time t
    - Output y[1, :, t, :] = backward hidden state at time t
    - Final h[0] = forward final, h[1] = backward final
    """
    B, T, _ = x.shape
    num_dirs = 2 if bidir else 1
    h = torch.zeros(num_dirs, B, H, dtype=x.dtype, device=x.device)

    fwd_outputs = []
    for t in range(T):
        gi = torch.nn.functional.linear(x[:, t, :], weight_ih[0], bias_ih[0])
        gh = torch.nn.functional.linear(h[0], weight_hh[0], bias_hh[0])
        r = torch.sigmoid(gi[:, :H] + gh[:, :H])
        z = torch.sigmoid(gi[:, H:2*H] + gh[:, H:2*H])
        n = torch.tanh(gi[:, 2*H:] + r * gh[:, 2*H:])
        h[0] = (1 - z) * n + z * h[0]
        fwd_outputs.append(h[0].clone())

    if not bidir:
        y = torch.stack(fwd_outputs, dim=1).unsqueeze(0)
        return y, h

    bwd_outputs = [None] * T
    for t in range(T - 1, -1, -1):
        gi = torch.nn.functional.linear(x[:, t, :], weight_ih[1], bias_ih[1])
        gh = torch.nn.functional.linear(h[1], weight_hh[1], bias_hh[1])
        r = torch.sigmoid(gi[:, :H] + gh[:, :H])
        z = torch.sigmoid(gi[:, H:2*H] + gh[:, H:2*H])
        n = torch.tanh(gi[:, 2*H:] + r * gh[:, 2*H:])
        h[1] = (1 - z) * n + z * h[1]
        bwd_outputs[t] = h[1].clone()

    y = torch.stack([
        torch.stack([fwd_outputs[t], bwd_outputs[t]], dim=0)
        for t in range(T)
    ], dim=2)
    return y, h
