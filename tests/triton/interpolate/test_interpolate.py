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

"""interpolate 的 CPU 回退与 NPU Cube 路径测试。"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from oil_gas_ops_prospect.interpolate import bilinear, interpolate, trilinear
from oil_gas_ops_prospect.interpolate import triton_interpolate as ti


def _npu_device():
    try:
        import torch_npu  # noqa: F401
    except ImportError:
        return None
    if hasattr(torch, "npu") and torch.npu.is_available():
        return torch.device("npu:0")
    return None


def _force_custom(monkeypatch, min_bytes: int = 0, chunk_bytes: int | None = None):
    monkeypatch.setenv("OIL_GAS_OPS_INTERPOLATE_IMPL", "triton")
    monkeypatch.setattr(ti, "_MIN_BYTES", min_bytes)
    monkeypatch.setattr(ti, "_REQUIRE_CUSTOM", True)
    if chunk_bytes is not None:
        monkeypatch.setattr(ti, "_CHUNK_BYTES", chunk_bytes)


@pytest.mark.parametrize("align_corners", [False, True])
@pytest.mark.parametrize(
    "shape,size",
    [
        ((2, 3, 8, 9), (16, 18)),
        ((1, 2, 5, 7), (13, 11)),
    ],
)
def test_bilinear_cpu_matches_torch(shape, size, align_corners):
    x = torch.randn(shape)
    got = bilinear(x, size=size, align_corners=align_corners)
    ref = F.interpolate(x, size=size, mode="bilinear", align_corners=align_corners)
    torch.testing.assert_close(got, ref)


@pytest.mark.parametrize("align_corners", [False, True])
@pytest.mark.parametrize(
    "shape,size",
    [
        ((1, 2, 4, 5, 6), (8, 10, 12)),
        ((2, 1, 3, 4, 5), (6, 8, 10)),
    ],
)
def test_trilinear_cpu_matches_torch(shape, size, align_corners):
    x = torch.randn(shape)
    got = trilinear(x, size=size, align_corners=align_corners)
    ref = F.interpolate(x, size=size, mode="trilinear", align_corners=align_corners)
    torch.testing.assert_close(got, ref)


def test_int_size_and_grad_cpu():
    x = torch.randn(1, 2, 4, 5, 6, requires_grad=True)
    got = interpolate(x, size=10, mode="trilinear")
    ref = F.interpolate(x, size=10, mode="trilinear")
    torch.testing.assert_close(got, ref)
    grad = torch.autograd.grad(got.square().sum(), x)[0]
    ref_grad = torch.autograd.grad(ref.square().sum(), x)[0]
    torch.testing.assert_close(grad, ref_grad)


def test_scale_factor_falls_back_to_torch():
    x = torch.randn(1, 2, 4, 5, 6)
    got = interpolate(x, scale_factor=2.0, mode="trilinear")
    ref = F.interpolate(x, scale_factor=2.0, mode="trilinear")
    torch.testing.assert_close(got, ref)


def test_invalid_inputs():
    with pytest.raises(ValueError):
        interpolate(torch.randn(1, 2, 3), size=(6,), mode="trilinear")
    with pytest.raises(ValueError):
        interpolate(torch.randn(1, 2, 4, 4, 4), size=(8, 8), mode="trilinear")


@pytest.mark.skipif(_npu_device() is None, reason="NPU required")
def test_npu_trilinear_matches_torch(monkeypatch):
    _force_custom(monkeypatch)
    device = _npu_device()
    x = torch.randn(1, 32, 33, 33, 33, device=device)
    got = trilinear(x, size=(128, 128, 128))
    ref = F.interpolate(x, size=(128, 128, 128), mode="trilinear", align_corners=False)
    torch.testing.assert_close(got, ref, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(_npu_device() is None, reason="NPU required")
def test_npu_trilinear_grad_matches_torch(monkeypatch):
    _force_custom(monkeypatch)
    device = _npu_device()
    x = torch.randn(1, 4, 33, 33, 33, device=device, requires_grad=True)
    weight = torch.randn(1, 4, 128, 128, 128, device=device)

    got = interpolate(x, size=(128, 128, 128), mode="trilinear")
    got_grad = torch.autograd.grad((got * weight).sum(), x)[0]

    x_ref = x.detach().clone().requires_grad_(True)
    ref = F.interpolate(x_ref, size=(128, 128, 128), mode="trilinear", align_corners=False)
    ref_grad = torch.autograd.grad((ref * weight).sum(), x_ref)[0]
    torch.testing.assert_close(got, ref, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(got_grad, ref_grad, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(_npu_device() is None, reason="NPU required")
def test_npu_chunking_matches_torch(monkeypatch):
    _force_custom(monkeypatch, chunk_bytes=1 << 20)
    device = _npu_device()
    x = torch.randn(1, 2, 33, 33, 33, device=device)
    got = trilinear(x, size=(128, 128, 128))
    ref = F.interpolate(x, size=(128, 128, 128), mode="trilinear", align_corners=False)
    torch.testing.assert_close(got, ref, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(_npu_device() is None, reason="NPU required")
def test_npu_bilinear_stays_native(monkeypatch):
    # 4D must never enter the 5D Cube path or poison its process-wide check.
    _force_custom(monkeypatch)
    monkeypatch.setattr(ti, "_KERNEL_CHECK_OK", None)
    device = _npu_device()
    x = torch.randn(1, 128, 65, 65, device=device)
    got = bilinear(x, size=(256, 256))
    ref = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False)
    torch.testing.assert_close(got, ref, rtol=1e-5, atol=1e-5)
    assert ti._KERNEL_CHECK_OK is None


@pytest.mark.skipif(_npu_device() is None, reason="NPU required")
def test_npu_small_output_falls_back(monkeypatch):
    # Default min-bytes keeps the native kernel for the 17^3 -> 64^3 / C=32 case.
    monkeypatch.setenv("OIL_GAS_OPS_INTERPOLATE_IMPL", "auto")
    device = _npu_device()
    x = torch.randn(1, 32, 17, 17, 17, device=device)
    got = trilinear(x, size=(64, 64, 64))
    ref = F.interpolate(x, size=(64, 64, 64), mode="trilinear", align_corners=False)
    torch.testing.assert_close(got, ref, rtol=1e-5, atol=1e-5)
