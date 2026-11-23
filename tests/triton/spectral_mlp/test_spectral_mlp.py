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

"""spectral_mlp 的 CPU 回退与 NPU kernel 前/反向精度测试。"""

from __future__ import annotations

import pytest
import torch

import _bootstrap  # noqa: F401  (side effect: 源码树里解析 oil_gas_ops_prospect)
from oil_gas_ops_prospect.spectral_mlp import spectral_mlp

GELU_C0 = 1.5957691
GELU_C1 = 0.044715


def _npu_device():
    try:
        import torch_npu  # noqa: F401
    except ImportError:
        return None
    if hasattr(torch, "npu") and torch.npu.is_available():
        return torch.device("npu:0")
    return None


def _gelu(x: torch.Tensor) -> torch.Tensor:
    return x / (1.0 + torch.exp(-GELU_C0 * (x + GELU_C1 * x * x * x)))


def _reference(xr, xi, up1, down1, gate1r, gate1i, bias1r, bias1i,
               up2, down2, gate2r, gate2i, bias2r, bias2i):
    """独立的 einsum 参考实现（不调用算子内部函数）。"""
    B, L = int(xr.shape[0]), int(xr.shape[1])
    spatial = tuple(int(v) for v in xr.shape[2:-2])
    G, D = int(xr.shape[-2]), int(xr.shape[-1])
    S = 1
    for v in spatial:
        S *= v
    R = int(up1.shape[2])
    DM = int(down1.shape[1])

    xr2 = xr.reshape(B, L, S, G, D)
    xi2 = xi.reshape(B, L, S, G, D)
    x1r = torch.einsum("blsgf,gfr->blsgr", xr2, up1)
    x1i = torch.einsum("blsgf,gfr->blsgr", xi2, up1)
    g1r = gate1r.view(1, L, 1, G, R)
    g1i = gate1i.view(1, L, 1, G, R)
    sum1 = x1r * g1r + x1i * g1i
    diff1 = x1r * g1r - x1i * g1i
    h1r = torch.einsum("blsgr,gdr->blsgd", diff1, down1) + bias1r.view(1, 1, 1, G, DM)
    h1i = torch.einsum("blsgr,gdr->blsgd", sum1, down1) + bias1i.view(1, 1, 1, G, DM)
    o1r = _gelu(h1r)
    o1i = _gelu(h1i)

    x2r = torch.einsum("blsgf,gfr->blsgr", o1r, up2)
    x2i = torch.einsum("blsgf,gfr->blsgr", o1i, up2)
    g2r = gate2r.view(1, L, 1, G, R)
    g2i = gate2i.view(1, L, 1, G, R)
    sum2 = x2r * g2r + x2i * g2i
    diff2 = x2r * g2r - x2i * g2i
    yr = torch.einsum("blsgr,gdr->blsgd", diff2, down2) + bias2r.view(1, 1, 1, G, D)
    yi = torch.einsum("blsgr,gdr->blsgd", sum2, down2) + bias2i.view(1, 1, 1, G, D)
    out_shape = (B, L, *spatial, G, D)
    return yr.reshape(out_shape), yi.reshape(out_shape)


def _make_inputs(B, L, spatial, G, D, DM, R, *, device="cpu", dtype=torch.float32, seed=0):
    gen = torch.Generator(device="cpu").manual_seed(seed)

    def f(*shape):
        return torch.randn(*shape, generator=gen).to(device=device, dtype=dtype) * 0.1
    xr = f(B, L, *spatial, G, D)
    xi = f(B, L, *spatial, G, D)
    return (
        xr, xi,
        f(G, D, R), f(G, DM, R),
        f(L, G, R), f(L, G, R), f(G, DM), f(G, DM),
        f(G, DM, R), f(G, D, R),
        f(L, G, R), f(L, G, R), f(G, D), f(G, D),
    )


SHAPES = [
    (1, 4, (4, 4), 2, 16, 16, 8),
    (2, 3, (2, 3), 4, 8, 8, 8),
    (1, 2, (1, 1), 1, 16, 16, 16),
    (1, 3, (2, 3), 2, 12, 20, 6),  # d/dm/rank 非 2 的幂，覆盖内核 padding
]


@pytest.mark.parametrize("shape", SHAPES)
def test_cpu_forward_matches_reference(shape):
    B, L, spatial, G, D, DM, R = shape
    args = _make_inputs(B, L, spatial, G, D, DM, R)
    got = spectral_mlp(*args)
    ref = _reference(*args)
    torch.testing.assert_close(got[0], ref[0], rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(got[1], ref[1], rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("shape", SHAPES)
def test_cpu_backward_matches_reference(shape):
    B, L, spatial, G, D, DM, R = shape
    base = _make_inputs(B, L, spatial, G, D, DM, R)
    leaves = [t.clone().requires_grad_(True) for t in base]
    out = spectral_mlp(*leaves)
    (out[0].sum() + 2.0 * out[1].sum()).backward()
    got = [t.grad for t in leaves]

    ref_leaves = [t.clone().requires_grad_(True) for t in base]
    ref = _reference(*ref_leaves)
    (ref[0].sum() + 2.0 * ref[1].sum()).backward()
    ref_grads = [t.grad for t in ref_leaves]

    for g, rg in zip(got, ref_grads):
        torch.testing.assert_close(g, rg, rtol=1e-6, atol=1e-6)


def test_cpu_non_contiguous_inputs():
    args = list(_make_inputs(1, 4, (4, 4), 2, 16, 16, 8))
    # 空间轴换序制造非连续输入
    args[0] = args[0].transpose(2, 3)
    args[1] = args[1].transpose(2, 3)
    got = spectral_mlp(*args)
    ref = _reference(*[t.contiguous() for t in args])
    torch.testing.assert_close(got[0], ref[0], rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(got[1], ref[1], rtol=1e-6, atol=1e-6)


def test_float64_falls_back_to_reference():
    args = _make_inputs(1, 2, (2, 2), 2, 8, 8, 8, dtype=torch.float64)
    got = spectral_mlp(*args)
    ref = _reference(*args)
    assert got[0].dtype == torch.float64
    torch.testing.assert_close(got[0], ref[0], rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(got[1], ref[1], rtol=1e-12, atol=1e-12)


def test_invalid_shapes_raise():
    args = list(_make_inputs(1, 2, (2, 2), 2, 16, 16, 8))

    bad = list(args)
    bad[1] = torch.randn(2, 2, 2, 2, 2, 16)  # xi 与 xr 不匹配
    with pytest.raises(ValueError):
        spectral_mlp(*bad)

    bad = list(args)
    bad[4] = torch.randn(2, 16, 4)  # down1 的 rank 与 up1 不一致
    with pytest.raises(ValueError):
        spectral_mlp(*bad)

    bad = list(args)
    bad[6] = torch.randn(1, 2, 4)  # gate1_r 应是 (L, G, rank)
    with pytest.raises(ValueError):
        spectral_mlp(*bad)


def test_gelu_helper_matches_reference_formula():
    from oil_gas_ops_prospect.spectral_mlp.triton_spectral_mlp import gelu_approx

    x = torch.linspace(-6.0, 6.0, 101)
    torch.testing.assert_close(gelu_approx(x), _gelu(x), rtol=0, atol=0)


@pytest.mark.skipif(_npu_device() is None, reason="NPU required")
@pytest.mark.parametrize("shape", [SHAPES[0], SHAPES[1], SHAPES[3]])
def test_npu_forward_backward_matches_reference(shape):
    device = _npu_device()
    B, L, spatial, G, D, DM, R = shape
    base = _make_inputs(B, L, spatial, G, D, DM, R, device=device)
    got = spectral_mlp(*base)
    ref = _reference(*base)
    torch.testing.assert_close(got[0], ref[0], rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(got[1], ref[1], rtol=1e-4, atol=1e-4)

    leaves = [t.clone().requires_grad_(True) for t in base]
    out = spectral_mlp(*leaves)
    (out[0].sum() + out[1].sum()).backward()
    got_grads = [t.grad for t in leaves]

    ref_leaves = [t.clone().requires_grad_(True) for t in base]
    ref_out = _reference(*ref_leaves)
    (ref_out[0].sum() + ref_out[1].sum()).backward()
    for g, rg in zip(got_grads, [t.grad for t in ref_leaves]):
        torch.testing.assert_close(g, rg, rtol=2e-3, atol=2e-3)


@pytest.mark.skipif(_npu_device() is None, reason="NPU required")
def test_npu_inference_path_skips_saved_intermediates():
    device = _npu_device()
    args = _make_inputs(1, 4, (4, 4), 2, 16, 16, 8, device=device)
    with torch.no_grad():
        got = spectral_mlp(*args)
    ref = _reference(*args)
    torch.testing.assert_close(got[0], ref[0], rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(got[1], ref[1], rtol=1e-4, atol=1e-4)
