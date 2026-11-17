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

"""complex_addcmul 前向 / 反向精度，对拍 CPU 原生 torch.addcmul。

分层：

1. **CPU 用例**（`TestCPU*`）：任何环境都跑，覆盖核心数学。算子本身是纯
   PyTorch 组合，CPU 上完全可用，因此这部分也是 `bash build.sh -u --composite`
   的实际覆盖来源。
2. **NPU 用例**（`TestNPU*`）：无 NPU 时自动 skip。除精度外还断言「NPU 原生
   `torch.addcmul` 仍不支持复数」，作为上游补齐后可移除本算子的信号。

参考实现一律取 CPU 原生 `torch.addcmul`，不用本算子的展开式自证。
"""

from __future__ import annotations

import pytest
import torch
from oil_gas_ops_prospect.complex_addcmul import complex_addcmul

HAS_NPU = hasattr(torch, "npu") and torch.npu.is_available()
npu_only = pytest.mark.skipif(not HAS_NPU, reason="需要 NPU")

_DTYPES = [torch.complex64, torch.complex128]
_SHAPES = [
    ((2, 3), (2, 3), (2, 3)),
    ((2, 3), (1, 3), (2, 3)),
    ((2, 3), (2, 3), (2, 1)),
    ((2, 3), (1, 1), (1, 1)),
    ((1, 3), (2, 3), (1, 1)),
    ((), (2, 3), (2, 3)),
    ((3, 1, 5), (3, 4, 5), (1, 4, 1)),
    ((2, 4, 8, 8), (2, 4, 8, 8), (2, 4, 8, 8)),
]
_VALUES = [1.0, 2.5, 1 + 2j, -0.5 - 0.25j]


def _real_dtype(dtype):
    return torch.float32 if dtype == torch.complex64 else torch.float64


def _make(shape, dtype, device, requires_grad=False):
    rd = _real_dtype(dtype)
    t = torch.complex(torch.randn(shape, dtype=rd), torch.randn(shape, dtype=rd))
    t = t.to(device)
    return t.requires_grad_(requires_grad) if requires_grad else t


def _tolerance(dtype):
    # 与 verify_complex_addcmul.py / 文档声明一致：complex64 ≤ 1e-5、complex128 ≤ 2e-14。
    # 实测上界见 verify 脚本注释（误差量级约 eps × |out|）。
    return 1e-5 if dtype == torch.complex64 else 2e-14


def _forward_case(device, dtype, shapes, value):
    torch.manual_seed(0)
    s1, s2, s3 = shapes
    input, t1, t2 = (_make(s, dtype, device) for s in (s1, s2, s3))
    expected = torch.addcmul(input.cpu(), t1.cpu(), t2.cpu(), value=value)
    actual = complex_addcmul(input, t1, t2, value=value)
    if device == "npu":
        torch.npu.synchronize()
    return actual.cpu(), expected, dtype


def _backward_case(device, dtype, shapes, value):
    torch.manual_seed(1)
    s1, s2, s3 = shapes
    out_shape = torch.broadcast_shapes(s1, s2, s3)
    base = [_make(s, dtype, device) for s in (s1, s2, s3)]
    seed = _make(out_shape, dtype, device)

    cpu_in = [t.cpu().clone().requires_grad_(True) for t in base]
    cpu_out = torch.addcmul(*cpu_in, value=value)
    cpu_grads = torch.autograd.grad(cpu_out, cpu_in, seed.cpu())

    npu_in = [t.clone().requires_grad_(True) for t in base]
    npu_out = complex_addcmul(*npu_in, value=value)
    npu_grads = torch.autograd.grad(npu_out, npu_in, seed)
    if device == "npu":
        torch.npu.synchronize()
    return [g.cpu() for g in npu_grads], cpu_grads, dtype


class TestCPUForward:
    @pytest.mark.parametrize("dtype", _DTYPES)
    @pytest.mark.parametrize("shapes", _SHAPES)
    @pytest.mark.parametrize("value", _VALUES)
    def test_matches_native(self, dtype, shapes, value):
        actual, expected, dtype = _forward_case("cpu", dtype, shapes, value)
        assert actual.dtype == dtype
        assert actual.shape == expected.shape
        torch.testing.assert_close(
            actual, expected, rtol=0, atol=_tolerance(dtype)
        )

    @pytest.mark.parametrize("dtype", _DTYPES)
    def test_dtype_promotion(self, dtype):
        """混合 dtype 按 torch.promote_types 提升，与原生一致。"""
        other = torch.complex128 if dtype == torch.complex64 else torch.complex64
        torch.manual_seed(2)
        a = _make((2, 3), dtype, "cpu")
        b = _make((2, 3), dtype, "cpu")
        c = _make((2, 3), other, "cpu")
        expected = torch.addcmul(c, a, b)
        actual = complex_addcmul(c, a, b)
        assert actual.dtype == expected.dtype
        torch.testing.assert_close(
            actual, expected, rtol=0, atol=_tolerance(expected.dtype)
        )

    def test_scalar_tensor_value(self):
        """value 允许 0 维张量。"""
        torch.manual_seed(3)
        a, b, c = (_make((2, 3), torch.complex64, "cpu") for _ in range(3))
        v = torch.tensor(2.5)
        torch.testing.assert_close(
            complex_addcmul(c, a, b, value=v),
            torch.addcmul(c, a, b, value=2.5),
            rtol=0,
            atol=1e-6,
        )


class TestCPUBackward:
    @pytest.mark.parametrize("dtype", _DTYPES)
    @pytest.mark.parametrize("shapes", _SHAPES)
    @pytest.mark.parametrize("value", [1.0, 2.5, 1 + 2j])
    def test_matches_native(self, dtype, shapes, value):
        actual, expected, dtype = _backward_case("cpu", dtype, shapes, value)
        for got, want in zip(actual, expected):
            assert got.shape == want.shape
            assert got.dtype == want.dtype
            torch.testing.assert_close(
                got, want, rtol=0, atol=_tolerance(dtype)
            )

    def test_gradient_accumulates_across_calls(self):
        """同一叶子被多次使用后梯度应累加。"""
        torch.manual_seed(4)
        t = _make((2, 3), torch.complex64, "cpu", requires_grad=True)
        c = _make((2, 3), torch.complex64, "cpu")

        out = complex_addcmul(c, t, t) + complex_addcmul(c, t, t)
        out.abs().sum().backward()

        ref = t.detach().clone().requires_grad_(True)
        ref_out = torch.addcmul(c, ref, ref) + torch.addcmul(c, ref, ref)
        ref_out.abs().sum().backward()

        torch.testing.assert_close(t.grad, ref.grad, rtol=0, atol=1e-5)

    def test_empty_tensor_backward(self):
        """空张量前向与反向都要走通（batch=0 的退化输入）。"""
        torch.manual_seed(6)
        leaves = [_make((0, 3), torch.complex64, "cpu") for _ in range(3)]
        out = complex_addcmul(*leaves)
        assert out.shape == torch.broadcast_shapes((0, 3), (0, 3), (0, 3))
        ref_leaves = [t.detach().clone().requires_grad_(True) for t in leaves]
        ref = torch.addcmul(*ref_leaves)
        assert out.shape == ref.shape

        got_leaves = [t.detach().clone().requires_grad_(True) for t in leaves]
        gout = _make((0, 3), torch.complex64, "cpu")
        got_grads = torch.autograd.grad(complex_addcmul(*got_leaves), got_leaves, gout)
        ref_grads = torch.autograd.grad(ref, ref_leaves, gout)
        for g, w in zip(got_grads, ref_grads):
            assert g.shape == w.shape

    def test_second_order_gradient(self):
        """二阶反向：CPU 走 torch 表达式时可求；不支持的设备会抛异常。"""
        t = _make((2, 3), torch.complex64, "cpu", requires_grad=True)
        c = _make((2, 3), torch.complex64, "cpu")
        out = complex_addcmul(c, t, t)
        first = torch.autograd.grad(out.abs().sum(), t, create_graph=True)[0]
        second = torch.autograd.grad(first.abs().sum(), t, allow_unused=True)[0]
        assert second is not None and torch.isfinite(second).all()


class TestValidation:
    @pytest.mark.parametrize(
        "bad_dtype", [torch.float32, torch.float64, torch.float16]
    )
    def test_rejects_real_dtype(self, bad_dtype):
        args = [torch.randn(2, 3, dtype=bad_dtype) for _ in range(3)]
        with pytest.raises(TypeError, match="只接受复数张量"):
            complex_addcmul(*args)

    def test_rejects_non_scalar_value(self):
        args = [_make((2, 3), torch.complex64, "cpu") for _ in range(3)]
        with pytest.raises(TypeError, match="必须是标量"):
            complex_addcmul(*args, value=torch.randn(3))

    def test_rejects_non_numeric_value(self):
        args = [_make((2, 3), torch.complex64, "cpu") for _ in range(3)]
        with pytest.raises(TypeError, match="实数或复数标量"):
            complex_addcmul(*args, value="x")

    def test_rejects_value_requiring_grad(self):
        """value 不返回梯度；带 requires_grad 时显式报错而不是静默丢梯度。"""
        args = [_make((2, 3), torch.complex64, "cpu") for _ in range(3)]
        v = torch.tensor(2.0, requires_grad=True)
        with pytest.raises(TypeError, match="value 不支持求导"):
            complex_addcmul(*args, value=v)


@npu_only
class TestNPU:
    @pytest.mark.parametrize("dtype", _DTYPES)
    @pytest.mark.parametrize("shapes", _SHAPES)
    @pytest.mark.parametrize("value", [1.0, 1 + 2j])
    def test_forward_matches_native(self, dtype, shapes, value):
        actual, expected, dtype = _forward_case("npu", dtype, shapes, value)
        assert actual.dtype == dtype
        torch.testing.assert_close(
            actual, expected, rtol=0, atol=_tolerance(dtype)
        )

    @pytest.mark.parametrize("dtype", _DTYPES)
    @pytest.mark.parametrize("shapes", _SHAPES[:6])
    @pytest.mark.parametrize("value", [1.0, 1 + 2j])
    def test_backward_matches_native(self, dtype, shapes, value):
        actual, expected, dtype = _backward_case("npu", dtype, shapes, value)
        for got, want in zip(actual, expected):
            assert got.shape == want.shape
            torch.testing.assert_close(
                got, want, rtol=0, atol=_tolerance(dtype)
            )

    def test_native_addcmul_still_rejects_complex(self):
        """上游补齐复数 dtype 后本算子可移除；届时此用例会失败。"""
        args = [_make((2, 3), torch.complex64, "npu") for _ in range(3)]
        with pytest.raises(RuntimeError):
            torch.addcmul(*args)
