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

"""complex_index_select 前向 / 反向精度，对拍 CPU 原生 torch.index_select。

分层：

1. **CPU 用例**（`TestCPU*`）：任何环境都跑，覆盖核心数学。算子本身是纯
   PyTorch 组合，CPU 上完全可用，因此这部分也是 `bash build.sh -u --composite`
   的实际覆盖来源。
2. **NPU 用例**（`TestNPU*`）：无 NPU 时自动 skip。除精度外还断言「NPU 原生
   `torch.index_select` 的反向仍不支持复数」，作为上游补齐后可移除本算子的
   信号（原生前向可用但派发到 AiCPU，慢路径不构成移除信号）。

参考实现一律取 CPU 原生 `torch.index_select`，不用本算子的展开式自证。
"""

from __future__ import annotations

import pytest
import torch
from oil_gas_ops_prospect.complex_index_select import complex_index_select

HAS_NPU = hasattr(torch, "npu") and torch.npu.is_available()
npu_only = pytest.mark.skipif(not HAS_NPU, reason="需要 NPU")

_DTYPES = [torch.complex64, torch.complex128]

# (shape, dim, index 长度) —— 覆盖 flat（连续且非末轴）与 real/imag（末轴 /
# 1-D）两条实现路径；负数 dim 与 int32 索引在专项用例中覆盖。
_CASES = [
    ((7, 5), 0, 4),          # flat：首轴
    ((3, 50, 5), 1, 31),     # flat：中间轴
    ((2, 3, 4, 5), 2, 2),    # flat：4-D
    ((77, 33), -1, 12),      # real/imag：末轴（负数 dim）
    ((12, 100, 5), 2, 3),    # real/imag：3-D 末轴
    ((100,), 0, 37),         # real/imag：1-D（dim=0 即末轴）
]


def _real_dtype(dtype):
    return torch.float32 if dtype == torch.complex64 else torch.float64


def _make(shape, dtype, device, requires_grad=False):
    rd = _real_dtype(dtype)
    t = torch.complex(torch.randn(shape, dtype=rd), torch.randn(shape, dtype=rd))
    t = t.to(device)
    return t.requires_grad_(requires_grad) if requires_grad else t


def _index_for(shape, dim, m, device, dtype=torch.int64, dup=False):
    """生成长度 m 的索引；默认唯一（randperm），dup=True 时允许重复。"""
    if dim < 0:
        dim += len(shape)
    n = shape[dim]
    if dup or m > n:
        g = torch.randint(0, n, (m,), dtype=torch.int64)
    else:
        g = torch.randperm(n)[:m]
    return g.to(dtype).to(device)


def _tolerance(dtype):
    # 与 verify_complex_index_select.py / 文档声明一致：反向 complex64 ≤ 1e-5、
    # complex128 ≤ 2e-14（910B2 实测含重复索引约 2.5e-7，阈值留足余量）。
    # 前向是纯 gather（无算术），逐位一致，用 atol=0 断言。
    return 1e-5 if dtype == torch.complex64 else 2e-14


def _forward_case(device, dtype, shape, dim, m, **index_kwargs):
    torch.manual_seed(0)
    z = _make(shape, dtype, device)
    index = _index_for(shape, dim, m, device, **index_kwargs)
    expected = torch.index_select(z.cpu(), dim, index.cpu())
    actual = complex_index_select(z, dim, index)
    if device == "npu":
        torch.npu.synchronize()
    return actual.cpu(), expected, dtype


def _backward_case(device, dtype, shape, dim, m, dup=False):
    torch.manual_seed(1)
    z = _make(shape, dtype, device)
    index = _index_for(shape, dim, m, device, dup=dup)
    d = dim if dim >= 0 else dim + len(shape)
    out_shape = list(shape)
    out_shape[d] = m
    seed = _make(tuple(out_shape), dtype, device)

    cpu_z = z.cpu().clone().requires_grad_(True)
    cpu_out = torch.index_select(cpu_z, dim, index.cpu())
    cpu_g = torch.autograd.grad(cpu_out, cpu_z, seed.cpu())[0]

    dev_z = z.clone().requires_grad_(True)
    dev_out = complex_index_select(dev_z, dim, index)
    dev_g = torch.autograd.grad(dev_out, dev_z, seed)[0]
    if device == "npu":
        torch.npu.synchronize()
    return dev_g.cpu(), cpu_g, dtype


class TestCPUForward:
    @pytest.mark.parametrize("dtype", _DTYPES)
    @pytest.mark.parametrize("case", _CASES)
    def test_matches_native(self, dtype, case):
        """纯 gather 无算术，与原生逐位一致（atol=0）。"""
        shape, dim, m = case
        actual, expected, dtype = _forward_case("cpu", dtype, shape, dim, m)
        assert actual.dtype == dtype
        assert actual.shape == expected.shape
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_int32_index(self):
        """int32 索引与 int64 结果一致（NPU 上原生只接受 int32/int64）。"""
        torch.manual_seed(2)
        z = _make((100, 7), torch.complex64, "cpu")
        i64 = _index_for((100, 7), 0, 33, "cpu", dtype=torch.int64)
        i32 = i64.to(torch.int32)
        got = complex_index_select(z, 0, i32)
        ref = torch.index_select(z, 0, i64)
        torch.testing.assert_close(got, ref, rtol=0, atol=0)

    def test_negative_dim_equivalent(self):
        """dim=-k 与 dim=rank-k 语义一致，且与原生对齐。"""
        torch.manual_seed(3)
        z = _make((3, 50, 5), torch.complex64, "cpu")
        idx = _index_for((3, 50, 5), 1, 20, "cpu")
        got_neg = complex_index_select(z, -2, idx)
        got_pos = complex_index_select(z, 1, idx)
        ref = torch.index_select(z, 1, idx)
        torch.testing.assert_close(got_neg, ref, rtol=0, atol=0)
        torch.testing.assert_close(got_pos, ref, rtol=0, atol=0)

    @pytest.mark.parametrize(
        ("tag", "maker", "dims"),
        [
            ("transpose", lambda z: z.t(), (0, 1)),                  # (7,100)
            ("slice", lambda z: z[:, ::2], (0, 1)),                  # (100,4)
            ("expand", lambda z: z.expand(4, 100, 7), (0, 1, 2)),    # stride 0
            ("row_stride", lambda z: z.t().contiguous().t(), (0, 1)),  # (100,7) 行主序
        ],
    )
    def test_non_contiguous_input(self, tag, maker, dims):
        """非连续输入走 real/imag 路径，仍与原生逐位一致。"""
        torch.manual_seed(4)
        z0 = _make((100, 7), torch.complex64, "cpu")
        z = maker(z0)
        for dim in dims:
            n = z.shape[dim]
            idx = torch.randperm(n)[: min(5, n)]
            got = complex_index_select(z, dim, idx)
            ref = torch.index_select(z, dim, idx)
            torch.testing.assert_close(got, ref, rtol=0, atol=0)

    def test_conj_input(self):
        """带 conj 位的输入（惰性共轭）：先 gather 后共轭 = 先共轭后 gather。"""
        torch.manual_seed(5)
        base = _make((50, 7), torch.complex64, "cpu")
        z = base.conj()
        idx = _index_for((50, 7), 0, 11, "cpu")
        got = complex_index_select(z, 0, idx)
        ref = torch.index_select(z, 0, idx)
        torch.testing.assert_close(got, ref, rtol=0, atol=0)
        torch.testing.assert_close(
            got, complex_index_select(base, 0, idx).conj(), rtol=0, atol=0
        )

    def test_empty_index(self):
        """空索引：输出沿 dim 长度为 0，与原生一致。"""
        torch.manual_seed(6)
        z = _make((7, 5), torch.complex64, "cpu")
        idx = torch.empty(0, dtype=torch.int64)
        got = complex_index_select(z, 0, idx)
        assert got.shape == (0, 5)
        ref = torch.index_select(z, 0, idx)
        torch.testing.assert_close(got, ref, rtol=0, atol=0)

    def test_empty_source_with_empty_index(self):
        """源张量沿 dim 为空时，空索引仍可 gather（batch=0 退化输入）。"""
        torch.manual_seed(7)
        z = _make((0, 5), torch.complex64, "cpu")
        idx = torch.empty(0, dtype=torch.int64)
        got = complex_index_select(z, 0, idx)
        assert got.shape == (0, 5)
        ref = torch.index_select(z, 0, idx)
        torch.testing.assert_close(got, ref, rtol=0, atol=0)


class TestCPUBackward:
    @pytest.mark.parametrize("dtype", _DTYPES)
    @pytest.mark.parametrize("case", _CASES)
    def test_matches_native(self, dtype, case):
        """唯一索引：散射零张量 + 按索引累加，与原生反向一致。"""
        shape, dim, m = case
        actual, expected, dtype = _backward_case("cpu", dtype, shape, dim, m)
        assert actual.dtype == dtype
        torch.testing.assert_close(actual, expected, rtol=0, atol=_tolerance(dtype))

    def test_duplicate_index_accumulates(self):
        """重复索引：同一行梯度应按出现次数累加，与原生一致。"""
        actual, expected, dtype = _backward_case(
            "cpu", torch.complex64, (7, 5), 0, 12, dup=True
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=_tolerance(dtype))

    def test_duplicate_index_exact_semantics(self):
        """索引 [0, 0]：第 0 行梯度恰为 grad_output 两行之和。"""
        torch.manual_seed(8)
        z = _make((4, 3), torch.complex64, "cpu", requires_grad=True)
        g = _make((2, 3), torch.complex64, "cpu")
        got = torch.autograd.grad(
            complex_index_select(z, 0, torch.tensor([0, 0])), z, g
        )[0]
        torch.testing.assert_close(got[0], g[0] + g[1], rtol=0, atol=0)
        torch.testing.assert_close(got[1:], torch.zeros_like(got[1:]), rtol=0, atol=0)

    @pytest.mark.parametrize("dtype", _DTYPES)
    @pytest.mark.parametrize(
        "case", [((6, 4), 0), ((4, 6), -1)], ids=["flat", "lastdim"]
    )
    def test_conj_bit_grad_output(self, dtype, case):
        """conj 位梯度（下游 .conj() 回流）在两条路径都应与原生一致。

        历史 bug：flat 反向 ``grad_output.contiguous().view(float)`` 对
        conjugate view 抛 RuntimeError（contiguous() 不消 conj 位），原生
        同场景正常；修复为反向先 ``resolve_conj()``。
        """
        shape, dim = case
        torch.manual_seed(11)
        z = _make(shape, dtype, "cpu", requires_grad=True)
        idx = _index_for(shape, dim, 3, "cpu")

        complex_index_select(z, dim, idx).conj().abs().sum().backward()

        ref = z.detach().clone().requires_grad_(True)
        torch.index_select(ref, dim, idx).conj().abs().sum().backward()
        assert z.grad is not None and z.grad.dtype == dtype
        torch.testing.assert_close(z.grad, ref.grad, rtol=0, atol=0)

    def test_gradient_accumulates_across_calls(self):
        """同一叶子被多次 gather 后梯度应累加。"""
        torch.manual_seed(9)
        z = _make((10, 3), torch.complex64, "cpu", requires_grad=True)
        i1 = torch.tensor([0, 1, 2])
        i2 = torch.tensor([1, 2, 3])
        out = complex_index_select(z, 0, i1) + complex_index_select(z, 0, i2)
        out.abs().sum().backward()

        ref = z.detach().clone().requires_grad_(True)
        ref_out = (
            torch.index_select(ref, 0, i1) + torch.index_select(ref, 0, i2)
        )
        ref_out.abs().sum().backward()
        torch.testing.assert_close(z.grad, ref.grad, rtol=0, atol=0)

    def test_empty_index_backward(self):
        """空索引反向应得到全零梯度。"""
        torch.manual_seed(10)
        z = _make((7, 5), torch.complex64, "cpu", requires_grad=True)
        g = _make((0, 5), torch.complex64, "cpu")
        got = torch.autograd.grad(
            complex_index_select(z, 0, torch.empty(0, dtype=torch.int64)), z, g
        )[0]
        torch.testing.assert_close(
            got, torch.zeros((7, 5), dtype=torch.complex64), rtol=0, atol=0
        )

    def test_second_order_matches_native(self):
        """二阶导与原生 index_select 行为一致：反向公式只消费 grad_output，
        不含可微保存张量，``create_graph`` 得到的梯度不带图（原生同样如此）。
        原生行为变化导致此用例失败时，应同步更新文档中的相应声明。"""
        torch.manual_seed(11)
        z = _make((20, 5), torch.complex64, "cpu", requires_grad=True)
        out = complex_index_select(z, 0, torch.arange(10))
        first = torch.autograd.grad(out.real.sum() + out.imag.sum(), z,
                                    create_graph=True)[0]
        assert first.requires_grad is False

        z_ref = z.detach().clone().requires_grad_(True)
        ref = torch.index_select(z_ref, 0, torch.arange(10))
        first_ref = torch.autograd.grad(
            ref.real.sum() + ref.imag.sum(), z_ref, create_graph=True
        )[0]
        assert first_ref.requires_grad == first.requires_grad


class TestValidation:
    @pytest.mark.parametrize(
        "bad_dtype", [torch.float32, torch.float64, torch.float16, torch.int64]
    )
    def test_rejects_real_dtype(self, bad_dtype):
        """实数 gather 在任何后端都原生可用，本算子不做转发。"""
        if bad_dtype == torch.int64:
            z = torch.randint(0, 10, (2, 3))
        else:
            z = torch.randn(2, 3, dtype=bad_dtype)
        idx = torch.tensor([0, 1])
        with pytest.raises(TypeError, match="只接受复数张量"):
            complex_index_select(z, 0, idx)

    def test_rejects_complex32(self):
        """complex32（实验性半精度复数）不支持：其 float 视图宽度换算与
        ``complex64`` / ``complex128`` 不同，反向会以无关的 shape 错误崩溃，
        显式拒绝优于误导性报错。"""
        if not hasattr(torch, "complex32"):
            pytest.skip("本 torch 无 complex32")
        z = torch.randn(2, 3, dtype=torch.complex32)
        idx = torch.tensor([0, 1])
        with pytest.raises(TypeError, match="complex64 / complex128"):
            complex_index_select(z, 0, idx)

    def test_rejects_non_tensor_index(self):
        z = _make((2, 3), torch.complex64, "cpu")
        with pytest.raises(TypeError, match="index 必须是张量"):
            complex_index_select(z, 0, [0, 1])

    def test_rejects_non_1d_index(self):
        z = _make((2, 3), torch.complex64, "cpu")
        with pytest.raises(ValueError, match="一维"):
            complex_index_select(z, 0, torch.tensor([[0, 1]]))

    @pytest.mark.parametrize(
        "bad_index_dtype", [torch.int16, torch.int8, torch.bool, torch.float32]
    )
    def test_rejects_bad_index_dtype(self, bad_index_dtype):
        z = _make((2, 3), torch.complex64, "cpu")
        idx = torch.tensor([0, 1]).to(bad_index_dtype)
        with pytest.raises(TypeError, match="int32 或 int64"):
            complex_index_select(z, 0, idx)

    @pytest.mark.parametrize("bad_dim", [2, -3, 100])
    def test_rejects_dim_out_of_range(self, bad_dim):
        z = _make((2, 3), torch.complex64, "cpu")
        idx = torch.tensor([0, 1])
        with pytest.raises(IndexError, match="Dimension out of range"):
            complex_index_select(z, bad_dim, idx)

    @pytest.mark.parametrize("bad_dim", [1.0, "0", None, True])
    def test_rejects_non_int_dim(self, bad_dim):
        z = _make((2, 3), torch.complex64, "cpu")
        idx = torch.tensor([0, 1])
        with pytest.raises(TypeError, match="dim 必须是 int"):
            complex_index_select(z, bad_dim, idx)

    def test_rejects_scalar_input(self):
        """0 维张量没有可 gather 的轴，与原生同样不可用（原生为
        RuntimeError；本算子统一为 IndexError）。"""
        z = _make((), torch.complex64, "cpu")
        idx = torch.tensor([0])
        with pytest.raises(IndexError, match="Dimension out of range"):
            complex_index_select(z, 0, idx)


@npu_only
class TestNPU:
    @pytest.mark.parametrize("dtype", _DTYPES)
    @pytest.mark.parametrize("case", _CASES)
    def test_forward_matches_native(self, dtype, case):
        shape, dim, m = case
        actual, expected, dtype = _forward_case("npu", dtype, shape, dim, m)
        assert actual.dtype == dtype
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_int32_index_npu(self):
        torch.manual_seed(12)
        z = _make((100, 7), torch.complex64, "npu")
        i64 = _index_for((100, 7), 0, 33, "npu", dtype=torch.int64)
        i32 = i64.to(torch.int32)
        got = complex_index_select(z, 0, i32)
        ref = torch.index_select(z.cpu(), 0, i64.cpu())
        torch.npu.synchronize()
        torch.testing.assert_close(got.cpu(), ref, rtol=0, atol=0)

    @pytest.mark.parametrize("dtype", _DTYPES)
    @pytest.mark.parametrize("case", _CASES)
    def test_backward_matches_native(self, dtype, case):
        """NPU 原生反向对复数报错，参考实现取 CPU 原生 autograd。"""
        shape, dim, m = case
        actual, expected, dtype = _backward_case("npu", dtype, shape, dim, m)
        assert actual.dtype == dtype
        torch.testing.assert_close(actual, expected, rtol=0, atol=_tolerance(dtype))

    def test_backward_duplicate_index_matches_native(self):
        actual, expected, dtype = _backward_case(
            "npu", torch.complex64, (7, 5), 0, 12, dup=True
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=_tolerance(dtype))

    @pytest.mark.parametrize(
        "case", [((6, 4), 0), ((4, 6), -1)], ids=["flat", "lastdim"]
    )
    def test_backward_conj_bit_grad_output(self, case):
        """conj 位梯度回流（下游 .conj()）：flat 反向曾因 conjugate view 的
        ``.view`` 抛错（原生正常），参考实现取 CPU 原生 autograd。"""
        shape, dim = case
        torch.manual_seed(12)
        z = _make(shape, torch.complex64, "npu")
        idx = _index_for(shape, dim, 3, "npu")

        dev_z = z.clone().requires_grad_(True)
        complex_index_select(dev_z, dim, idx).conj().abs().sum().backward()
        torch.npu.synchronize()
        dev_g = dev_z.grad.cpu()

        cpu_z = z.cpu().clone().requires_grad_(True)
        torch.index_select(cpu_z, dim, idx.cpu()).conj().abs().sum().backward()
        torch.testing.assert_close(
            dev_g, cpu_z.grad, rtol=0, atol=_tolerance(torch.complex64)
        )

    def test_native_backward_still_rejects_complex(self):
        """移除信号：上游 ``aclnnIndexAdd`` 补齐复数 dtype 后本算子可移除，
        届时此用例会失败。原生前向可用（AiCPU 慢路径），反向直接报 161002。"""
        z = _make((4, 3), torch.complex64, "npu", requires_grad=True)
        out = torch.index_select(z, 0, torch.tensor([0, 1], device="npu"))
        seed = _make((2, 3), torch.complex64, "npu")
        with pytest.raises(RuntimeError):
            torch.autograd.grad(out, z, seed)
