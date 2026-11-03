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

"""complex_mul pybind 封装的输出别名契约。

kernel 把 2 路（反向 4 路）输出写进同一块缓冲，返回的 view 共享 storage，
由此引出两条契约：

1. **前向**返回值必须彼此不共享 storage，否则 ``autograd.Function`` 返回多 view
   时 PyTorch 拒绝其原地写。
2. **反向**不得为拆别名额外 clone 梯度输入。``contiguous()`` 已能物化 stride 0
   的广播视图，额外的 ``clone()`` 在 NPU 上是多余的整张量拷贝。
"""

from __future__ import annotations

import unittest
from unittest import mock

import torch
from oil_gas_ops_prospect import complex_mul as cm


def _packed(numel: int, count: int) -> list[torch.Tensor]:
    """模拟 C++ 侧 ``at::empty({count, numel})`` + ``select(0, i)`` 的输出。"""
    buf = torch.empty(count, numel)
    return [buf.select(0, i) for i in range(count)]


class _FakeForwardExt:
    """记录 forward 收到的入参，并返回共享 storage 的 packed 输出。"""

    def __init__(self) -> None:
        self.received: list[torch.Tensor] = []

    def complex_mul_forward(self, ur, ui, vr, vi):
        self.received = [ur, ui, vr, vi]
        return _packed(ur.numel(), 2)


class _FakeBackwardExt:
    """记录 backward 收到的入参，并返回共享 storage 的 4 路梯度。"""

    def __init__(self) -> None:
        self.received: list[torch.Tensor] = []

    def complex_mul_backward(self, gwr, gwi, ur, ui, vr, vi):
        self.received = [gwr, gwi, ur, ui, vr, vi]
        return _packed(ur.numel(), 4)


class SplitOutputsTests(unittest.TestCase):
    def test_forward_outputs_do_not_share_storage(self):
        ext = _FakeForwardExt()
        with mock.patch.object(cm, "_load_extension", return_value=ext):
            wr, wi = cm._ascendc_forward(*(torch.randn(16) for _ in range(4)))

        self.assertNotEqual(
            wr.untyped_storage().data_ptr(),
            wi.untyped_storage().data_ptr(),
            "前向两路输出共享 storage 会让下游原地写触发 "
            "'output of a function that returns multiple views'",
        )

    def test_forward_outputs_are_independent_after_inplace_write(self):
        ext = _FakeForwardExt()
        with mock.patch.object(cm, "_load_extension", return_value=ext):
            wr, wi = cm._ascendc_forward(*(torch.randn(16) for _ in range(4)))

        before = wi.clone()
        wr.add_(1.0)
        self.assertTrue(torch.equal(wi, before), "写 wr 不应污染 wi")

    def test_backward_outputs_are_not_cloned(self):
        """反向返回 4 路共享 storage 的 view 是允许的，不应额外 clone。"""
        ext = _FakeBackwardExt()
        with mock.patch.object(cm, "_load_extension", return_value=ext):
            grads = cm._ascendc_backward(*(torch.randn(16) for _ in range(6)))

        storage_ptrs = {g.untyped_storage().data_ptr() for g in grads}
        self.assertEqual(
            len(storage_ptrs),
            1,
            "反向不应 clone 输出；多块 storage 说明又加回了整张量拷贝",
        )


class KernelInputTests(unittest.TestCase):
    def test_contiguous_tensor_is_passed_through_without_copy(self):
        t = torch.randn(64)
        out = cm._kernel_in(t)
        self.assertTrue(out.is_contiguous())
        self.assertEqual(
            out.data_ptr(),
            t.data_ptr(),
            "已连续的张量不应被复制；data_ptr 变化说明多了一次整张量拷贝",
        )

    def test_non_contiguous_tensor_is_materialized(self):
        base = torch.randn(8, 8)
        view = base.t()  # 非连续
        self.assertFalse(view.is_contiguous())
        out = cm._kernel_in(view)
        self.assertTrue(out.is_contiguous())
        self.assertEqual(out.data_ptr() != view.data_ptr(), True)

    def test_stride_zero_broadcast_view_is_materialized(self):
        """``(wr + wi).sum()`` 时 autograd 给的是 stride 0 的广播 grad。"""
        scalar = torch.randn(())
        view = scalar.expand(4, 8)
        self.assertEqual(view.stride(), (0, 0))
        self.assertFalse(view.is_contiguous())

        out = cm._kernel_in(view)
        self.assertTrue(out.is_contiguous())
        self.assertEqual(out.data_ptr() != view.data_ptr(), True)
        self.assertTrue(torch.equal(out, view))

    def test_alias_inputs_are_separated_by_contiguous(self):
        """两路 grad 互为同一对象的 view 时，物化后必须指向不同缓冲。"""
        scalar = torch.randn(())
        view = scalar.expand(4, 8)
        a, b = cm._kernel_in(view), cm._kernel_in(view)
        self.assertNotEqual(a.data_ptr(), b.data_ptr())


if __name__ == "__main__":
    unittest.main()
