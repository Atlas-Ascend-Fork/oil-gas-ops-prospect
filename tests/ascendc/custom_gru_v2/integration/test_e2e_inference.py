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

"""Integration tests: CustomGRU v2 end-to-end inference.

Validates inference shapes, bidirectional, and nn.Module integration.

设备标记整改（按用例标记设备）：本文件历史版本创建的是默认 CPU 张量，
导致所有用例静默走 ``_torch_gru_forward`` CPU fallback，从未真正命中
NPU kernel（"假 NPU"，见 REMOTE_TEST_RESULTS §10.4）。现整改为：

- 每条用例带 ``@npu_only()`` 标记（conftest.npu_only，无 NPU 时 SKIP）；
  模型/输入显式 ``to(device)``（NPU）。
- 输出断言 ``y.device.type == "npu"``，杜绝静默 fallback。
- 数值断言（与 PyTorch GRU 参考对比）暂以 ``xfail(strict=False, reason=...)``
  登记 —— kernel 前向在 t>=1 与 PyTorch GRU 语义不一致（缺陷 D2，评估见
  远端报告；kernel 内部 carry 的数值修复属于 op_kernel 层，另行处理）。
  kernel 数值问题修复后，这些 xfail 将转为通过，届时删除标记。

Fixes from review:
- P0-1: multi-layer tests removed (num_layers not in API)
- P0-6: bidirectional test now validates values against correct reference
- P2-2: IT-INFER-004 (graph capture) removed — requires NPU + complex setup, marked separately
- P1-3: bidirectional uses independent forward/reverse weights
"""
from __future__ import annotations

import pytest
import torch

from conftest import npu_only, npu_hardware_only, get_device, torch_gru_ref as _torch_gru_ref

# 缺陷 D2：NPU kernel 前向数值与 PyTorch GRU 语义不一致（t>=1 漂移）。
# 根因：op_kernel 标量段与矢量段跨流水线缺少同步（评审 H1）。已修复：
# custom_gruv2.cpp 补齐 SetFlag/WaitFlag<HardEvent> 事件同步，因此本文件
# 恢复严格数值门禁（不再使用 xfail，见 test_npu_contract.py 说明）。


@npu_only()
def test_it_infer_001_single_sequence():
    """单序列推理（NPU，T=1000 长序列）。"""
    from oil_gas_ops_prospect.custom_gru_v2 import custom_gru_v2
    H, F_in = 24, 3
    device = get_device()
    torch.manual_seed(1234)
    x = torch.randn(1, 1000, F_in, device=device)
    wi = torch.randn(3 * H, F_in, device=device) * 0.1
    wh = torch.randn(3 * H, H, device=device) * 0.1
    bi = torch.randn(3 * H, device=device) * 0.1
    bh = torch.randn(3 * H, device=device) * 0.1
    y, h = custom_gru_v2(x, wi, wh, bi, bh, H, False)
    assert y.shape == (1, 1, 1000, H)
    assert h.shape == (1, 1, H)
    assert not torch.isnan(y).any()
    assert y.device.type == device.split(":")[0], \
        f"输出落在 {y.device}，期望 {device}（防止静默 CPU fallback）"


@npu_only()
def test_it_infer_002_batch_inference():
    """批量推理（NPU，B=32 T=100）。"""
    from oil_gas_ops_prospect.custom_gru_v2 import custom_gru_v2
    H, F_in = 24, 3
    device = get_device()
    torch.manual_seed(1234)
    x = torch.randn(32, 100, F_in, device=device)
    wi = torch.randn(3 * H, F_in, device=device) * 0.1
    wh = torch.randn(3 * H, H, device=device) * 0.1
    bi = torch.randn(3 * H, device=device) * 0.1
    bh = torch.randn(3 * H, device=device) * 0.1
    y, h = custom_gru_v2(x, wi, wh, bi, bh, H, False)
    assert y.shape == (1, 32, 100, H)
    assert h.shape == (1, 32, H)
    assert y.device.type == device.split(":")[0]


@npu_only()
def test_it_infer_003_bidirectional_values():
    """双向 + 数值验证（NPU）：扁平 [6H] 权重（前半 = 前向，后半 = 反向）。"""
    from oil_gas_ops_prospect.custom_gru_v2 import custom_gru_v2
    H, F_in = 8, 3
    torch.manual_seed(42)
    x_cpu = torch.randn(1, 50, F_in)
    wi3 = torch.randn(2, 3 * H, F_in) * 0.1
    wh3 = torch.randn(2, 3 * H, H) * 0.1
    bi3 = torch.randn(2, 3 * H) * 0.1
    bh3 = torch.randn(2, 3 * H) * 0.1
    y_ref, _h_ref = _torch_gru_ref(x_cpu, wi3, wh3, bi3, bh3, H, bidir=True)

    # 3D 权重可整组传给 wrapper（内部按 kernel 契约扁平化）；这里保持
    # 历史写法 reshape(-1) 与本用例原实现一致，两者等价。
    y, h = custom_gru_v2(x_cpu.npu(), wi3.reshape(-1, F_in).npu(), wh3.reshape(-1, H).npu(),
                          bi3.reshape(-1).npu(), bh3.reshape(-1).npu(), H, True)
    assert y.shape == (2, 1, 50, H)
    assert h.shape == (2, 1, H)
    assert y.device.type == "npu"
    max_diff_0 = (y.cpu()[0] - y_ref[0]).abs().max().item()
    max_diff_1 = (y.cpu()[1] - y_ref[1]).abs().max().item()
    assert max_diff_0 <= 0.01, f"Bidirectional y[0] max_diff={max_diff_0} > 0.01"
    assert max_diff_1 <= 0.01, f"Bidirectional y[1] max_diff={max_diff_1} > 0.01"


@npu_only()
def test_it_infer_004_module_integration():
    """nn.Module 单向集成（NPU）。"""
    from oil_gas_ops_prospect.custom_gru_v2 import CustomGRU
    device = get_device()
    torch.manual_seed(1234)
    model = CustomGRU(3, 24, bidirectional=False).to(device)
    x = torch.randn(1, 100, 3, device=device)
    y, h = model(x)
    assert y.shape == (1, 1, 100, 24)
    assert h.shape == (1, 1, 24)
    assert not torch.isnan(y).any()
    assert y.device.type == device.split(":")[0]


@npu_only()
def test_it_infer_005_module_bidirectional():
    """nn.Module 双向集成（NPU）。

    D1（contract）修复后：模块将单份 [3H] 权重按"同权重复用"语义复制成
    [6H]，NPU kernel 不再因 host tiling 形状校验拒绝（原
    Maximum GetWorkspaceSize failed: 561002）。数值对比见 003（D2 xfail）。
    """
    from oil_gas_ops_prospect.custom_gru_v2 import CustomGRU
    device = get_device()
    torch.manual_seed(1234)
    model = CustomGRU(3, 16, bidirectional=True).to(device)
    x = torch.randn(4, 50, 3, device=device)
    y, h = model(x)
    assert y.shape == (2, 4, 50, 16)
    assert h.shape == (2, 4, 16)
    assert not torch.isnan(y).any()
    assert y.device.type == device.split(":")[0]


@npu_hardware_only()
def test_it_infer_006_npu_graph_capture():
    """NPU graph capture test — only runs on NPU hardware with compiled extension.

    Note: make_graphed_callables requires the backward to produce gradients
    with shapes matching the forward outputs. CustomGRU's backward uses a
    Python-based _torch_gru_forward recompute, which creates a different
    computation graph shape. This is a known limitation of autograd.Function
    with Python backward + graph capture. Skip if not supported.
    """
    try:
        from oil_gas_ops_prospect.custom_gru_v2 import _load_extension
        _load_extension()
    except (RuntimeError, ImportError):
        pytest.skip("custom_gru_v2_npu extension not compiled — skipping graph capture test")
    from oil_gas_ops_prospect.custom_gru_v2 import CustomGRU
    device = get_device()
    model = CustomGRU(3, 24, bidirectional=False).to(device)
    x = torch.randn(1, 100, 3, device=device)
    model(x)
    try:
        graphed = torch.npu.make_graphed_callables(model, x)
    except (RuntimeError, ValueError) as e:
        pytest.skip(f"Graph capture not supported with Python backward: {e}")
    y1, h1 = graphed(x)
    y2, h2 = graphed(x)
    assert torch.equal(y1, y2), "Graph replay produced inconsistent results"
