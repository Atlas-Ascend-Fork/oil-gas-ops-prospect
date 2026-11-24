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

"""NPU 契约级断言测试（fix/gru-bidir-module-npu 修复验证 + 真实 NPU 覆盖）。

对应远端验证报告中的缺陷/矩阵（REMOTE_TEST_RESULTS §10.1 / §10.2）与评审意见
（docs/reviews/GRU_BIDIR_MODULE_REVIEW.md）：

- D1（contract）：模块 `CustomGRU(bidirectional=True)` 持单份 [3H] 权重直接进
  kernel 被 host tiling 拒绝（``GetWorkspaceSize failed: 561002``）。修复：
  `_contract_normalize_weights` 将 [3H]+bidir 按 CPU fallback 一致的
  "同权重复用" 语义复制为 [6H]；3D 权重扁平化；单向形态不变。
- D2（numerics）：NPU kernel 前向在 t>=1 相对 PyTorch GRU 产生数值漂移。
  根因（评审 H1 处理项）：kernel 的标量段（gates/h 读取写回）与矢量段跨
  流水线缺少同步，`PipeBarrier<PIPE_V>` 不提供 S<->V、V->MTE3 的跨管线
  同步；op_kernel 已补齐 `SetFlag/WaitFlag<HardEvent>` 同步。本文件相应
  移除 xfail，改为严格门禁（按方向分方向对比 `y` 与 `output_h`，
  覆盖多组 T/B/H，阈值 0.01 / 0.01）。

设备标记（评审 M1）：每条 kernel 级用例 ``@npu_only()`` +
  ``@requires_compiled_kernel()``，模型/输入显式上 NPU，输出断言
  ``y.device.type == "npu"``——`npu_only` 只保证硬件在位，不会阻止 wrapper
  在扩展导入失败时静默走 PyTorch fallback；两者叠加才能证明"用到了自定义
  kernel"。CPU fallback 语义用例（评审 H2）在测试内部用 ``monkeypatch``
  临时移除 ``OIL_GAS_OPS_REQUIRE_ASCENDC_CUSTOM_GRU_V2``，避免与
  ``run_all_tests.sh`` 的全局强制模式冲突。
"""
from __future__ import annotations

import pytest
import torch

from conftest import (
    npu_only,
    get_device,
    requires_compiled_kernel,
    torch_gru_ref as _torch_gru_ref,
)

_SEED = 2026
_ATOL = 0.01


def _cpu_fallback_env_removed(monkeypatch):
    """评审 H2：即使 run_all_tests.sh 全局设置了强制 NPU 算子，本用例也要
    能合法地验证 CPU fallback 语义（临时移除强制变量，结束后自动恢复）。"""
    monkeypatch.delenv("OIL_GAS_OPS_REQUIRE_ASCENDC_CUSTOM_GRU_V2", raising=False)


@npu_only()
@requires_compiled_kernel()
def test_kernel_contract_lowlevel_flat6h_bidir():
    """底层 API 双向（kernel 契约扁平 [6H] 形态）。"""
    from oil_gas_ops_prospect.custom_gru_v2 import custom_gru_v2
    H, F_in = 8, 3
    device = get_device()
    torch.manual_seed(_SEED)
    x = torch.randn(2, 5, F_in, device=device)
    wi = torch.randn(6 * H, F_in, device=device) * 0.1
    wh = torch.randn(6 * H, H, device=device) * 0.1
    bi = torch.randn(6 * H, device=device) * 0.1
    bh = torch.randn(6 * H, device=device) * 0.1
    y, h = custom_gru_v2(x, wi, wh, bi, bh, H, True)
    assert y.shape == (2, 2, 5, H)
    assert h.shape == (2, 2, H)
    assert y.device.type == "npu", f"输出落在 {y.device}，期望 NPU（防止静默 CPU fallback）"


@npu_only()
@requires_compiled_kernel()
def test_kernel_contract_lowlevel_3d_bidir_normalized():
    """底层 API 双向（3D [2,3H] 形态）：修复后由 wrapper 扁平化进 kernel。"""
    from oil_gas_ops_prospect.custom_gru_v2 import custom_gru_v2
    H, F_in = 8, 3
    device = get_device()
    torch.manual_seed(_SEED)
    x = torch.randn(2, 5, F_in, device=device)
    wi = torch.randn(2, 3 * H, F_in, device=device) * 0.1
    wh = torch.randn(2, 3 * H, H, device=device) * 0.1
    bi = torch.randn(2, 3 * H, device=device) * 0.1
    bh = torch.randn(2, 3 * H, device=device) * 0.1
    y, h = custom_gru_v2(x, wi, wh, bi, bh, H, True)
    assert y.shape == (2, 2, 5, H)
    assert h.shape == (2, 2, H)
    assert y.device.type == "npu"


@npu_only()
@requires_compiled_kernel()
def test_module_unidirectional_npu():
    """module 单向 (bias True/False)：修复前即可用，作为对照 + 设备断言。"""
    from oil_gas_ops_prospect.custom_gru_v2 import CustomGRU
    device = get_device()
    torch.manual_seed(_SEED)
    for bias in (True, False):
        model = CustomGRU(3, 8, bidirectional=False, bias=bias).to(device)
        x = torch.randn(2, 5, 3, device=device)
        y, h = model(x)
        assert y.shape == (1, 2, 5, 8)
        assert h.shape == (1, 2, 8)
        assert y.device.type == "npu"
        assert not torch.isnan(y).any()


@npu_only()
@requires_compiled_kernel()
def test_module_bidirectional_npu_d1_regression():
    """module 双向 (bias True/False)：D1 契约缺口回归测试。

    修复前：``aclnnCustomGRUV2GetWorkspaceSize failed: 561002``
    （host tiling 要求双向权重 [6H]，模块曾把单份 [3H] 直接送进去）。
    """
    from oil_gas_ops_prospect.custom_gru_v2 import CustomGRU
    device = get_device()
    torch.manual_seed(_SEED)
    for bias in (True, False):
        model = CustomGRU(3, 8, bidirectional=True, bias=bias).to(device)
        x = torch.randn(2, 5, 3, device=device)
        y, h = model(x)
        assert y.shape == (2, 2, 5, 8)
        assert h.shape == (2, 2, 8)
        assert y.device.type == "npu"
        assert not torch.isnan(y).any()


def test_cpu_fallback_semantics_preserved(monkeypatch):
    """CPU fallback 语义不回归（评审 H2）：monkeypatch 移除强制 NPU 变量。"""
    _cpu_fallback_env_removed(monkeypatch)
    from oil_gas_ops_prospect.custom_gru_v2 import custom_gru_v2, _torch_gru_forward
    H, F_in = 8, 3
    torch.manual_seed(_SEED)
    x = torch.randn(2, 5, F_in)
    wi3 = torch.randn(2, 3 * H, F_in) * 0.1
    wh3 = torch.randn(2, 3 * H, H) * 0.1
    bi3 = torch.randn(2, 3 * H) * 0.1
    bh3 = torch.randn(2, 3 * H) * 0.1
    y_ref, h_ref = _torch_gru_forward(x, wi3, wh3, bi3, bh3, H, True)

    # 3D 权重（wrapper 内部扁平化后 CPU fallback 拆分方向，值应与 3D 历史语义一致）
    y3, h3 = custom_gru_v2(x, wi3, wh3, bi3, bh3, H, True)
    assert (y3 - y_ref).abs().max().item() <= 1e-5
    assert (h3 - h_ref).abs().max().item() <= 1e-5

    # 单份 [3H] + 双向（历史形态，同权重复用）
    wi_dup = torch.stack((wi3[0], wi3[0]), dim=0)
    wh_dup = torch.stack((wh3[0], wh3[0]), dim=0)
    bi_dup = torch.stack((bi3[0], bi3[0]), dim=0)
    bh_dup = torch.stack((bh3[0], bh3[0]), dim=0)
    y_dup_ref, h_dup_ref = _torch_gru_forward(x, wi_dup, wh_dup, bi_dup, bh_dup, H, True)
    y2, h2 = custom_gru_v2(x, wi3[0], wh3[0], bi3[0], bh3[0], H, True)
    assert (y2 - y_dup_ref).abs().max().item() <= 1e-5
    assert (h2 - h_dup_ref).abs().max().item() <= 1e-5


_SHAPE_MATRIX = [
    # (B, T, H, F_in) —— 评审 H1：严格门禁需覆盖多组 T/B/H，并按方向对比
    (1, 1, 8, 3),
    (1, 6, 8, 3),
    (2, 6, 8, 3),
    (2, 100, 8, 3),
    (3, 20, 16, 5),
    (1, 50, 24, 3),
]


@npu_only()
@requires_compiled_kernel()
@pytest.mark.parametrize("shp", _SHAPE_MATRIX)
def test_npu_forward_values_vs_torch_gru(shp):
    """单向 NPU kernel 数值严格门禁 vs PyTorch GRU（y 与 output_h，按方向对比）。"""
    from oil_gas_ops_prospect.custom_gru_v2 import custom_gru_v2
    B, T, H, F_in = shp
    torch.manual_seed(_SEED + B + T + H)
    x_cpu = torch.randn(B, T, F_in)
    wi = torch.randn(3 * H, F_in) * 0.5
    wh = torch.randn(3 * H, H) * 0.5
    bi = torch.randn(3 * H) * 0.5
    bh = torch.randn(3 * H) * 0.5
    y_ref, h_ref = _torch_gru_ref(x_cpu, wi.unsqueeze(0), wh.unsqueeze(0),
                                  bi.unsqueeze(0), bh.unsqueeze(0), H, bidir=False)
    y, h = custom_gru_v2(x_cpu.npu(), wi.npu(), wh.npu(), bi.npu(), bh.npu(), H, False)
    max_diff_y = (y.cpu()[0] - y_ref[0]).abs().max().item()
    max_diff_h = (h.cpu()[0] - h_ref[0]).abs().max().item()
    assert max_diff_y <= _ATOL, f"y kernel vs GRU max_diff={max_diff_y} (B={B},T={T},H={H})"
    assert max_diff_h <= _ATOL, f"output_h kernel vs GRU max_diff={max_diff_h}"


@npu_only()
@requires_compiled_kernel()
@pytest.mark.parametrize("shp", _SHAPE_MATRIX)
def test_npu_bidirectional_values_vs_torch_gru(shp):
    """双向 NPU kernel 数值严格门禁 vs PyTorch GRU（逐方向 y 与 output_h）。"""
    from oil_gas_ops_prospect.custom_gru_v2 import custom_gru_v2
    B, T, H, F_in = shp
    torch.manual_seed(_SEED + B + T + H + 1)
    x_cpu = torch.randn(B, T, F_in)
    wi3 = torch.randn(2, 3 * H, F_in) * 0.5
    wh3 = torch.randn(2, 3 * H, H) * 0.5
    bi3 = torch.randn(2, 3 * H) * 0.5
    bh3 = torch.randn(2, 3 * H) * 0.5
    y_ref, h_ref = _torch_gru_ref(x_cpu, wi3, wh3, bi3, bh3, H, bidir=True)
    y, h = custom_gru_v2(x_cpu.npu(), wi3.npu(), wh3.npu(), bi3.npu(), bh3.npu(), H, True)
    for d in (0, 1):
        max_diff_y = (y.cpu()[d] - y_ref[d]).abs().max().item()
        max_diff_h = (h.cpu()[d] - h_ref[d]).abs().max().item()
        assert max_diff_y <= _ATOL, f"y[{d}] kernel vs GRU max_diff={max_diff_y} (B={B},T={T},H={H})"
        assert max_diff_h <= _ATOL, f"output_h[{d}] kernel vs GRU max_diff={max_diff_h}"


@npu_only()
@requires_compiled_kernel()
def test_npu_module_bidirectional_values_vs_torch_gru():
    """模块 tied 双向终验：独立参考（复制权重 → torch_gru_ref 双向）四元对比。

    评审第二轮 v4 要求：不能只比较 `y[0]/h[0]`；反向方向 `y[1]/h[1]` 也必须
    与"同权复用"语义下的独立参考一致。参考路径：把单份 [3H] 权重显式
    stack 成 [2,3H]（两份相同），用 conftest 的 `torch_gru_ref`（独立再实现，
    与被测 fallback 的 `_torch_gru_forward` 共享实现路径无重叠）计算双向参考。
    """
    from oil_gas_ops_prospect.custom_gru_v2 import CustomGRU
    device = get_device()
    H, F_in = 8, 3
    torch.manual_seed(_SEED)
    model = CustomGRU(F_in, H, bidirectional=True, bias=True).to(device)
    x_cpu = torch.randn(2, 6, F_in)
    with torch.no_grad():
        wi = model.weight_ih.detach().cpu()
        wh = model.weight_hh.detach().cpu()
        bi = model.bias_ih.detach().cpu()
        bh = model.bias_hh.detach().cpu()
    # tied → 双向双份（同权复用），独立参考 torch_gru_ref(bidir=True)
    wi3 = torch.stack((wi, wi), dim=0)
    wh3 = torch.stack((wh, wh), dim=0)
    bi3 = torch.stack((bi, bi), dim=0)
    bh3 = torch.stack((bh, bh), dim=0)
    y_ref, h_ref = _torch_gru_ref(x_cpu, wi3, wh3, bi3, bh3, H, bidir=True)
    y, h = model(x_cpu.npu())
    max_diffs = {}
    for d in (0, 1):
        max_diffs[f"y[{d}]"] = (y.cpu()[d] - y_ref[d]).abs().max().item()
        max_diffs[f"output_h[{d}]"] = (h.cpu()[d] - h_ref[d]).abs().max().item()
    for name, v in max_diffs.items():
        assert v <= _ATOL, f"module-bidir tied {name} vs GRU(bidir ref) diff={v}"
