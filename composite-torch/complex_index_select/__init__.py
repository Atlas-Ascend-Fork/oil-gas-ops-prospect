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

"""complex_index_select：复数版 ``torch.index_select``。

torch_npu 当前不支持复数 ``index_select`` 的反向（``aclnnIndexAdd`` 报
161002），且 complex64 前向被派发到 AiCPU 标量核（``GatherV2``），大形状比
float32 的 ``GatherV3`` 慢两个数量级（见 CANN issue #61）。本算子用纯
PyTorch 算子组合出等价语义：

- 连续输入且 ``dim`` 非末轴时走 flat 路径：``view(float)`` 后 gather，
  带宽与 float32 基线一致；
- 末轴 ``dim``（含 1-D）或非连续输入走 real/imag 路径。

用法::

    from oil_gas_ops_prospect.complex_index_select import complex_index_select

    out = complex_index_select(z, 1, index)     # 等价 torch.index_select(z, 1, index)
    out = complex_index_select(z, -1, index)    # 负数 dim 与 torch 归一化一致

实数 dtype 不在本算子职责内：调用方应直接用 ``torch.index_select``。
"""

__all__ = ["complex_index_select"]

from .complex_index_select import complex_index_select
