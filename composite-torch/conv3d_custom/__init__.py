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

"""Conv3d / conv3d：torch.nn.Conv3d / F.conv3d 的等价替换（纯 PyTorch 组合实现）。

用法（与 torch 原生一一对应，仅引用的包不同）::

    from oil_gas_ops_prospect.conv3d_custom import Conv3d
    from oil_gas_ops_prospect.conv3d_custom.functional import conv3d

语义、约束与精度验证见本目录 ``conv3d.py`` / ``functional.py`` 与
``docs/zh/op_list.md``。
"""

__all__ = ["Conv3d"]

from .conv3d import Conv3d
