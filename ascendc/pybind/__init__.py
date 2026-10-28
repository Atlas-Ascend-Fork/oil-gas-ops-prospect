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

"""oil_gas_ops_prospect：按算子名暴露的 Python 包根（AscendC 侧源码在本目录）。

各算子是本包下的模块，不要在这里 ``from .complex_mul import complex_mul``，
否则会把同名模块遮蔽成函数。

用法::

    from oil_gas_ops_prospect.complex_mul import complex_mul
    from oil_gas_ops_prospect.fused_bias_softmax import fused_bias_softmax
    from oil_gas_ops_prospect.conv3d_gelu import conv3d_gelu
"""

from . import _aclnn_cache  # noqa: F401
