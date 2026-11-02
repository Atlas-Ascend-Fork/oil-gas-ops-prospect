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

import torch
import torch_npu  # noqa: F401
torch.npu.set_device(0)
from oil_gas_ops_prospect.custom_gru_v2 import custom_gru_v2  # noqa: E402
x = torch.randn(32, 100, 3, device='npu:0')
wi = torch.randn(72, 3, device='npu:0') * 0.1
wh = torch.randn(72, 24, device='npu:0') * 0.1
bi = torch.randn(72, device='npu:0') * 0.1
bh = torch.randn(72, device='npu:0') * 0.1
y, h = custom_gru_v2(x, wi, wh, bi, bh, 24, False)
print(f'racecheck batch32: y.shape={y.shape}')
torch.npu.synchronize()
print('racecheck: PASSED')
