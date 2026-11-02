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
x = torch.randn(1, 100, 3, device='npu:0')
wi = torch.randn(72, 3, device='npu:0') * 0.1
wh = torch.randn(72, 24, device='npu:0') * 0.1
bi = torch.randn(72, device='npu:0') * 0.1
bh = torch.randn(72, device='npu:0') * 0.1
y, h = custom_gru_v2(x, wi, wh, bi, bh, 24, False)
print(f'memcheck standard: y.shape={y.shape}, h.shape={h.shape}')
wi2 = torch.randn(48, 3, device='npu:0') * 0.1
wh2 = torch.randn(48, 8, device='npu:0') * 0.1
bi2 = torch.randn(48, device='npu:0') * 0.1
bh2 = torch.randn(48, device='npu:0') * 0.1
x2 = torch.randn(1, 50, 3, device='npu:0')
y2, h2 = custom_gru_v2(x2, wi2, wh2, bi2, bh2, 8, True)
print(f'memcheck bidir: y2.shape={y2.shape}, h2.shape={h2.shape}')
x3 = torch.randn(1, 1, 3, device='npu:0')
wi3 = torch.randn(24, 3, device='npu:0') * 0.1
wh3 = torch.randn(24, 8, device='npu:0') * 0.1
bi3 = torch.randn(24, device='npu:0') * 0.1
bh3 = torch.randn(24, device='npu:0') * 0.1
y3, h3 = custom_gru_v2(x3, wi3, wh3, bi3, bh3, 8, False)
print(f'memcheck single_step: y3.shape={y3.shape}')
torch.npu.synchronize()
print('memcheck: ALL PASSED')
