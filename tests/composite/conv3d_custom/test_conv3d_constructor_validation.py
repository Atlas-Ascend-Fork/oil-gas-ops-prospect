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

import pytest
import torch

from oil_gas_ops_prospect.conv3d_custom import Conv3d as CustomConv3d


@pytest.mark.parametrize('kwargs', [
    {'in_channels': 4, 'out_channels': 4, 'groups': 0},
    {'in_channels': 4, 'out_channels': 4, 'groups': -1},
    {'in_channels': 3, 'out_channels': 4, 'groups': 2},
    {'in_channels': 4, 'out_channels': 5, 'groups': 2},
    {'in_channels': 4, 'out_channels': 4, 'padding_mode': 'wrap'},
])
def test_invalid_constructor_matches_native_value_error(kwargs):
    with pytest.raises(ValueError):
        torch.nn.Conv3d(kernel_size=3, **kwargs)
    with pytest.raises(ValueError):
        CustomConv3d(kernel_size=3, **kwargs)


@pytest.mark.parametrize('kwargs', [
    {'in_channels': 2, 'out_channels': 3, 'groups': 1},
    {'in_channels': 4, 'out_channels': 6, 'groups': 2},
    {'in_channels': 4, 'out_channels': 4, 'groups': 4},
])
def test_valid_cpu_initialization_and_forward_match_native(kwargs):
    torch.manual_seed(52)
    native = torch.nn.Conv3d(kernel_size=3, **kwargs)
    torch.manual_seed(52)
    custom = CustomConv3d(kernel_size=3, **kwargs)
    for key, expected in native.state_dict().items():
        torch.testing.assert_close(custom.state_dict()[key], expected, rtol=0, atol=0)
    data = torch.randn(1, kwargs['in_channels'], 4, 4, 4)
    torch.testing.assert_close(custom(data), native(data), rtol=0, atol=0)
