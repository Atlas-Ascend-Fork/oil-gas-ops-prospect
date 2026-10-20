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

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F


SOURCE = Path(__file__).resolve().parents[3] / 'composite-torch/conv3d_custom/functional.py'
SPEC = importlib.util.spec_from_file_location('conv3d_under_test', SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@pytest.mark.parametrize('option', ['stride', 'padding', 'dilation'])
@pytest.mark.parametrize('value', [1, [1, 1, 1], (1, 1, 1), np.int64(1)])
def test_sequence_parameters_match_native_forward_and_backward(option, value):
    torch.manual_seed(72)
    x = torch.randn(1, 2, 4, 4, 4, dtype=torch.float64, requires_grad=True)
    weight = torch.randn(3, 2, 2, 2, 2, dtype=torch.float64, requires_grad=True)
    arguments = {option: value}
    actual = MODULE.conv3d(x, weight, **arguments)
    expected = F.conv3d(x, weight, **arguments)
    torch.testing.assert_close(actual, expected)
    upstream = torch.randn_like(expected)
    actual_grad = torch.autograd.grad(actual, (x, weight), upstream, retain_graph=True)
    expected_grad = torch.autograd.grad(expected, (x, weight), upstream)
    for left, right in zip(actual_grad, expected_grad):
        torch.testing.assert_close(left, right)
