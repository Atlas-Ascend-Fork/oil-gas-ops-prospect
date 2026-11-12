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

"""Unbatched Conv3d forward/backward parity through the public entry points."""

import pytest
import torch
import torch.nn.functional as F

from oil_gas_ops_prospect.conv3d_custom import Conv3d
from oil_gas_ops_prospect.conv3d_custom.functional import conv3d


@pytest.fixture(params=["cpu", "npu"])
def device(request):
    if request.param == "npu":
        if not hasattr(torch, "npu") or not torch.npu.is_available():
            pytest.skip("requires an available torch_npu device")
    return torch.device(request.param)


def _random(shape, device, requires_grad=True):
    dtype = torch.float64 if device.type == "cpu" else torch.float32
    return torch.randn(shape, dtype=dtype).to(device).requires_grad_(requires_grad)


def _assert_equal(actual, expected):
    assert actual.shape == expected.shape
    delta = (actual - expected).detach()
    if actual.dtype == torch.float64:
        error = delta.abs().max().item()
        tolerance = 1e-12
    else:
        error = (delta.norm() / expected.detach().norm().clamp_min(1e-30)).item()
        tolerance = 1e-5
    assert error <= tolerance, f"gradient/output error {error:.3e} > {tolerance:.3e}"
    return error


def _compare(function, reference, leaves, reference_leaves):
    # Evaluate the native path first, so unsupported inputs cannot masquerade as a bug.
    expected = reference(*reference_leaves)
    grad_output = torch.randn_like(expected)
    reference_inputs = [t for t in reference_leaves if t is not None and t.requires_grad]
    expected_gradients = torch.autograd.grad(expected, reference_inputs, grad_output)
    actual = function(*leaves)
    errors = [_assert_equal(actual, expected)]
    inputs = [t for t in leaves if t is not None and t.requires_grad]
    gradients = torch.autograd.grad(actual, inputs, grad_output)
    assert len(gradients) == len(expected_gradients)
    for gradient, expected_gradient in zip(gradients, expected_gradients):
        errors.append(_assert_equal(gradient, expected_gradient))
    print(f"precision device={actual.device} dtype={actual.dtype} shape={tuple(actual.shape)} max_error={max(errors):.6e}")


CASES = [
    pytest.param(2, 1, 0, 1, 1, True, id="basic"),
    pytest.param(2, (2, 1, 2), 0, 1, 1, True, id="stride"),
    pytest.param(2, 1, 1, (2, 1, 2), 1, False, id="dilation"),
    pytest.param(2, 2, 1, 2, 2, True, id="grouped-stride-dilation"),
    pytest.param(2, 1, 0, 1, 2, True, id="groups"),
    pytest.param(1, 1, 2, 1, 1, True, id="negative-backward-padding"),
    pytest.param(2, 2, "valid", 1, 2, False, id="valid"),
    pytest.param(2, 1, "same", 1, 2, True, id="same-even-kernel"),
]


@pytest.mark.parametrize("batched", [False, True], ids=["unbatched", "batched"])
@pytest.mark.parametrize("kernel,stride,padding,dilation,groups,has_bias", CASES)
def test_forward_and_all_gradients(device, batched, kernel, stride, padding, dilation, groups, has_bias):
    if device.type == "npu" and kernel == 1 and padding == 2:
        pytest.skip("native NPU backward rejects negative backprop padding; covered on CPU")
    torch.manual_seed(2026)
    shape = (4, 5, 6, 7) if not batched else (1, 4, 5, 6, 7)
    x = _random(shape, device)
    weight = _random((4, 4 // groups, kernel, kernel, kernel), device)
    bias = _random((4,), device) if has_bias else None
    leaves = [x, weight, bias]
    reference = [t.detach().clone().requires_grad_() if t is not None else None for t in leaves]
    options = dict(stride=stride, padding=padding, dilation=dilation, groups=groups)
    _compare(lambda *args: conv3d(*args, **options), lambda *args: F.conv3d(*args, **options),
             leaves, reference)


@pytest.mark.parametrize("batched", [False, True], ids=["unbatched", "batched"])
@pytest.mark.parametrize("requires", [(True, False, False), (False, True, False),
                                     (False, False, True), (True, True, True)],
                         ids=["input-only", "weight-only", "bias-only", "all"])
def test_selected_gradients(device, batched, requires):
    torch.manual_seed(2026)
    shape = (2, 4, 5, 6) if not batched else (1, 2, 4, 5, 6)
    leaves = [_random(shape, device, requires[0]),
              _random((3, 2, 2, 2, 2), device, requires[1]),
              _random((3,), device, requires[2])]
    reference = [t.detach().clone().requires_grad_(t.requires_grad) for t in leaves]
    _compare(lambda *args: conv3d(*args, stride=2), lambda *args: F.conv3d(*args, stride=2),
             leaves, reference)


@pytest.mark.parametrize("batched", [False, True], ids=["unbatched", "batched"])
@pytest.mark.parametrize("padding", [1, "same"], ids=["numeric", "same"])
def test_module_custom_path(device, batched, padding):
    torch.manual_seed(2026)
    dtype = torch.float64 if device.type == "cpu" else torch.float32
    native = torch.nn.Conv3d(2, 3, 3, padding=padding).to(device=device, dtype=dtype)
    custom = Conv3d(2, 3, 3, padding=padding).to(device=device, dtype=dtype)
    custom.load_state_dict(native.state_dict())
    custom.use_custom = True
    shape = (2, 4, 5, 6) if not batched else (1, 2, 4, 5, 6)
    x = _random(shape, device)
    reference_x = x.detach().clone().requires_grad_()
    _compare(lambda value, *_params: custom(value), lambda value, *_params: native(value),
             [x, custom.weight, custom.bias], [reference_x, native.weight, native.bias])


@pytest.mark.parametrize("stride", [1, 2])
def test_unbatched_double_backward(stride):
    torch.manual_seed(2026)
    x = torch.randn(1, 3, 3, 3, dtype=torch.float64, requires_grad=True)
    weight = torch.randn(1, 1, 2, 2, 2, dtype=torch.float64, requires_grad=True)
    bias = torch.randn(1, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda *args: conv3d(*args, stride=stride),
                                    (x, weight, bias), fast_mode=True)
    assert torch.autograd.gradgradcheck(lambda *args: conv3d(*args, stride=stride),
                                        (x, weight, bias), fast_mode=True)
