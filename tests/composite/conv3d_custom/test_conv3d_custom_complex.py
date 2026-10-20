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

"""Compare complex Conv3d gradients with the native public implementation."""

import pytest
import torch
import torch.nn.functional as F

from oil_gas_ops_prospect.conv3d_custom import Conv3d
from oil_gas_ops_prospect.conv3d_custom.functional import conv3d


@pytest.fixture(params=[
    pytest.param(("cpu", torch.complex128), id="cpu-complex128"),
    pytest.param(("cpu", torch.complex64), id="cpu-complex64"),
    pytest.param(("npu", torch.complex64), id="npu-complex64"),
])
def backend(request):
    device, dtype = request.param
    if device == "npu":
        pytest.importorskip("torch_npu")
        if not torch.npu.is_available():
            pytest.skip("requires an available NPU")
        previous_hf32 = torch.npu.conv.allow_hf32
        torch.npu.conv.allow_hf32 = False
        try:
            yield torch.device(device), dtype
        finally:
            torch.npu.conv.allow_hf32 = previous_hf32
    else:
        yield torch.device(device), dtype


def _random(shape, backend, requires_grad=True):
    device, dtype = backend
    # CANN 9.0 does not implement complex normal_ on NPU; generate fixtures on CPU.
    return torch.randn(shape, dtype=dtype).to(device).requires_grad_(requires_grad)


def _assert_equal(actual, expected):
    assert actual.shape == expected.shape
    actual = actual.detach().cpu()
    expected = expected.detach().cpu()
    delta = actual - expected
    if actual.dtype == torch.complex128:
        error = delta.abs().max().item()
        tolerance = 1e-12
    else:
        error = (delta.norm() / expected.detach().norm().clamp_min(1e-30)).item()
        tolerance = 1e-5
    assert error <= tolerance, f"complex output/gradient error {error:.3e} > {tolerance:.3e}"


def _compare(function, reference, leaves):
    native = [value.detach().clone().requires_grad_(value.requires_grad)
              if value is not None else None for value in leaves]
    # Prove native forward/backward support before testing the custom implementation.
    expected = reference(*native)
    grad_output = torch.randn(expected.shape, dtype=expected.dtype).to(expected.device)
    expected_gradients = torch.autograd.grad(
        expected, [value for value in native if value is not None and value.requires_grad],
        grad_output,
    )
    actual = function(*leaves)
    _assert_equal(actual, expected)
    gradients = torch.autograd.grad(
        actual, [value for value in leaves if value is not None and value.requires_grad],
        grad_output,
    )
    assert len(gradients) == len(expected_gradients)
    for gradient, expected_gradient in zip(gradients, expected_gradients):
        _assert_equal(gradient, expected_gradient)


CASES = [
    pytest.param(2, 1, 0, 1, 1, True, False, id="basic"),
    pytest.param(2, (2, 1, 2), 0, 1, 1, True, False, id="stride"),
    pytest.param(2, 1, 1, (2, 1, 2), 1, False, False, id="dilation"),
    pytest.param(2, 1, 0, 1, 2, True, False, id="groups"),
    pytest.param(2, 2, 1, 2, 2, False, False, id="grouped-stride-dilation"),
    pytest.param(3, 1, "same", 1, 1, True, False, id="same"),
    pytest.param(2, 1, 0, 1, 1, True, True, id="noncontiguous"),
]


@pytest.mark.parametrize("kernel,stride,padding,dilation,groups,bias,noncontiguous", CASES)
def test_complex_function_gradients(
        backend, kernel, stride, padding, dilation, groups, bias, noncontiguous):
    x = _random((1, 2, 5, 5, 5), backend)
    if noncontiguous:
        x = x.transpose(-1, -2)
    weight = _random((4, 2 // groups, kernel, kernel, kernel), backend)
    offset = _random((4,), backend) if bias else None
    _compare(
        lambda a, b, c: conv3d(a, b, c, stride, padding, dilation, groups),
        lambda a, b, c: F.conv3d(a, b, c, stride, padding, dilation, groups),
        [x, weight, offset],
    )


@pytest.mark.parametrize("mode", ["all", "input", "weight", "bias"])
def test_complex_selected_gradients(backend, mode):
    leaves = [
        _random((1, 2, 4, 4, 4), backend, mode in ("all", "input")),
        _random((3, 2, 2, 2, 2), backend, mode in ("all", "weight")),
        _random((3,), backend, mode in ("all", "bias")),
    ]
    _compare(conv3d, F.conv3d, leaves)


def test_complex_module_gradients(backend):
    device, dtype = backend
    module = Conv3d(2, 3, 2).to(device=device, dtype=dtype)
    module.use_custom = True
    x = _random((1, 2, 4, 4, 4), backend)
    _compare(
        lambda a, b, c: module(a),
        lambda a, b, c: F.conv3d(a, b, c),
        [x, module.weight, module.bias],
    )


@pytest.mark.parametrize("stride", [1, 2])
def test_complex_double_backward_cpu(stride):
    backend = (torch.device("cpu"), torch.complex128)
    leaves = (
        _random((1, 1, 3, 3, 3), backend),
        _random((1, 1, 2, 2, 2), backend),
        _random((1,), backend),
    )

    def function(x, weight, bias):
        return conv3d(x, weight, bias, stride=stride)

    assert torch.autograd.gradcheck(function, leaves, fast_mode=True)
    assert torch.autograd.gradgradcheck(function, leaves, fast_mode=True)
