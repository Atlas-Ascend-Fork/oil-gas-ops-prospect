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

from oil_gas_ops_prospect.complex_reciprocal import complex_reciprocal


@pytest.fixture(params=["cpu", "npu"])
def device(request):
    if request.param == "npu":
        pytest.importorskip("torch_npu")
        if not torch.npu.is_available():
            pytest.skip("NPU is not available")
        return torch.device("npu:0")
    return torch.device("cpu")


def reference_parts(xr, xi, shape):
    return [
        value.detach().cpu().clone().requires_grad_(True)
        if isinstance(value, torch.Tensor)
        else torch.full(shape, value)
        for value in (xr, xi)
    ]


def assert_parts_gradients(xr, xi, shape):
    real_ref, imag_ref = reference_parts(xr, xi, shape)
    expected = 1 / torch.complex(real_ref, imag_ref)
    output = complex_reciprocal(xr, xi)
    torch.testing.assert_close(
        output.detach().cpu().reshape(-1), expected.detach().reshape(-1)
    )
    assert output.requires_grad
    (0.7 * output.real + 1.3 * output.imag).sum().backward()
    (0.7 * expected.real + 1.3 * expected.imag).sum().backward()
    for actual, reference in zip((xr, xi), (real_ref, imag_ref)):
        if isinstance(actual, torch.Tensor):
            assert actual.grad is not None
            torch.testing.assert_close(actual.grad.cpu(), reference.grad)
    return output


@pytest.mark.parametrize("shape", [(), (1,), (1, 1)])
@pytest.mark.parametrize("parts", ["both_tensors", "real_number", "imag_number"])
def test_scalar_parts_gradients(device, shape, parts):
    xr = torch.full(shape, 2.0, device=device, requires_grad=True)
    xi = torch.full(shape, 3.0, device=device, requires_grad=True)
    if parts == "real_number":
        xr = 2.0
    elif parts == "imag_number":
        xi = 3.0
    output = assert_parts_gradients(xr, xi, shape)
    assert output.shape == ()
    assert output.dtype == torch.complex64


@pytest.mark.parametrize("shape", [(0,), (2, 0)])
@pytest.mark.parametrize("parts", ["both_tensors", "real_number", "imag_number"])
def test_empty_parts_gradients(device, shape, parts):
    xr = torch.full(shape, 2.0, device=device, requires_grad=True)
    xi = torch.full(shape, 3.0, device=device, requires_grad=True)
    if parts == "real_number":
        xr = 2.0
    elif parts == "imag_number":
        xi = 3.0
    output = assert_parts_gradients(xr, xi, shape)
    assert output.shape == shape
    assert output.dtype == torch.complex64


def test_zero_scalar_parts_use_tensor_reciprocal_semantics(device):
    xr = torch.tensor(0.0, device=device)
    xi = torch.tensor(0.0, device=device)
    expected = 1 / torch.complex(xr.cpu(), xi.cpu())
    output = complex_reciprocal(xr, xi)
    torch.testing.assert_close(output.cpu(), expected, equal_nan=True)


@pytest.mark.parametrize("shape", [(), (1,), (1, 1)])
def test_cpu_double_derivatives(shape):
    xr = torch.full(shape, 2.0, dtype=torch.float64, requires_grad=True)
    xi = torch.full(shape, 3.0, dtype=torch.float64, requires_grad=True)
    assert complex_reciprocal(xr, xi).dtype == torch.complex128
    assert torch.autograd.gradcheck(complex_reciprocal, (xr, xi))
    assert torch.autograd.gradgradcheck(complex_reciprocal, (xr, xi))
