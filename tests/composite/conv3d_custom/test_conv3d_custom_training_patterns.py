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

"""conv3d_custom 精度测试（二）：实际训练提取的 11 种调用模式（前向 + 反向）。

用例来源：在 Conv3d.forward 注入日志、跑 1 个 epoch 训练后去重得到的
全部唯一 Conv3d 调用模式（共 11 种）。参数覆盖 kernel 1/3/5/7/8/9、
stride 1/4、groups 1/2/7、bias 有/无、padding_mode zeros/replicate。

对比方式：Conv3d（强制 use_custom）vs nn.Conv3d 原生，
前向 + grad_input / grad_weight / grad_bias 全对比。

- float64（空间维度缩至 20）：最大绝对误差阈值 1e-12，数学等价性；
- float32（同为缩尺）：L2 相对误差阈值 1e-5（rtol 口径，见 _common.py），
  对应 NPU 实机精度档（NPU 实机尺寸对比由
  bench_conv3d_custom_perf.py 承担）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from _common import assert_errors, criterion_for, module_level_errors  # noqa: E402

# 实际训练提取的 11 种调用模式（空间维度缩至 20）。
# 用例字段依次为：id、input_shape、weight_shape、S、P、G、bias、padding_mode。
TRAINING_PATTERNS = [
    ("train01_smooth_1x1", (1, 1, 20, 20, 20), (32, 1, 1, 1, 1), 1, 0, 1, True, "zeros"),
    ("train02_proj_1x1", (1, 32, 20, 20, 20), (1, 32, 1, 1, 1), 1, 0, 1, True, "zeros"),
    ("train03_k5_replicate", (1, 7, 20, 20, 20), (7, 7, 5, 5, 5), 1, 0, 1, True, "replicate"),
    ("train04_dw_k7_g7", (1, 7, 20, 20, 20), (7, 1, 7, 7, 7), 1, 0, 7, True, "replicate"),
    ("train05_dw_k9_g7", (1, 7, 20, 20, 20), (7, 1, 9, 9, 9), 1, 0, 7, True, "replicate"),
    ("train06_k3_replicate", (1, 7, 20, 20, 20), (7, 7, 3, 3, 3), 1, 0, 1, True, "replicate"),
    ("train07_stride4_k8_nobias", (1, 7, 20, 20, 20), (19, 7, 8, 8, 8), 4, 0, 1, False, "replicate"),
    ("train08_patch_embed", (1, 19, 20, 20, 20), (512, 19, 1, 1, 1), 1, 0, 1, True, "zeros"),
    ("train09_up_1x1", (1, 512, 20, 20, 20), (1024, 512, 1, 1, 1), 1, 0, 1, True, "zeros"),
    ("train10_down_1x1", (1, 1024, 20, 20, 20), (512, 1024, 1, 1, 1), 1, 0, 1, True, "zeros"),
    ("train11_dw_smooth_g2", (1, 2, 20, 20, 20), (2, 1, 3, 3, 3), 1, 0, 2, True, "replicate"),
]

DTYPES = [torch.float64, torch.float32]
DTYPE_IDS = ["f64", "f32"]


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
@pytest.mark.parametrize("pattern", TRAINING_PATTERNS, ids=[c[0] for c in TRAINING_PATTERNS])
def test_training_patterns(pattern, dtype):
    """11 种实际训练调用模式：模块级前向 + 三梯度与 nn.Conv3d 原生对比。"""
    metric, tol = criterion_for(dtype)
    _, in_shape, w_shape, s, p, g, has_bias, padding_mode = pattern
    errs = module_level_errors(
        in_shape, w_shape[0], w_shape[2:],
        stride=s, padding=p, groups=g, bias=has_bias,
        padding_mode=padding_mode, dtype=dtype, metric=metric,
    )
    assert_errors(errs, pattern[0], tol)


# CPU autocast changes the convolution operands even when its parameters are
# float32. Check the full public APIs and gradients of the original leaves.
def _cpu_autocast_result(samples, grad_output, required, *, entry, native, enabled, config):
    from oil_gas_ops_prospect.conv3d_custom import Conv3d
    from oil_gas_ops_prospect.conv3d_custom.functional import conv3d

    x, weight, bias = [value.detach().clone() if value is not None else None for value in samples]
    x.requires_grad_(required[0])
    if entry == "module":
        layer_type = torch.nn.Conv3d if native else Conv3d
        layer = layer_type(x.shape[1], weight.shape[0], weight.shape[2:], bias=bias is not None, **config)
        layer = layer.to(dtype=x.dtype)
        with torch.no_grad():
            layer.weight.copy_(weight)
            if bias is not None:
                layer.bias.copy_(bias)
        layer.weight.requires_grad_(required[1])
        if layer.bias is not None:
            layer.bias.requires_grad_(required[2])
        if not native:
            layer.use_custom = True  # The documented CPU validation path.
        leaves = (x, layer.weight, layer.bias)
    else:
        weight.requires_grad_(required[1])
        if bias is not None:
            bias.requires_grad_(required[2])
        leaves = (x, weight, bias)

    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=enabled):
        if entry == "module":
            result = layer(x)
        else:
            function = torch.nn.functional.conv3d if native else conv3d
            result = function(x, weight, bias, **config)
    selected = [value for value in leaves if value is not None and value.requires_grad]
    grads = torch.autograd.grad(result, selected, grad_output.to(result.dtype)) if selected else ()
    return result.detach(), tuple(value.detach() for value in grads)


_CPU_GRAD_MASKS = [
    (x, weight, bias)
    for x in (False, True)
    for weight in (False, True)
    for bias in (False, True)
]


@pytest.mark.parametrize("required", _CPU_GRAD_MASKS)
@pytest.mark.parametrize("entry", ["function", "module"])
@pytest.mark.parametrize("enabled", [False, True], ids=["ordinary", "autocast"])
def test_cpu_autocast_uses_effective_quantized_operands(required, entry, enabled):
    x = torch.tensor([1.001, 1.003]).reshape(1, 1, 1, 1, 2)
    weight = torch.tensor([1.001, 1.003]).reshape(2, 1, 1, 1, 1)
    bias = torch.zeros(2)
    grad_output = torch.tensor([0.75, -0.75, -0.75, 0.75]).reshape(1, 2, 1, 1, 2)
    args = dict(samples=(x, weight, bias), grad_output=grad_output, required=required,
                entry=entry, enabled=enabled, config={})
    expected, expected_grads = _cpu_autocast_result(native=True, **args)
    actual, actual_grads = _cpu_autocast_result(native=False, **args)
    assert actual.dtype == expected.dtype
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert len(actual_grads) == len(expected_grads)
    for grad, reference in zip(actual_grads, expected_grads):
        assert grad.dtype == reference.dtype == torch.float32
        if enabled:
            # Both 1.001 and 1.003 round to 1 in bfloat16. Using the original
            # float32 operands instead leaves an incorrect ~0.0015 residual.
            assert torch.count_nonzero(reference).item() == 0
            assert torch.count_nonzero(grad).item() == 0
        else:
            torch.testing.assert_close(grad, reference, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("config", [
    {"padding": 1},
    {"groups": 2, "stride": (2, 1, 1), "padding": (1, 0, 1)},
    {"dilation": (2, 1, 2), "padding": (1, 0, 1)},
])
def test_cpu_autocast_random_output_and_three_gradients(config):
    generator = torch.Generator().manual_seed(2026)
    groups = config.get("groups", 1)
    x = torch.randn(2, 4, 5, 6, 7, generator=generator)
    weight = torch.randn(6, 4 // groups, 2, 3, 2, generator=generator)
    bias = torch.randn(6, generator=generator)
    shape = torch.nn.functional.conv3d(x, weight, bias, **config).shape
    grad_output = torch.randn(shape, generator=generator)
    args = dict(samples=(x, weight, bias), grad_output=grad_output, required=(True, True, True),
                entry="function", enabled=True, config=config)
    expected, expected_grads = _cpu_autocast_result(native=True, **args)
    actual, actual_grads = _cpu_autocast_result(native=False, **args)
    assert actual.dtype == expected.dtype == torch.bfloat16
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for grad, reference in zip(actual_grads, expected_grads):
        assert grad.dtype == reference.dtype == torch.float32
        error = (grad.double() - reference.double()).norm() / reference.double().norm().clamp_min(1e-30)
        assert error.item() <= 1e-2, error.item()


@pytest.mark.parametrize("entry", ["function", "module"])
def test_cpu_autocast_preserves_float64(entry):
    x = torch.tensor([1.001, 1.003], dtype=torch.float64).reshape(1, 1, 1, 1, 2)
    weight = torch.tensor([1.001, 1.003], dtype=torch.float64).reshape(2, 1, 1, 1, 1)
    bias = torch.zeros(2, dtype=torch.float64)
    grad_output = torch.tensor([0.75, -0.75, -0.75, 0.75], dtype=torch.float64).reshape(1, 2, 1, 1, 2)
    args = dict(samples=(x, weight, bias), grad_output=grad_output, required=(True, True, True),
                entry=entry, enabled=True, config={})
    expected, expected_grads = _cpu_autocast_result(native=True, **args)
    actual, actual_grads = _cpu_autocast_result(native=False, **args)
    assert actual.dtype == expected.dtype == torch.float64
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for grad, reference in zip(actual_grads, expected_grads):
        assert grad.dtype == reference.dtype == torch.float64
        torch.testing.assert_close(grad, reference, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("enabled", [False, True], ids=["ordinary", "autocast"])
def test_cpu_autocast_without_bias(enabled):
    generator = torch.Generator().manual_seed(13)
    x = torch.randn(1, 2, 4, 4, 4, generator=generator)
    weight = torch.randn(3, 2, 3, 3, 3, generator=generator)
    shape = torch.nn.functional.conv3d(x, weight, padding=1).shape
    grad_output = torch.randn(shape, generator=generator)
    args = dict(samples=(x, weight, None), grad_output=grad_output, required=(True, True, False),
                entry="function", enabled=enabled, config={"padding": 1})
    expected, expected_grads = _cpu_autocast_result(native=True, **args)
    actual, actual_grads = _cpu_autocast_result(native=False, **args)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for grad, reference in zip(actual_grads, expected_grads):
        assert grad.dtype == reference.dtype == torch.float32
        if enabled:
            error = (grad.double() - reference.double()).norm() / reference.double().norm().clamp_min(1e-30)
            assert error.item() <= 1e-2, error.item()
        else:
            torch.testing.assert_close(grad, reference, rtol=1e-5, atol=1e-5)
