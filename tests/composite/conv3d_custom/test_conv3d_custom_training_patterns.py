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
