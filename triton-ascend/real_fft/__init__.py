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

"""Real-input FFT / iFFT Triton-Ascend operators.

一维入口为 ``real_rfft`` / ``real_irfft``；二维入口 ``real_rfft2`` / ``real_irfft2``
由它们组合而成（见 ``fft2d.py``），不引入新的 kernel。
"""

__all__ = ["make_basis", "real_irfft", "real_irfft2", "real_rfft", "real_rfft2"]

from .fft2d import real_irfft2, real_rfft2
from .triton_fft_real import make_basis, real_irfft, real_rfft
