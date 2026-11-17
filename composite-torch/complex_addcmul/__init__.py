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

"""complex_addcmul：复数三输入逐元素算子 ``input + value * tensor1 * tensor2``。

torch_npu 当前不支持 complex64 / complex128 的 ``torch.addcmul``，
``aclnnAddcmul`` 直接报 ``AclNN_Parameter_Error(EZ1001): Tensor self not
implemented for DT_COMPLEX64``（见 CANN issue #29）。本算子用纯 PyTorch 算子
组合出等价语义，并把复数乘法交给 ``complex_mul``：

- NPU 且 ``OIL_GAS_OPS_COMPLEX_MUL_IMPL=ascendc`` 时，``complex64`` 的乘法
  落到 AscendC ``ComplexMul`` / ``ComplexMulGrad``；``complex128`` 因该算子
  只接受 float32，始终走 torch 表达式；
- 其余情况走 ``complex_mul`` 内置的 torch 参考实现。

用法::

    from oil_gas_ops_prospect.complex_addcmul import complex_addcmul

    out = complex_addcmul(input, tensor1, tensor2)          # input + tensor1 * tensor2
    out = complex_addcmul(input, tensor1, tensor2, value=2.5)
    out = complex_addcmul(input, tensor1, tensor2, value=1 + 2j)

实数 dtype 不在本算子职责内：调用方应直接用 ``torch.addcmul``。
"""

__all__ = ["complex_addcmul"]

from .complex_addcmul import complex_addcmul
