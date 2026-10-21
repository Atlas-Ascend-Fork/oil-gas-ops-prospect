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

# 仓库版本定义。需与 pyproject.toml / setup.py 中的 version 保持一致。
set(OIL_GAS_OPS_PROSPECT_VERSION "0.1.0")

# 已验证通过的 CANN 版本。
set(OIL_GAS_OPS_PROSPECT_CANN_VERSION "9.0.0")

# 已验证通过的 SOC 型号，与 ascendc/scripts/config.py 的 SUPPORTED_SOC_VERSIONS 对应。
set(OIL_GAS_OPS_PROSPECT_SOC_VERSIONS "Ascend910B1")

# 构建期依赖（由 CANN Toolkit 提供，不在本仓内构建）。
#   msopgen / opbuild  —  AscendC 自定义算子工程生成与算子编译
#   bisheng-compiler   —  AscendC kernel 编译
#   googletest         —  ophost 侧 tiling UT，见 Third_Party_Open_Source_Software_List.yaml
set(OIL_GAS_OPS_PROSPECT_BUILD_DEPENDENCIES "toolkit>=9.0.0;googletest>=1.14.0")

# 运行期依赖。
#   runtime / opbase   —  aclnn 接口调用
#   torch / torch_npu  —  Python 侧算子封装与 autograd
#   triton-ascend      —  Triton-Ascend 算子；由用户从 Ascend 软件源预装
set(OIL_GAS_OPS_PROSPECT_RUN_DEPENDENCIES "toolkit>=9.0.0;torch;torch-npu;triton-ascend==3.2.1")
