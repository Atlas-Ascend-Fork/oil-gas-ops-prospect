#!/bin/bash
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

CURRENT_DIR=$(
    cd $(dirname ${BASH_SOURCE:-$0})
    pwd
)
ACLNN_EXE=execute_complex_mul_grad_op
ACLNN_VERIFY_ARGS="output/output_gur.bin output/golden_gur.bin output/output_gui.bin output/golden_gui.bin output/output_gvr.bin output/golden_gvr.bin output/output_gvi.bin output/golden_gvi.bin"
# shellcheck source=../../_common/aclnn/run_aclnn.sh
source "${CURRENT_DIR}/../../_common/aclnn/run_aclnn.sh"
