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
#
# FusedSoftmaxGrad 专属编译选项。
#
# MIX kernel（KERNEL_TYPE_MIX_AIC_1_2）含手写核间同步（SyncAll + MTE2_V/V_MTE3/
# MTE3_V/S_V 事件）。CANN 9.x op_build 工具链对 dav-c220 默认注入 --cce-auto-sync，
# 其屏障自动推断与手写同步冲突，产生非确定错值（barrier-inference gap，
# 华为 flash-attention-grad 同样关闭）。此处显式关闭。
if(COMMAND add_ops_compile_options)
    add_ops_compile_options(
        OP_NAME FusedSoftmaxGrad
        OPTIONS --cce-auto-sync=off
    )
endif()
