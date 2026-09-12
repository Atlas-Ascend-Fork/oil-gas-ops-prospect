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

"""在 import torch / ACLNN 之前把执行器缓存关掉。

CANN 9 上 aclnn 会缓存输出指针；方形 bf16 等 shape 会空跑。必须在
torch 加载 ACL 运行时之前生效。本模块只有赋值、没有滞后 import，
包根 ``__init__`` 在子模块 ``import torch`` 之前加载本模块，避免
G.FMT.05（导入夹杂赋值）和 G.FMT.07（第三方须先于本地模块）。
"""

import os

os.environ.setdefault("ACLNN_CACHE_LIMIT", "0")
