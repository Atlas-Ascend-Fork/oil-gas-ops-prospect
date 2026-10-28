/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * 给 ops-nn conv3d_v2 的 `#include "log/log.h"` 用。
 * msopgen 自定义算子工程没有 op_common 的 slog 头，这里把 OP_LOG* 收成空宏。
 * 不改 ops-nn 源码。
 */
#ifndef OIL_GAS_OPS_NN_LOG_LOG_H
#define OIL_GAS_OPS_NN_LOG_LOG_H

#ifndef OP_LOGD
#define OP_LOGD(...) \
    do {             \
    } while (0)
#endif
#ifndef OP_LOGI
#define OP_LOGI(...) \
    do {             \
    } while (0)
#endif
#ifndef OP_LOGW
#define OP_LOGW(...) \
    do {             \
    } while (0)
#endif
#ifndef OP_LOGE
#define OP_LOGE(...) \
    do {             \
    } while (0)
#endif

#endif // OIL_GAS_OPS_NN_LOG_LOG_H
