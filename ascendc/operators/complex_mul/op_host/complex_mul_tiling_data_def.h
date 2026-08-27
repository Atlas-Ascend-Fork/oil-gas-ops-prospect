/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COMPLEX_MUL_TILING_DATA_DEF_H
#define COMPLEX_MUL_TILING_DATA_DEF_H

#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(ComplexMulTilingDataDef)
    TILING_DATA_FIELD_DEF(int64_t, totalElems);
    TILING_DATA_FIELD_DEF(int64_t, tileFloats);
    TILING_DATA_FIELD_DEF(int64_t, coreNum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ComplexMulOg, ComplexMulTilingDataDef)

} // namespace optiling

#endif // COMPLEX_MUL_TILING_DATA_DEF_H
