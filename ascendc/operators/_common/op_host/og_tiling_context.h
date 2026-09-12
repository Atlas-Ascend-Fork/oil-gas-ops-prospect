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
 * 逐元素算子的公共 host 编排：gert tiling 上下文、同形状输出推导以及
 * float32/ND 端口和 910B 配置声明。
 *
 * 算子文件只保留具名回调、端口顺序和 OpType 注册，避免各算子重复维护 CANN
 * 注册样板。与 og_elementwise_tiling.h 不同，本文件依赖 gert / register / platform
 * 头，只能在 msopgen 工程里编译，不参与 ophost UT。
 */

#ifndef OG_TILING_CONTEXT_H
#define OG_TILING_CONTEXT_H

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <initializer_list>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace OgOpsCommon {

/** shape 的元素总数（空 shape 记 1，与 gert 语义一致）。 */
inline int64_t NumelOf(const gert::Shape& shape)
{
    int64_t n = 1;
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        n *= shape.GetDim(i);
    }
    return n;
}

/** 把 POD tiling 结构写进 gert raw tiling buffer。 */
template <typename TilingT>
inline bool StoreTilingData(gert::TilingData* rawTiling, const TilingT& tilingData)
{
    if (rawTiling == nullptr || rawTiling->GetData() == nullptr) {
        return false;
    }
    if (rawTiling->GetCapacity() < sizeof(TilingT)) {
        return false;
    }
    const auto* src = reinterpret_cast<const unsigned char*>(&tilingData);
    auto* dst = static_cast<unsigned char*>(rawTiling->GetData());
    std::copy_n(src, sizeof(TilingT), dst);
    rawTiling->SetDataSize(sizeof(TilingT));
    return true;
}

/** raw tiling 落盘，并设置 blockDim、tilingKey 和 workspace。 */
template <typename TilingT, typename PlanT>
inline bool FinalizeTilingContext(gert::TilingContext* context, const TilingT& tilingData, const PlanT& plan)
{
    if (context == nullptr) {
        return false;
    }
    if (!StoreTilingData(context->GetRawTilingData(), tilingData)) {
        return false;
    }

    context->SetBlockDim(plan.blockDim);
    context->SetTilingKey(plan.tilingKey);

    size_t* workspaces = context->GetWorkspaceSizes(1);
    if (workspaces == nullptr) {
        return false;
    }
    workspaces[0] = static_cast<size_t>(plan.totalWorkspace);
    return true;
}

/**
 * 逐元素算子的统一 tiling 入口编排。compute 保留各算子的 UB 预算和 tiling key
 * 计算，仅把 platform、shape、落盘等固定流程集中到一处。
 */
template <typename TilingT, typename PlanT>
inline ge::graphStatus RunElementwiseTiling(gert::TilingContext* context, uint32_t shapeInputIndex,
                                            bool (*compute)(int64_t, platform_ascendc::PlatformAscendC&, TilingT&,
                                                            PlanT&))
{
    if (context == nullptr || context->GetPlatformInfo() == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const gert::StorageShape* storage = context->GetInputShape(shapeInputIndex);
    if (storage == nullptr) {
        return ge::GRAPH_FAILED;
    }

    platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
    TilingT tilingData{};
    PlanT plan;
    const int64_t totalElems = NumelOf(storage->GetStorageShape());
    if (!compute(totalElems, platform, tilingData, plan) || !FinalizeTilingContext(context, tilingData, plan)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

/** 将一个输入 shape 原样复制到前 outputCount 个输出。 */
inline ge::graphStatus InferSameShapeOutputs(gert::InferShapeContext* context, uint32_t sourceInputIndex,
                                             uint32_t outputCount)
{
    const gert::Shape* source = context->GetInputShape(sourceInputIndex);
    if (source == nullptr) {
        return ge::GRAPH_FAILED;
    }
    for (uint32_t i = 0; i < outputCount; ++i) {
        if (context->GetOutputShape(i) == nullptr) {
            return ge::GRAPH_FAILED;
        }
    }
    for (uint32_t i = 0; i < outputCount; ++i) {
        *context->GetOutputShape(i) = *source;
    }
    return ge::GRAPH_SUCCESS;
}

/** 将前 outputCount 个输出 dtype 固定为 float32。 */
inline ge::graphStatus InferFloatOutputs(gert::InferDataTypeContext* context, uint32_t outputCount)
{
    for (uint32_t i = 0; i < outputCount; ++i) {
        context->SetOutputDataType(i, ge::DT_FLOAT);
    }
    return ge::GRAPH_SUCCESS;
}

} // namespace OgOpsCommon

namespace ops {

/** 为单个 OpDef 端口设置 REQUIRED / float32 / ND 属性。 */
template <typename PortT>
inline void ConfigureRequiredFloatNdPort(PortT& port)
{
    port.ParamType(REQUIRED);
    port.DataType({ge::DT_FLOAT});
    port.Format({ge::FORMAT_ND});
    port.UnknownShapeFormat({ge::FORMAT_ND});
}

/** 按给定顺序声明逐元素算子的输入和输出 ABI。 */
inline void DeclareRequiredFloatNdIo(OpDef& op, std::initializer_list<const char*> inputs,
                                     std::initializer_list<const char*> outputs)
{
    for (const char* name : inputs) {
        auto&& port = op.Input(name);
        ConfigureRequiredFloatNdPort(port);
    }
    for (const char* name : outputs) {
        auto&& port = op.Output(name);
        ConfigureRequiredFloatNdPort(port);
    }
}

/** 添加两个逐元素算子共享的 Ascend 910B 动态编译配置。 */
inline void AddAscend910bElementwiseConfig(OpDef& op, const char* kernelEntry)
{
    OpAICoreConfig config;
    config.DynamicCompileStaticFlag(true);
    config.DynamicFormatFlag(true);
    config.DynamicRankSupportFlag(true);
    config.DynamicShapeSupportFlag(true);
    config.NeedCheckSupportFlag(false);
    config.PrecisionReduceFlag(false);

    struct ExtendedSetting {
        const char* key;
        const char* value;
    };
    const ExtendedSetting settings[] = {
        {"opFile.value", kernelEntry},
        {"opInterface.value", kernelEntry},
        {"aclnnSupport.value", "support_aclnn"},
        {"jitCompile.flag", "false"},
    };
    for (const auto& setting : settings) {
        config.ExtendCfgInfo(setting.key, setting.value);
    }
    op.AICore().AddConfig("ascend910b", config);
}

} // namespace ops

#endif // OG_TILING_CONTEXT_H
