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
 * 本仓自定义算子的 aclnn 两段式接口声明，供 examples/ 与 tests/st/ 共用。
 *
 * 装了算子包时优先用 op_api/include 下生成的 aclnn_*.h；没装时回落到手写声明，
 * 这样源码树里的 examples / ST 工程仍可独立编译。声明按算子逐个用 __has_include
 * 判定，只装了部分算子也能正常编译。
 */

#ifndef OG_ACLNN_DECLS_H
#define OG_ACLNN_DECLS_H

#include "acl/acl.h"
#include "aclnn/acl_meta.h"

#if __has_include("aclnn_complex_mul.h")
#include "aclnn_complex_mul.h"
#else
extern "C" {
aclnnStatus aclnnComplexMulGetWorkspaceSize(const aclTensor* ur, const aclTensor* ui, const aclTensor* vr,
                                            const aclTensor* vi, const aclTensor* wrOut, const aclTensor* wiOut,
                                            uint64_t* workspaceSize, aclOpExecutor** executor);
aclnnStatus aclnnComplexMul(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);
}
#endif

#if __has_include("aclnn_complex_mul_grad.h")
#include "aclnn_complex_mul_grad.h"
#else
extern "C" {
aclnnStatus aclnnComplexMulGradGetWorkspaceSize(const aclTensor* gwr, const aclTensor* gwi, const aclTensor* ur,
                                                const aclTensor* ui, const aclTensor* vr, const aclTensor* vi,
                                                const aclTensor* gurOut, const aclTensor* guiOut,
                                                const aclTensor* gvrOut, const aclTensor* gviOut,
                                                uint64_t* workspaceSize, aclOpExecutor** executor);
aclnnStatus aclnnComplexMulGrad(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);
}
#endif

#if __has_include("aclnn_fused_bias_softmax.h")
#include "aclnn_fused_bias_softmax.h"
#else
extern "C" {
aclnnStatus aclnnFusedBiasSoftmaxGetWorkspaceSize(const aclTensor* x, const aclTensor* bias1, const aclTensor* bias2,
                                                  const aclTensor* y, uint64_t* workspaceSize,
                                                  aclOpExecutor** executor);
aclnnStatus aclnnFusedBiasSoftmax(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);
}
#endif

#if __has_include("aclnn_fused_softmax_grad.h")
#include "aclnn_fused_softmax_grad.h"
#else
extern "C" {
aclnnStatus aclnnFusedSoftmaxGradGetWorkspaceSize(const aclTensor* probs, const aclTensor* gradOutput,
                                                  const aclTensor* values, const aclTensor* out,
                                                  uint64_t* workspaceSize, aclOpExecutor** executor);
aclnnStatus aclnnFusedSoftmaxGrad(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);
}
#endif

#endif // OG_ACLNN_DECLS_H
