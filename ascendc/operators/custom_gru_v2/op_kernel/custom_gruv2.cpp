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

#include "kernel_operator.h"
using namespace AscendC;

extern "C" __global__ __aicore__ void custom_gruv2(GM_ADDR x, GM_ADDR weight_ih, GM_ADDR weight_hh, GM_ADDR bias_ih,
                                                   GM_ADDR bias_hh, GM_ADDR y, GM_ADDR output_h, GM_ADDR workspace,
                                                   GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    uint32_t B = tiling_data.batchSize;
    uint32_t T = tiling_data.seqLen;
    uint32_t F_in = tiling_data.inputSize;
    uint32_t H = tiling_data.hiddenSize;
    uint32_t hasBias = tiling_data.hasBias;
    uint32_t bidirectional = tiling_data.bidirectional;
    uint32_t dirs = bidirectional ? 2 : 1;
    uint32_t Ha = (H + 7) & ~7;

    uint32_t blockNum = GetBlockNum();
    uint32_t blockIdx = GetBlockIdx();
    uint32_t batchPerBlock = (B + blockNum - 1) / blockNum;
    if (batchPerBlock == 0)
        batchPerBlock = 1;
    uint32_t bStart = blockIdx * batchPerBlock;
    if (bStart >= B)
        return;
    uint32_t bEnd = (bStart + batchPerBlock > B) ? B : (bStart + batchPerBlock);
    uint32_t localB = bEnd - bStart;

    GlobalTensor<float> xGm;
    xGm.SetGlobalBuffer((__gm__ float*)x, B * T * F_in);
    GlobalTensor<float> wiGm;
    wiGm.SetGlobalBuffer((__gm__ float*)weight_ih, dirs * 3 * H * F_in);
    GlobalTensor<float> whGm;
    whGm.SetGlobalBuffer((__gm__ float*)weight_hh, dirs * 3 * H * H);
    GlobalTensor<float> yGm;
    yGm.SetGlobalBuffer((__gm__ float*)y, dirs * B * T * H);
    GlobalTensor<float> ohGm;
    ohGm.SetGlobalBuffer((__gm__ float*)output_h, dirs * B * H);

    GlobalTensor<float> biGm;
    GlobalTensor<float> bhGm;
    if (hasBias) {
        biGm.SetGlobalBuffer((__gm__ float*)bias_ih, dirs * 3 * H);
        bhGm.SetGlobalBuffer((__gm__ float*)bias_hh, dirs * 3 * H);
    }

    TPipe pipe;
    TBuf<TPosition::VECCALC> hBuf, gBuf, rBuf, zBuf, t1Buf, t2Buf, oneBuf;
    pipe.InitBuffer(hBuf, localB * Ha * sizeof(float));
    pipe.InitBuffer(gBuf, 3 * Ha * sizeof(float));
    pipe.InitBuffer(rBuf, Ha * sizeof(float));
    pipe.InitBuffer(zBuf, Ha * sizeof(float));
    pipe.InitBuffer(t1Buf, Ha * sizeof(float));
    pipe.InitBuffer(t2Buf, Ha * sizeof(float));
    pipe.InitBuffer(oneBuf, Ha * sizeof(float));

    LocalTensor<float> h = hBuf.Get<float>();
    LocalTensor<float> gates = gBuf.Get<float>();
    LocalTensor<float> r = rBuf.Get<float>();
    LocalTensor<float> z = zBuf.Get<float>();
    LocalTensor<float> t1 = t1Buf.Get<float>();
    LocalTensor<float> t2 = t2Buf.Get<float>();
    LocalTensor<float> one = oneBuf.Get<float>();
    // Div(float) 的分子常量 1.0；一次性初始化（评审侧正式 sigmoid
    // 实现 AscendC highlevel sigmoid_impl.h 同样使用 Div 而非 Reciprocal：
    // Reciprocal 矢量指令精度差（每步 ~1e-3），逐步累积造成 D2 的
    // 0.0137 累积偏差（fp64 真值锚定）。
    Duplicate<float>(one, 1.0f, Ha);

    DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = H * sizeof(float);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;

    for (uint32_t dir = 0; dir < dirs; dir++) {
        Duplicate<float>(h, 0.0f, localB * Ha);
        // Sync: Duplicate (vector) writes to h must complete before scalar reads h(bi*Ha+k)
        PipeBarrier<PIPE_V>();
        // Cross-pipe sync (V -> S/M): scalar reads of h must observe the vector-written
        // h buffer. PipeBarrier<PIPE_V> only orders within the vector pipe, it does
        // NOT make the vector writes visible to scalar loads. Without this pair the
        // first gate computation of the direction can read stale h (defect D2 in
        // remote test report).
        // V -> S: scalar reads of h must observe the vector-initialized h
        // (event naming per repo-verified fused_softmax_grad_impl.h pattern).
        event_t syncVSInit = static_cast<event_t>(EVENT_ID0);
        SetFlag<HardEvent::V_S>(syncVSInit);
        WaitFlag<HardEvent::V_S>(syncVSInit);
        uint32_t wiOff = dir * 3 * H * F_in;
        uint32_t whOff = dir * 3 * H * H;
        uint32_t biOff = dir * 3 * H;
        uint32_t bhOff = dir * 3 * H;

        // NOTE: weight_ih, weight_hh, x, bias are read from GM via scalar access.
        // This is the primary performance bottleneck (aiv_scalar_ratio=99.3%).
        // Optimization path: DataCopy weight/x tiles to UB before the GEMV loop,
        // then use vector Dot/Matmul. Deferred to next iteration — correctness first.

        for (int32_t t = (dir == 0 ? 0 : (int32_t)T - 1); (dir == 0 ? t < (int32_t)T : t >= 0);
             t += (dir == 0 ? 1 : -1)) {
            for (uint32_t bi = 0; bi < localB; bi++) {
                uint32_t b = bStart + bi;
                for (uint32_t d = 0; d < H; d++) {
                    float s = 0.0f;
                    if (hasBias)
                        s = biGm(biOff + d) + bhGm(bhOff + d);
                    for (uint32_t k = 0; k < F_in; k++)
                        s += wiGm(wiOff + d * F_in + k) * xGm(b * T * F_in + t * F_in + k);
                    for (uint32_t k = 0; k < H; k++)
                        s += whGm(whOff + d * H + k) * h(bi * Ha + k);
                    gates(d) = s;
                }
                for (uint32_t d = 0; d < H; d++) {
                    float s = 0.0f;
                    if (hasBias)
                        s = biGm(biOff + H + d) + bhGm(bhOff + H + d);
                    for (uint32_t k = 0; k < F_in; k++)
                        s += wiGm(wiOff + (H + d) * F_in + k) * xGm(b * T * F_in + t * F_in + k);
                    for (uint32_t k = 0; k < H; k++)
                        s += whGm(whOff + (H + d) * H + k) * h(bi * Ha + k);
                    gates(Ha + d) = s;
                }
                for (uint32_t d = 0; d < H; d++) {
                    float an = 0.0f;
                    if (hasBias)
                        an = biGm(biOff + 2 * H + d);
                    for (uint32_t k = 0; k < F_in; k++)
                        an += wiGm(wiOff + (2 * H + d) * F_in + k) * xGm(b * T * F_in + t * F_in + k);
                    gates(2 * Ha + d) = an;
                    float ghn = 0.0f;
                    if (hasBias)
                        ghn = bhGm(bhOff + 2 * H + d);
                    for (uint32_t k = 0; k < H; k++)
                        ghn += whGm(whOff + (2 * H + d) * H + k) * h(bi * Ha + k);
                    t2(d) = ghn;
                }

                for (uint32_t p = H; p < Ha; p++) {
                    gates(p) = 0.0f;
                    gates(Ha + p) = 0.0f;
                    gates(2 * Ha + p) = 0.0f;
                    t2(p) = 0.0f;
                }

                // Sync: scalar writes to gates/t2 must complete before vector reads
                // (S -> V cross-pipe hazard; a barrier inside PIPE_V does not order
                // scalar writes issued on the scalar pipe).
                PipeBarrier<PIPE_V>();
                // S -> V: vector instructions (Muls/Exp/...) must observe this
                // timestep's scalar writes to gates/t2. Earlier TEMP-DIAG-E1
                // diagnostic disabled this pair and broke the t>=1 carry
                // (defect D2); restored per approved plan v4.
                event_t syncSV = static_cast<event_t>(EVENT_ID1);
                SetFlag<HardEvent::S_V>(syncSV);
                WaitFlag<HardEvent::S_V>(syncSV);

                // r = sigmoid(gates): 1 / (1 + exp(-x))，除法用 Div（全精度），
                // 避免矢量 Reciprocal 的低精度近似（D2 数值修复）
                Muls<float>(t1, gates, -1.0f, Ha);
                Exp<float>(r, t1, Ha);
                Adds<float>(r, r, 1.0f, Ha);
                Div<float>(r, one, r, Ha);

                Muls<float>(t1, gates[Ha], -1.0f, Ha);
                Exp<float>(z, t1, Ha);
                Adds<float>(z, z, 1.0f, Ha);
                Div<float>(z, one, z, Ha);

                // n = tanh(x_n)：x_n = r*t2 + gates_n；直接用
                // 2/(1+exp(-2x)) - 1（省一次 Muls，仅一次 Exp + 一次 Div）
                Mul<float>(t1, r, t2, Ha);
                Add<float>(t1, gates[2 * Ha], t1, Ha);
                Muls<float>(t1, t1, -2.0f, Ha);
                Exp<float>(r, t1, Ha);
                Adds<float>(r, r, 1.0f, Ha);
                Div<float>(r, one, r, Ha);
                Muls<float>(r, r, 2.0f, Ha);
                Adds<float>(r, r, -1.0f, Ha);

                Sub<float>(t1, h[bi * Ha], r, Ha);
                Mul<float>(t1, z, t1, Ha);
                Add<float>(h[bi * Ha], r, t1, Ha);
                // Sync: vector write to h must complete before next timestep's scalar reads
                PipeBarrier<PIPE_V>();
            }
            // Sync h (vector-written) -> GM copy (MTE3) -> next-timestep scalar reads (S):
            //   1) V -> MTE3: DataCopyPad must not read h before this timestep's update
            //      retires (otherwise y may capture stale h, corrupting output rows);
            //   2) MTE3 -> V/S: the copy engine must finish reading h before the next
            //      timestep rewrites h (DataCopyPad consumes h asynchronously).
            event_t syncVM3 = static_cast<event_t>(EVENT_ID2);
            SetFlag<HardEvent::V_MTE3>(syncVM3);
            WaitFlag<HardEvent::V_MTE3>(syncVM3);
            for (uint32_t bi = 0; bi < localB; bi++) {
                uint32_t b = bStart + bi;
                DataCopyPad(yGm[dir * B * T * H + b * T * H + t * H], h[bi * Ha], copyParams);
            }
            event_t syncM3V = static_cast<event_t>(EVENT_ID2);
            SetFlag<HardEvent::MTE3_V>(syncM3V);
            WaitFlag<HardEvent::MTE3_V>(syncM3V);
            // V -> S: next timestep's scalar reads of h must observe this
            // timestep's vector update (V_S per fused pattern; was V_M).
            event_t syncVSNext = static_cast<event_t>(EVENT_ID0);
            SetFlag<HardEvent::V_S>(syncVSNext);
            WaitFlag<HardEvent::V_S>(syncVSNext);
        }
        // Final-state copy: same V -> MTE3 -> (end) synchronisation as the y copies.
        event_t syncVM3F = static_cast<event_t>(EVENT_ID2);
        SetFlag<HardEvent::V_MTE3>(syncVM3F);
        WaitFlag<HardEvent::V_MTE3>(syncVM3F);
        for (uint32_t bi = 0; bi < localB; bi++) {
            uint32_t b = bStart + bi;
            DataCopyPad(ohGm[dir * B * H + b * H], h[bi * Ha], copyParams);
        }
        // MTE3 -> V back-edge (approved plan v4, site E): the asynchronous
        // output_h copy reads UB h; the next direction loop resets h with
        // Duplicate before reusing it. Without this pair, dir 0's copy may
        // race dir 1's reset (output_h[0] unstable - review 12.2).
        event_t syncM3VF = static_cast<event_t>(EVENT_ID2);
        SetFlag<HardEvent::MTE3_V>(syncM3VF);
        WaitFlag<HardEvent::MTE3_V>(syncM3VF);
    }
}
