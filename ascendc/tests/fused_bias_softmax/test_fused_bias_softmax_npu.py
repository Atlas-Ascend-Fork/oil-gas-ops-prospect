#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""FusedBiasSoftmaxOg / FusedSoftmaxGradOg 精度与性能微基准。

语义（均为原地写回）：
    forward:  y = softmax(x + bias1 + bias2, -1)
    backward: g = go @ V^T; probs ← probs ⊙ (g − rowsum(probs⊙g))

probs 分布用 softmax 输出（openfold 生产输入形态）：backward MIX 路径的
g(half) 量化误差经 rowsum(p·g) 的放大与 Σp² 成正比，softmax 分布（Σp²≤1）
是该误差上界成立的前提。
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "../../.."))
sys.path[:0] = [
    THIS_DIR,                           # 就地编译出的 fused_bias_softmax_npu 扩展
    os.path.join(REPO_ROOT, "python"),  # 仓内 oil_gas_ops_prospect 包（未装 whl 时也能用）
]


def build_ext() -> None:
    """就地编译 fused_attention_npu pybind 扩展（前置条件：已安装 OPP 包）。"""
    import subprocess

    subprocess.run(
        [sys.executable, "setup.py", "build_ext", "--inplace"],
        cwd=THIS_DIR,
        check=True,
    )


def softmax_probs(*shape, dtype, temp=1.0):
    return torch.softmax(temp * torch.randn(*shape, device="npu"), -1).to(dtype)


def torch_forward(x, b1, b2):
    logits = x.float().clone()
    if b1 is not None:
        logits = logits + b1.float()
    if b2 is not None:
        logits = logits + b2.float()
    return torch.softmax(logits, -1).to(x.dtype)


def torch_backward(probs, go, v):
    g = torch.matmul(go.float(), v.float().transpose(-1, -2))
    return (probs.float() * (g - (g * probs.float()).sum(-1, keepdim=True))).to(probs.dtype)


def max_diff(ref, out):
    return (ref.float() - out.float()).abs().max().item()


def over_tol(ref, out, atol, rtol):
    if ref.shape != out.shape:
        print(f"  [SHAPE MISMATCH] ref={tuple(ref.shape)} out={tuple(out.shape)}")
    err = (ref.float() - out.float()).abs()
    cap = atol + rtol * ref.float().abs()
    return int((err > cap).sum().item())


def bench(fn, n_warm=5, n_iter=20):
    for _ in range(n_warm):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / n_iter * 1e3


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--build-ext", action="store_true", help="先就地编译 fused_bias_softmax_npu 扩展")
    return p.parse_args(argv)


def run_forward_cases(ext, fails):
    """forward 精度用例。用例字段：名称 / 形状 / dtype / 温度 / bias 开关 / 容差。"""
    fwd_cases = [
        ("bf16-dual-bcast [2,4,64,128]", (2, 4, 64, 128), torch.bfloat16, 1.0, True, True, 2e-2, 4e-3),
        ("bf16-dual [2,4,256,256]", (2, 4, 256, 256), torch.bfloat16, 1.0, True, True, 2e-2, 4e-3),
        ("fp32-none [2,4,64,64]", (2, 4, 64, 64), torch.float32, 1.0, False, False, 1e-5, 0.0),
        ("bf16-peaked [2,4,128,128]", (2, 4, 128, 128), torch.bfloat16, 4.0, True, True, 2e-2, 4e-3),
    ]
    print("== forward ==")
    for name, shape, dtype, temp, has_b1, has_b2, atol, rtol in fwd_cases:
        x = torch.randn(*shape, dtype=dtype, device="npu")
        b1 = torch.randn(*shape[:-2], 1, shape[-1], dtype=dtype, device="npu") if has_b1 else None
        b2 = torch.randn(*shape, dtype=dtype, device="npu") if has_b2 else None
        ref = torch_forward(x, b1, b2)
        out = ext.fused_bias_softmax_forward(x, b1, b2)
        torch.npu.synchronize()
        inplace = out.data_ptr() == x.data_ptr()
        over = over_tol(ref, out, atol, rtol)
        ok = over == 0 and inplace
        fails += not ok
        print(f"{name}: max_diff={max_diff(ref, out):.2e} over={over} inplace={inplace} "
              f"{'PASS' if ok else 'FAIL'}")
    return fails


def run_backward_cases(ext, fails):
    """backward 精度用例。用例字段：名称 / B,H,Q,S,D / dtype / 温度 / 容差。"""
    bwd_cases = [
        ("bf16-MIX [2,4,64,64] D64", 2, 4, 64, 64, 64, torch.bfloat16, 1.0, 2e-2, 4e-3),
        ("bf16-MIX [8,8,256,256] D64 超核", 8, 8, 256, 256, 64, torch.bfloat16, 1.0, 2e-2, 4e-3),
        ("bf16-VEC [2,2,128,128] D20 非对齐", 2, 2, 128, 128, 20, torch.bfloat16, 1.0, 2e-2, 4e-3),
        ("fp32-VEC [2,4,64,64] D64", 2, 4, 64, 64, 64, torch.float32, 1.0, 1e-5, 0.0),
        ("bf16-MIX Q!=S [2,4,64,128] D64", 2, 4, 64, 128, 64, torch.bfloat16, 1.0, 2e-2, 4e-3),
    ]
    print("== backward ==")
    for name, b, h, q, s_dim, d_dim, dtype, temp, atol, rtol in bwd_cases:
        probs = softmax_probs(b, h, q, s_dim, dtype=dtype, temp=temp)
        go = torch.randn(b, h, q, d_dim, dtype=dtype, device="npu")
        v = torch.randn(b, h, s_dim, d_dim, dtype=dtype, device="npu")
        ref = torch_backward(probs, go, v)
        work = probs.clone()
        out = ext.fused_softmax_grad_backward(work, go.clone(), v.clone())
        torch.npu.synchronize()
        inplace = out.data_ptr() == work.data_ptr()
        over = over_tol(ref, out, atol, rtol)
        ok = over == 0 and inplace
        fails += not ok
        print(f"{name}: max_diff={max_diff(ref, out):.2e} over={over} inplace={inplace} "
              f"{'PASS' if ok else 'FAIL'}")
    return fails


def run_e2e_cases(ext, fails):
    """openfold attention_core 使用形态链路（前向融合 + 反向融合手动反传）。"""
    print("== e2e ==")
    b, h, q, s, d = 2, 4, 64, 128, 32
    q_t = torch.randn(b, h, q, d, dtype=torch.bfloat16, device="npu")
    k_t = torch.randn(b, h, s, d, dtype=torch.bfloat16, device="npu")
    v_t = torch.randn(b, h, s, d, dtype=torch.bfloat16, device="npu")
    b1 = torch.randn(b, h, 1, s, dtype=torch.bfloat16, device="npu")
    b2 = torch.randn(b, h, q, s, dtype=torch.bfloat16, device="npu")
    g = torch.randn(b, h, q, d, dtype=torch.bfloat16, device="npu")

    q1, k1, v1 = q_t.clone(), k_t.clone(), v_t.clone()
    logits = torch.matmul(q1, k1.transpose(-1, -2))
    logits = ext.fused_bias_softmax_forward(logits, b1, b2)
    attn_out = torch.matmul(logits, v1)
    grad_v = torch.matmul(logits.transpose(-1, -2), g)
    g_logits = ext.fused_softmax_grad_backward(logits, g.contiguous(), v1)
    grads_fused = [attn_out.detach(), g_logits, torch.matmul(g_logits, k1), grad_v]

    q2, k2, v2 = q_t.float(), k_t.float(), v_t.float()
    probs2 = torch.softmax(torch.matmul(q2, k2.transpose(-1, -2)) + b1.float() + b2.float(), -1)
    ref_out = torch.matmul(probs2, v2)
    g2 = g.float()
    g_logits2 = probs2 * (torch.matmul(g2, v2.transpose(-1, -2))
                          - (torch.matmul(g2, v2.transpose(-1, -2)) * probs2).sum(-1, keepdim=True))
    grads_ref = [ref_out, g_logits2, torch.matmul(g_logits2, k2),
                 torch.matmul(probs2.transpose(-1, -2), g2)]

    names = ["o", "dL/dlogits", "dL/dq", "dL/dv"]
    for name, got, want in zip(names, grads_fused, grads_ref):
        d_max = max_diff(want, got)
        ok = d_max < 0.6
        fails += not ok
        print(f"{name}: max_diff={d_max:.2e} {'PASS' if ok else 'FAIL'}")
    return fails


def run_perf_bench(ext):
    """性能微基准。"""
    print("== perf (us/call, perf_counter, 20 iter) ==")
    probs = softmax_probs(8, 8, 256, 256, dtype=torch.bfloat16)
    go = torch.randn(8, 8, 256, 64, dtype=torch.bfloat16, device="npu")
    v = torch.randn(8, 8, 256, 64, dtype=torch.bfloat16, device="npu")
    t_fwd = bench(lambda: ext.fused_bias_softmax_forward(probs, None, None))
    t_bwd = bench(lambda: ext.fused_softmax_grad_backward(probs, go, v))
    t_ref_bwd = bench(lambda: torch_backward(probs, go, v))
    print(f"forward [8,8,256,256]: fused={t_fwd:.1f}us")
    print(f"backward [8,8,256,256]: fused={t_bwd:.1f}us vs torch-decomposed={t_ref_bwd:.1f}us "
          f"({t_ref_bwd / t_bwd:.2f}x)")


def main(argv=None):
    args = parse_args(argv)
    if args.build_ext:
        build_ext()

    import fused_bias_softmax_npu as ext

    fails = 0
    fails = run_forward_cases(ext, fails)
    fails = run_backward_cases(ext, fails)
    fails = run_e2e_cases(ext, fails)
    run_perf_bench(ext)

    print("ALL PASS" if fails == 0 else f"{fails} FAILS")
    return 0 if fails == 0 else 1



if __name__ == "__main__":
    sys.exit(main())
