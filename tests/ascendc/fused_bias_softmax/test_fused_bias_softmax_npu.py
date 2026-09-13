#!/usr/bin/env python3
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

"""FusedBiasSoftmax / FusedSoftmaxGrad 精度与性能微基准。

语义（均为原地写回）：
    forward:  y = softmax(x + bias1 + bias2, -1)
    backward: g = go @ V^T; probs ← probs ⊙ (g − rowsum(probs⊙g))

probs 分布用 softmax 输出（openfold 生产输入形态）：backward MIX 路径的
g(half) 量化误差经 rowsum(p·g) 的放大与 Σp² 成正比，softmax 分布（Σp²≤1）
是该误差上界成立的前提。
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time

import _aclnn_cache  # noqa: F401
import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# 仓库根：tests/ascendc/fused_bias_softmax -> 上三层
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "../../.."))
# pybind 封装是算子交付件的一部分，就地编译产物落在 ascendc/pybind/。
PYBIND_DIR = os.path.join(REPO_ROOT, "ascendc", "pybind")
sys.path.insert(0, PYBIND_DIR)

_spec = importlib.util.spec_from_file_location("local_pkg", os.path.join(REPO_ROOT, "scripts", "local_pkg.py"))
if _spec is not None and _spec.loader is not None:
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    _mod.activate(REPO_ROOT)


def build_ext() -> None:
    """就地编译 fused_bias_softmax_npu pybind 扩展（前置条件：已安装 OPP 包）。"""
    import subprocess

    subprocess.run(
        [sys.executable, os.path.join(REPO_ROOT, "ascendc", "pybind_setup.py"), "build_ext", "--inplace"],
        cwd=PYBIND_DIR,
        check=True,
    )


def _fresh(t):
    """新分配一块连续缓冲，避开 aclnn 执行器缓存把结果写到旧指针。"""
    if t is None:
        return None
    out = torch.empty_like(t)
    out.copy_(t)
    return out.contiguous()


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
        x, b1, b2 = _fresh(x), _fresh(b1), _fresh(b2)
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
        work, go, v = _fresh(probs), _fresh(go), _fresh(v)
        out = ext.fused_softmax_grad_backward(work, go, v)
        torch.npu.synchronize()
        inplace = out.data_ptr() == work.data_ptr()
        over = over_tol(ref, out, atol, rtol)
        ok = over == 0 and inplace
        fails += not ok
        print(f"{name}: max_diff={max_diff(ref, out):.2e} over={over} inplace={inplace} "
              f"{'PASS' if ok else 'FAIL'}")
    return fails


def run_alias_isolation(ext, fails):
    """同 shape 不同缓冲：后一次不得写到前一块，也不得复用前一块的数值。"""
    print("== alias isolation ==")
    cases = [
        ((2, 4, 64, 64), torch.float32, False, 1e-5, 0.0),
        ((2, 4, 64, 128), torch.bfloat16, True, 2e-2, 4e-3),
    ]
    for shape, dtype, has_b, atol, rtol in cases:
        a = torch.randn(*shape, dtype=dtype, device="npu")
        b = torch.randn(*shape, dtype=dtype, device="npu")
        a1 = torch.randn(*shape[:-2], 1, shape[-1], dtype=dtype, device="npu") if has_b else None
        a2 = torch.randn(*shape, dtype=dtype, device="npu") if has_b else None
        b1 = torch.randn(*shape[:-2], 1, shape[-1], dtype=dtype, device="npu") if has_b else None
        b2 = torch.randn(*shape, dtype=dtype, device="npu") if has_b else None
        ref_a = torch_forward(a, a1, a2)
        ref_b = torch_forward(b, b1, b2)
        snap_a = a.clone()
        out_a = ext.fused_bias_softmax_forward(a, a1, a2)
        torch.npu.synchronize()
        got_a = out_a.clone()
        out_b = ext.fused_bias_softmax_forward(b, b1, b2)
        torch.npu.synchronize()
        clobber = max_diff(got_a, a)
        untouched_input = max_diff(snap_a, a)  # a 是原地，应已变成 softmax(a)
        over_a = over_tol(ref_a, got_a, atol, rtol)
        over_b = over_tol(ref_b, out_b, atol, rtol)
        # 第二跳之后第一块仍是自己的结果，不是第二块、也不是原始输入
        stay_a = over_tol(ref_a, a, atol, rtol)
        leak_b_into_a = max_diff(out_b, a)
        ok = over_a == 0 and over_b == 0 and stay_a == 0 and clobber == 0 and leak_b_into_a > atol
        fails += not ok
        tag = f"{list(shape)} {str(dtype).replace('torch.', '')}"
        print(f"{tag}: over_a={over_a} over_b={over_b} stay_a={stay_a} "
              f"ptr_a={a.data_ptr()} ptr_b={b.data_ptr()} {'PASS' if ok else 'FAIL'}")
        if not ok:
            print(f"  clobber={clobber:.3e} leak_b_into_a={leak_b_into_a:.3e} "
                  f"untouched_input={untouched_input:.3e}")
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
    logits = _fresh(torch.matmul(q1, k1.transpose(-1, -2)).float().to(torch.bfloat16))
    logits = ext.fused_bias_softmax_forward(logits, _fresh(b1), _fresh(b2))
    attn_out = torch.matmul(logits, v1)
    grad_v = torch.matmul(logits.transpose(-1, -2), g)
    g_logits = ext.fused_softmax_grad_backward(_fresh(logits), _fresh(g), _fresh(v1))
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

    try:
        torch.npu.config.allow_internal_format = False
    except Exception:
        pass

    fails = 0
    fails = run_forward_cases(ext, fails)
    fails = run_alias_isolation(ext, fails)
    fails = run_backward_cases(ext, fails)
    fails = run_e2e_cases(ext, fails)
    run_perf_bench(ext)

    print("ALL PASS" if fails == 0 else f"{fails} FAILS")
    return 0 if fails == 0 else 1



if __name__ == "__main__":
    sys.exit(main())
