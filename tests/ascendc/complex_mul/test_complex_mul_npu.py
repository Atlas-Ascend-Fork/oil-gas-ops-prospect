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

"""ComplexMul / ComplexMulGrad 精度与性能微基准。"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time

import _aclnn_cache  # noqa: F401
import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# 仓库根：tests/ascendc/complex_mul -> 上三层
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
    """就地编译 complex_mul_npu pybind 扩展（前置条件：已安装 OPP 包）。"""
    import subprocess

    subprocess.run(
        [sys.executable, os.path.join(REPO_ROOT, "ascendc", "pybind_setup.py"), "build_ext", "--inplace"],
        cwd=PYBIND_DIR,
        check=True,
    )


def rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    a64 = a.detach().float().cpu()
    b64 = b.detach().float().cpu()
    num = torch.linalg.norm((a64 - b64).reshape(-1))
    den = torch.linalg.norm(a64.reshape(-1)).clamp_min(1e-30)
    return float(num / den)


def torch_fwd(ur, ui, vr, vi):
    return ur * vr - ui * vi, ur * vi + ui * vr


def torch_bwd(gwr, gwi, ur, ui, vr, vi):
    gur = gwr * vr + gwi * vi
    gui = -gwr * vi + gwi * vr
    gvr = gwr * ur + gwi * ui
    gvi = -gwr * ui + gwi * ur
    return gur, gui, gvr, gvi


def bench(fn, n_warm=5, n_iter=20):
    for _ in range(n_warm):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / n_iter * 1e3


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shape", type=int, nargs="+", default=[1, 1, 192, 320, 320])
    p.add_argument("--rtol", type=float, default=1e-5)
    p.add_argument("--build-ext", action="store_true", help="先就地编译 complex_mul_npu 扩展")
    return p.parse_args(argv)


def load_extension():
    """导入 NPU 扩展；缺少 NPU 环境或扩展时返回 None（调用方跳过）。"""
    try:
        import torch_npu  # noqa: F401
    except ImportError:
        print("[SKIP] torch_npu 不可用，本测试需要 NPU 环境")
        return None
    if not (hasattr(torch, "npu") and torch.npu.is_available()):
        print("[SKIP] 未检测到 NPU 设备；精度/性能对比必须在真实 910B 上运行")
        return None
    try:
        import complex_mul_npu as ext
    except ImportError as exc:
        print(f"[FAIL] 无法导入 complex_mul_npu 扩展（先跑 --build-ext）: {exc}")
        return None
    return ext


def make_inputs(shape: tuple[int, ...], device: torch.device):
    """(ur, ui, vr, vi, gwr, gwi)：全部同 shape 的 float32 随机张量。"""
    ur = torch.randn(*shape, device=device, dtype=torch.float32)
    return (ur, *(torch.randn_like(ur) for _ in range(5)))


def check_accuracy(ext, tensors, shape) -> list[float]:
    """自定义算子 vs torch 闭式解的相对误差（前向 2 项 + 反向 4 项）。"""
    ur, ui, vr, vi, gwr, gwi = tensors

    # warmup custom
    for _ in range(2):
        ext.complex_mul_forward(ur, ui, vr, vi)
        ext.complex_mul_backward(gwr, gwi, ur, ui, vr, vi)
    torch.npu.synchronize()

    wr_t, wi_t = torch_fwd(ur, ui, vr, vi)
    wr_c, wi_c = ext.complex_mul_forward(ur, ui, vr, vi)
    torch.npu.synchronize()
    e_wr = rel_err(wr_t, wr_c)
    e_wi = rel_err(wi_t, wi_c)
    print(f"[fwd] shape={shape} rel_err wr={e_wr:.3e} wi={e_wi:.3e}")

    ref_bwd = torch_bwd(gwr, gwi, ur, ui, vr, vi)
    got_bwd = ext.complex_mul_backward(gwr, gwi, ur, ui, vr, vi)
    torch.npu.synchronize()
    errs = {
        name: rel_err(ref, got)
        for name, ref, got in zip(("gur", "gui", "gvr", "gvi"), ref_bwd, got_bwd)
    }
    print("[bwd] rel_err " + " ".join(f"{k}={v:.3e}" for k, v in errs.items()))
    return [e_wr, e_wi, *errs.values()]


def report_perf(ext, tensors) -> None:
    ur, ui, vr, vi, gwr, gwi = tensors
    ms_t = bench(lambda: torch_fwd(ur, ui, vr, vi))
    ms_c = bench(lambda: ext.complex_mul_forward(ur, ui, vr, vi))
    ms_tb = bench(lambda: torch_bwd(gwr, gwi, ur, ui, vr, vi))
    ms_cb = bench(lambda: ext.complex_mul_backward(gwr, gwi, ur, ui, vr, vi))
    print(f"[perf] fwd torch={ms_t:.3f}ms ascendc={ms_c:.3f}ms speedup={ms_t/ms_c:.3f}x")
    print(f"[perf] bwd torch={ms_tb:.3f}ms ascendc={ms_cb:.3f}ms speedup={ms_tb/ms_cb:.3f}x")


def check_autograd(tensors) -> None:
    """经 autograd 封装再对一遍，验证 Function 的接线（强制走自定义算子）。"""
    os.environ["OIL_GAS_OPS_COMPLEX_MUL_IMPL"] = "ascendc"
    os.environ["OIL_GAS_OPS_REQUIRE_ASCENDC_COMPLEX_MUL"] = "1"

    from oil_gas_ops_prospect.complex_mul import complex_mul

    ur, ui, vr, vi, gwr, gwi = tensors
    ref_bwd = torch_bwd(gwr, gwi, ur, ui, vr, vi)
    leaves = [t.detach().requires_grad_(True) for t in (ur, ui, vr, vi)]
    wr2, wi2 = complex_mul(*leaves)
    (wr2 * gwr + wi2 * gwi).sum().backward()
    parts = " ".join(
        f"{name}={rel_err(ref, leaf.grad):.3e}"
        for name, ref, leaf in zip(("gur", "gui", "gvr", "gvi"), ref_bwd, leaves)
    )
    print(f"[autograd] {parts}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.build_ext:
        build_ext()

    ext = load_extension()
    if ext is None:
        return 1

    device = torch.device("npu:0")
    torch.npu.set_device(0)
    shape = tuple(args.shape)
    tensors = make_inputs(shape, device)

    errs = check_accuracy(ext, tensors, shape)
    report_perf(ext, tensors)
    check_autograd(tensors)

    if max(errs) > args.rtol:
        print(f"FAIL max_rel_err={max(errs):.3e} > rtol={args.rtol}")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
