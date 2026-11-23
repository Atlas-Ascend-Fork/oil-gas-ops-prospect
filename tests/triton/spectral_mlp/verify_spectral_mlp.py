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

"""spectral_mlp 的精度与性能验证（NPU）。

精度：以 CPU float64 上的 einsum 参考为真值，逐条比较前向输出与全部 14 个
输入/参数的梯度。
性能：同设备 fp32 的 einsum + ``npu_gelu``（模型原实现的激活）作为基线，
比较前向与「前向 + 反向（loss 为输出求和）」耗时。基线不使用逐元素拼出的
GELU 近似，避免把基线实现慢算成算子收益。

用法::

    python3 tests/triton/spectral_mlp/verify_spectral_mlp.py --mode all
    python3 tests/triton/spectral_mlp/verify_spectral_mlp.py --mode precision
    python3 tests/triton/spectral_mlp/verify_spectral_mlp.py --mode perf --img 128
    python3 tests/triton/spectral_mlp/verify_spectral_mlp.py --json report/results.json

``--mode precision`` / ``--mode perf`` 与 ``tests/suites.json`` 的 ``npu_python``
协议一致；精度不达标时进程返回非 0。
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import torch

import _bootstrap  # noqa: F401  (side effect: 源码树里解析 oil_gas_ops_prospect)
from oil_gas_ops_prospect.spectral_mlp import spectral_mlp

GELU_C0 = 1.5957691
GELU_C1 = 0.044715


def _gelu(x: torch.Tensor) -> torch.Tensor:
    return x / (1.0 + torch.exp(-GELU_C0 * (x + GELU_C1 * x * x * x)))


def _reference(xr, xi, up1, down1, gate1r, gate1i, bias1r, bias1i,
               up2, down2, gate2r, gate2i, bias2r, bias2i, act=_gelu):
    """einsum 参考实现（与算子内部实现独立，便于作为真值）。

    精度对拍用默认的 ``_gelu``（与 kernel/回退同式）；性能基线改用
    ``_model_gelu``，即模型原本使用的融合 GELU，避免把"基线里的逐元素
    激活实现慢"算成算子收益。
    """
    B, L = int(xr.shape[0]), int(xr.shape[1])
    spatial = tuple(int(v) for v in xr.shape[2:-2])
    G, D = int(xr.shape[-2]), int(xr.shape[-1])
    S = 1
    for v in spatial:
        S *= v
    R = int(up1.shape[2])
    DM = int(down1.shape[1])

    xr2 = xr.reshape(B, L, S, G, D)
    xi2 = xi.reshape(B, L, S, G, D)
    x1r = torch.einsum("blsgf,gfr->blsgr", xr2, up1)
    x1i = torch.einsum("blsgf,gfr->blsgr", xi2, up1)
    s1 = x1r * gate1r.view(1, L, 1, G, R) + x1i * gate1i.view(1, L, 1, G, R)
    d1 = x1r * gate1r.view(1, L, 1, G, R) - x1i * gate1i.view(1, L, 1, G, R)
    h1r = torch.einsum("blsgr,gdr->blsgd", d1, down1) + bias1r.view(1, 1, 1, G, DM)
    h1i = torch.einsum("blsgr,gdr->blsgd", s1, down1) + bias1i.view(1, 1, 1, G, DM)
    o1r, o1i = act(h1r), act(h1i)

    x2r = torch.einsum("blsgf,gfr->blsgr", o1r, up2)
    x2i = torch.einsum("blsgf,gfr->blsgr", o1i, up2)
    s2 = x2r * gate2r.view(1, L, 1, G, R) + x2i * gate2i.view(1, L, 1, G, R)
    d2 = x2r * gate2r.view(1, L, 1, G, R) - x2i * gate2i.view(1, L, 1, G, R)
    yr = torch.einsum("blsgr,gdr->blsgd", d2, down2) + bias2r.view(1, 1, 1, G, D)
    yi = torch.einsum("blsgr,gdr->blsgd", s2, down2) + bias2i.view(1, 1, 1, G, D)
    out_shape = (B, L, *spatial, G, D)
    return yr.reshape(out_shape), yi.reshape(out_shape)


def _make_inputs(B, L, spatial, G, D, DM, R, *, device="cpu", dtype=torch.float32, seed=0):
    gen = torch.Generator(device="cpu").manual_seed(seed)

    def f(*shape):
        return torch.randn(*shape, generator=gen).to(device=device, dtype=dtype) * 0.1
    return (
        f(B, L, *spatial, G, D), f(B, L, *spatial, G, D),
        f(G, D, R), f(G, DM, R),
        f(L, G, R), f(L, G, R), f(G, DM), f(G, DM),
        f(G, DM, R), f(G, D, R),
        f(L, G, R), f(L, G, R), f(G, D), f(G, D),
    )


def _business_shape(args):
    """业务形态：一次 rFFT 轴上的 (img//2+1) 个低频模数 × 其余两轴展平。"""
    kept = min(args.modes, args.img // 2 + 1)
    return args.batch, kept, (args.img, args.img), args.groups, args.width // args.groups, \
        args.width // args.groups, args.rank


def _rel_l2(got: torch.Tensor, ref: torch.Tensor) -> float:
    denom = ref.norm().item()
    return float((got - ref).norm().item() / denom) if denom > 0 else 0.0


def _model_gelu(x: torch.Tensor) -> torch.Tensor:
    """模型原实现使用的 GELU：优先 fused ``npu_gelu``，否则精确 erf GELU。"""
    try:
        import torch_npu
        if x.device.type == "npu" and hasattr(torch_npu, "npu_gelu"):
            return torch_npu.npu_gelu(x)
    except Exception:
        pass
    return torch.nn.functional.gelu(x)


def _max_abs(got: torch.Tensor, ref: torch.Tensor) -> float:
    return float((got - ref).abs().max().item())


def _matched_ratio(got: torch.Tensor, ref: torch.Tensor, atol: float, rtol: float) -> float:
    """官方混合容差下的通过率：|a-g| <= atol + rtol*|g| 的元素占比。"""
    err = (got - ref).abs()
    lim = atol + rtol * ref.abs()
    finite = torch.isfinite(ref) & torch.isfinite(got)
    if not bool(finite.any()):
        return 1.0
    return float((err[finite] <= lim[finite]).float().mean())


def _sync(device):
    if device.type == "npu":
        torch.npu.synchronize()


def _precision_cases(args):
    cases = [
        ("small-b2", (2, 3, (2, 3), 4, 8, 8, 8)),
        ("small-d16", (1, 4, (4, 4), 2, 16, 16, 8)),
        ("odd-dims", (1, 3, (2, 3), 2, 12, 20, 6)),
    ]
    if args.img > 0:
        cases.append(("business", _business_shape(args)))
    return cases


def _run_precision(device, args) -> list[dict]:
    rows = []
    for name, shape in _precision_cases(args):
        B, L, spatial, G, D, DM, R = shape
        # 真值在 CPU float64 上计算，避免 NPU fp32 基线自身的舍入掩盖差异。
        base64 = _make_inputs(B, L, spatial, G, D, DM, R, dtype=torch.float64)
        ref64 = _reference(*base64)
        base = [t.float() for t in base64]
        dev_args = [t.to(device) for t in base]

        if device.type == "npu":
            got = spectral_mlp(*dev_args)
        else:
            got = spectral_mlp(*base)
        got0 = got[0].float().cpu().double()
        got1 = got[1].float().cpu().double()
        fwd = {
            "case": name,
            "shape": {"B": B, "L": L, "spatial": list(spatial), "G": G, "d": D, "dm": DM, "rank": R},
            "fwd_max_abs": max(_max_abs(got0, ref64[0]), _max_abs(got1, ref64[1])),
            "fwd_rel_l2": max(_rel_l2(got0, ref64[0]), _rel_l2(got1, ref64[1])),
            "fwd_matched_ratio": min(_matched_ratio(got0, ref64[0], args.atol, args.rtol),
                                     _matched_ratio(got1, ref64[1], args.atol, args.rtol)),
        }
        # 官方标准：通过率 >= 0.99，且绝对误差不超过 fp32 硬上限 1e-2。
        fwd["fwd_pass"] = fwd["fwd_matched_ratio"] >= args.matched_ratio and fwd["fwd_max_abs"] <= args.max_abs_limit

        # 反向：所有输入/参数都参与求导，与 float64 参考的 autograd 对比。
        # 损失用固定的随机权重对输出加权求和（同仓内其它算子的梯度对拍写法），
        # 让梯度量级保持在 O(1)，避免「对 1e8 个输出直接求和」把 fp32 的规约
        # 舍入放大成绝对误差。
        gen = torch.Generator().manual_seed(1234)
        w0_ref = torch.randn(ref64[0].shape, generator=gen)
        w1_ref = torch.randn(ref64[1].shape, generator=gen)
        autograd_src = dev_args if device.type == "npu" else base
        leaves = [t.clone().requires_grad_(True) for t in autograd_src]
        out = spectral_mlp(*leaves)
        w0 = w0_ref.float().to(out[0].device)
        w1 = w1_ref.float().to(out[1].device)
        loss = (out[0] * w0).sum() + (out[1] * w1).sum()
        grads = torch.autograd.grad(loss, leaves)
        _sync(device)

        ref_leaves = [t.clone().requires_grad_(True) for t in base64]
        ref_out = _reference(*ref_leaves)
        ref_grads = torch.autograd.grad(
            (ref_out[0] * w0_ref).sum() + (ref_out[1] * w1_ref).sum(), ref_leaves
        )

        names = ["xr", "xi", "up1", "down1", "gate1_r", "gate1_i", "bias1_r", "bias1_i",
                 "up2", "down2", "gate2_r", "gate2_i", "bias2_r", "bias2_i"]
        worst_abs = 0.0
        worst_rel = 0.0
        worst_matched = 1.0
        per_grad = []
        for name_g, g, rg in zip(names, grads, ref_grads):
            g64 = g.float().cpu().double()
            err_abs = _max_abs(g64, rg)
            err_rel = _rel_l2(g64, rg)
            matched = _matched_ratio(g64, rg, args.atol_bwd, args.rtol_bwd)
            worst_abs = max(worst_abs, err_abs)
            worst_rel = max(worst_rel, err_rel)
            worst_matched = min(worst_matched, matched)
            per_grad.append({"name": name_g, "max_abs": err_abs, "rel_l2": err_rel, "matched_ratio": matched})
        bwd = {
            "case": name,
            "bwd_max_abs": worst_abs,
            "bwd_rel_l2": worst_rel,
            "bwd_matched_ratio": worst_matched,
            "bwd_pass": worst_rel <= args.bwd_rel_l2 and worst_abs <= args.max_abs_limit,
            "grads": per_grad,
        }
        status = "PASS" if fwd["fwd_pass"] and bwd["bwd_pass"] else "FAIL"
        print(f"[precision] {name}: B={B} L={L} spatial={spatial} G={G} d={D} dm={DM} rank={R}")
        print(f"    fwd  matched_ratio={fwd['fwd_matched_ratio']:.5f} max_abs={fwd['fwd_max_abs']:.3e}"
              f" rel_l2={fwd['fwd_rel_l2']:.3e} -> {'PASS' if fwd['fwd_pass'] else 'FAIL'}")
        print(f"    bwd  matched_ratio={bwd['bwd_matched_ratio']:.5f} max_abs={bwd['bwd_max_abs']:.3e}"
              f" rel_l2={bwd['bwd_rel_l2']:.3e} -> {'PASS' if bwd['bwd_pass'] else 'FAIL'}")
        rows.append({"case": name, "status": status, **fwd, **bwd})
    return rows


def _time(fn, warmup: int, iters: int, device, repeats: int = 1) -> float:
    """重复 ``repeats`` 轮、每轮 ``iters`` 次，返回中位数（ms）。"""
    for _ in range(warmup):
        fn()
    _sync(device)
    vals = []
    for _ in range(max(1, repeats)):
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        _sync(device)
        vals.append((time.perf_counter() - t0) / max(1, iters) * 1e3)
    vals.sort()
    return vals[len(vals) // 2]


def _run_perf(device, args) -> list[dict]:
    if device.type != "npu":
        print("[perf] skipped: NPU not available")
        return []
    shape = _business_shape(args)
    B, L, spatial, G, D, DM, R = shape
    base = _make_inputs(B, L, spatial, G, D, DM, R, device=device)
    rows = []

    ref_in = [t.detach().requires_grad_(True) for t in base]
    # 前向基线用无梯度张量：只测推理前向，不把 autograd 建图开销算进来。
    def ref_fwd():
        return _reference(*base, act=_model_gelu)

    def triton_fwd():
        return spectral_mlp(*base)

    ref_ms = _time(ref_fwd, args.warmup, args.iters, device, args.repeats)
    triton_ms = _time(triton_fwd, args.warmup, args.iters, device, args.repeats)

    leaves = [t.detach().requires_grad_(True) for t in base]

    def op_step():
        o = spectral_mlp(*leaves)
        torch.autograd.grad(o[0].sum() + o[1].sum(), leaves, retain_graph=False)

    op_step()
    _sync(device)
    op_fb_ms = _time(op_step, args.warmup, args.iters, device, args.repeats)

    def ref_step():
        o = _reference(*ref_in, act=_model_gelu)
        torch.autograd.grad(o[0].sum() + o[1].sum(), ref_in, retain_graph=False)

    ref_step()
    _sync(device)
    ref_fb_ms = _time(ref_step, args.warmup, args.iters, device, args.repeats)

    # 显存不在进程内测峰值：allocator 的复用与常驻张量会让同一个算子在
    # 不同调用顺序下报出几 GB 的差异。显存收益写成可复算的中间量公式，
    # 见 tests/triton/spectral_mlp/report/README.md。
    rows.append({
        "case": "business",
        "shape": {"B": B, "L": L, "spatial": list(spatial), "G": G, "d": D, "dm": DM, "rank": R},
        "fwd_torch_ms": ref_ms,
        "fwd_triton_ms": triton_ms,
        "fwd_speedup": ref_ms / triton_ms if triton_ms > 0 else 0.0,
        "fwd_bwd_torch_ms": ref_fb_ms,
        "fwd_bwd_triton_ms": op_fb_ms,
        "fwd_bwd_speedup": ref_fb_ms / op_fb_ms if op_fb_ms > 0 else 0.0,
    })
    row = rows[0]
    print(f"[perf] spectral_mlp fwd torch={row['fwd_torch_ms']:.3f}ms "
          f"ascendc={row['fwd_triton_ms']:.3f}ms speedup={row['fwd_speedup']:.2f}x")
    print(f"[perf] spectral_mlp fwd+bwd torch={row['fwd_bwd_torch_ms']:.3f}ms "
          f"ascendc={row['fwd_bwd_triton_ms']:.3f}ms speedup={row['fwd_bwd_speedup']:.2f}x")
    return rows


def _environment(device) -> dict:
    env = {
        "host": platform.node(),
        "device": "cpu" if device.type != "npu" else "Ascend",
        "torch": torch.__version__,
        "python": platform.python_version(),
    }
    try:
        import torch_npu
        env["torch_npu"] = torch_npu.__version__
        env["device"] = torch_npu.npu.get_device_name(device) if device.type == "npu" else "cpu"
    except Exception:
        pass
    try:
        import triton
        env["triton"] = triton.__version__
    except Exception:
        pass
    return env


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("all", "precision", "perf"), default="all")
    ap.add_argument("--img", type=int, default=32, help="立方潜空间边长（128 为业务形态）")
    ap.add_argument("--modes", type=int, default=64, help="保留的低频模数上限")
    ap.add_argument("--width", type=int, default=128, help="通道数 C = G * d")
    ap.add_argument("--groups", type=int, default=8, help="分组数 G")
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=3, help="每项计时的重复轮数，取中位数")
    ap.add_argument("--atol", type=float, default=2 ** -13, help="fp32 混合容差 atol（默认 2^-13）")
    ap.add_argument("--rtol", type=float, default=2 ** -13, help="fp32 混合容差 rtol（默认 2^-13）")
    ap.add_argument("--atol-bwd", type=float, default=2 ** -13)
    ap.add_argument("--rtol-bwd", type=float, default=2 ** -13)
    ap.add_argument("--matched-ratio", type=float, default=0.99, help="官方标准的逐元素通过率下限")
    ap.add_argument("--max-abs-limit", type=float, default=1e-2, help="fp32 绝对误差上限（官方标准为 1e-2）")
    ap.add_argument("--bwd-rel-l2", type=float, default=1e-3,
                    help="反向梯度的相对 L2 阈值：跨 S 大规约的形状在 Cube fp32 下约 3e-4")
    ap.add_argument("--device-id", type=int, default=0)
    ap.add_argument("--json", default="", help="结果写入该 JSON 文件")
    ap.add_argument("--build-ext", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)

    try:
        import torch_npu  # noqa: F401
        device = torch.device(f"npu:{args.device_id}") if torch.npu.is_available() else torch.device("cpu")
    except ImportError:
        device = torch.device("cpu")

    result = {
        "operator": "spectral_mlp",
        "date": time.strftime("%Y-%m-%d"),
        "environment": _environment(device),
        "tolerance": {
            "atol": args.atol, "rtol": args.rtol,
            "atol_bwd": args.atol_bwd, "rtol_bwd": args.rtol_bwd,
            "matched_ratio": args.matched_ratio, "max_abs_limit": args.max_abs_limit,
            "backward_rel_l2": args.bwd_rel_l2,
        },
        "accuracy": [],
        "perf": [],
        "status": "PASS",
    }

    # 性能段先跑：精度段会在 allocator 里留下大量大块张量，之后再计时会让
    # 参考实现的分配变慢（实测能差 ~9%），两边不可比。
    if args.mode in ("all", "perf"):
        result["perf"] = _run_perf(device, args)

    if args.mode in ("all", "precision"):
        result["accuracy"] = _run_precision(device, args)
        if any(r["status"] != "PASS" for r in result["accuracy"]):
            result["status"] = "FAIL"

    if args.json:
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
        print(f"[json] wrote {path}")

    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
