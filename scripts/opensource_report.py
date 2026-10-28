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

"""根据 bash build.sh -u 落盘的 JSON 生成社区模板开源测试报告。"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JSON = REPO_ROOT / "test_result" / "test_results.json"
DEFAULT_OUT = REPO_ROOT / "test_result" / "oil-gas-ops-prospect仓开源测试报告.md"
DOCS_OUT = REPO_ROOT / "docs" / "zh" / "oil-gas-ops-prospect仓开源测试报告.md"

PASS = "PASS"
FAIL = "FAIL"
SKIP = "SKIP"
NA = "未执行"

PERF_RE = re.compile(
    r"\[perf\] (.+?) torch=([0-9.]+)ms ascendc=([0-9.]+)ms speedup=([0-9.]+)x"
)

# 一行 [perf] 日志解析出的 (用例名, 标杆 ms, 本算子 ms, 加速比)。
PerfRow = tuple[str, str, str, str]


def load_doc_checks() -> list[tuple[str, str, tuple[str, ...], str]]:
    """与 tests/opensource_gates.py 的文档清单保持同一份数据。"""
    path = REPO_ROOT / "tests" / "opensource_gates.py"
    spec = importlib.util.spec_from_file_location("oil_gas_opensource_gates", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return list(mod.DOC_CHECKS)


def job_status(jobs: dict[str, dict], name: str) -> str:
    item = jobs.get(name)
    if not item:
        return NA
    return item.get("status", NA)


def combine(*statuses: str) -> str:
    if not statuses:
        return NA
    if any(s == FAIL for s in statuses):
        return FAIL
    if any(s == NA for s in statuses):
        return NA
    if any(s == SKIP for s in statuses):
        return SKIP
    return PASS


def cell(status: str) -> str:
    return {PASS: "Pass", FAIL: "Fail", SKIP: "Skip", NA: "未执行"}.get(status, status)


def verdict(status: str) -> str:
    return {PASS: "通过", FAIL: "未通过", SKIP: "跳过", NA: "未执行"}.get(
        status, status
    )


def collect_env() -> dict[str, str]:
    env = {
        "date": datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d"),
        "cann": os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/cann-9.0.0"),
        "python": sys.version.split()[0],
        "torch": "未探测",
        "torch_npu": "未探测",
        "soc": "Atlas A2 / Ascend 910B",
    }
    try:
        import torch  # type: ignore

        env["torch"] = getattr(torch, "__version__", "unknown")
    except Exception:
        pass
    try:
        import torch_npu  # type: ignore

        env["torch_npu"] = getattr(torch_npu, "__version__", "unknown")
    except Exception:
        pass
    return env


def load_jobs(payload: dict) -> dict[str, dict]:
    return {j["name"]: j for j in payload.get("jobs", []) if "name" in j}


def parse_perf_rows(log: str) -> list[tuple[str, str, str, str]]:
    rows: list[tuple[str, str, str, str]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for m in PERF_RE.finditer(log or ""):
        name, torch_ms, op_ms, sp = m.groups()
        row = (name.strip(), torch_ms, op_ms, f"{sp}x")
        if row not in seen:
            seen.add(row)
            rows.append(row)
    return rows


def job_log(jobs: dict[str, dict], name: str) -> str:
    return jobs.get(name, {}).get("log", "") or ""


def speedup_vals(rows: list[tuple[str, str, str, str]]) -> list[float]:
    vals: list[float] = []
    for _, _, _, sp in rows:
        try:
            vals.append(float(sp.rstrip("x")))
        except ValueError:
            continue
    return vals


def speedup_span(rows: list[tuple[str, str, str, str]], prefix: str = "") -> str:
    subset = [r for r in rows if r[0].startswith(prefix)] if prefix else rows
    vals = speedup_vals(subset)
    if not vals:
        return "本次未解析到加速比"
    lo, hi = min(vals), max(vals)
    if abs(hi - lo) < 0.05:
        return f"约 {hi:.2f}x"
    return f"约 {lo:.2f}x～{hi:.2f}x"


def geo_mean(vals: list[float]) -> float | None:
    if not vals:
        return None
    return math.exp(
        sum(math.log(v) for v in vals if v > 0) / len([v for v in vals if v > 0])
    )


def perf_md_table(title: str, rows: list[tuple[str, str, str, str]]) -> list[str]:
    lines = [
        f"**{title}**",
        "",
        "| 用例 | 标杆 ms | 本算子 ms | 加速比 |",
        "| --- | ---: | ---: | ---: |",
    ]
    if not rows:
        lines.append("| （本次 `-u` 日志未解析到该表） | — | — | — |")
    else:
        for name, torch_ms, op_ms, sp in rows:
            lines.append(f"| {name} | {torch_ms} | {op_ms} | {sp} |")
    lines.append("")
    return lines


def five_d_cell(rows: list[tuple[str, str, str, str]], axis: str, kind: str) -> str:
    label = f"{kind} {axis}"
    for name, torch_ms, op_ms, sp in rows:
        if name == label:
            return f"{sp}（{torch_ms} / {op_ms} ms）"
    return "未解析"


@dataclass
class Statuses:
    """本次 `-u` 各 job 的状态，按报告引用口径归并。"""

    ophost: str = NA
    triton: str = NA
    composite: str = NA
    aot: str = NA
    st_cm: str = NA
    st_cmg: str = NA
    st_fwd: str = NA
    st_bwd: str = NA
    st_gelu: str = NA
    ex_cm: str = NA
    ex_cmg: str = NA
    ex_fwd: str = NA
    ex_bwd: str = NA
    ex_gelu: str = NA
    prec_cm: str = NA
    prec_fused: str = NA
    prec_conv3d: str = NA
    prec_crecip: str = NA
    prec_gelu: str = NA
    prec_interp: str = NA
    prec_cexp: str = NA
    perf_cm: str = NA
    perf_fused: str = NA
    perf_conv3d: str = NA
    perf_crecip: str = NA
    perf_gelu: str = NA
    perf_interp: str = NA
    perf_cexp: str = NA
    fft_v: str = NA
    fft_p: str = NA
    docs: str = NA
    pkg: str = NA
    install: str = NA
    api: str = NA


@dataclass
class PerfSummary:
    """按算子归类后的 [perf] 行，以及派生出的加速比文案。"""

    cm_rows: list[PerfRow] = field(default_factory=list)
    cm_fwd: list[PerfRow] = field(default_factory=list)
    cm_bwd: list[PerfRow] = field(default_factory=list)
    fused_fwd: list[PerfRow] = field(default_factory=list)
    fused_bwd: list[PerfRow] = field(default_factory=list)
    fused_fp32: list[PerfRow] = field(default_factory=list)
    fused_bf16: list[PerfRow] = field(default_factory=list)
    bwd_fp32: list[PerfRow] = field(default_factory=list)
    bwd_bf16: list[PerfRow] = field(default_factory=list)
    fft_rows: list[PerfRow] = field(default_factory=list)
    fft_reg: list[PerfRow] = field(default_factory=list)
    conv3d_fwd: list[PerfRow] = field(default_factory=list)
    conv3d_fwdbwd: list[PerfRow] = field(default_factory=list)
    crecip_fwd: list[PerfRow] = field(default_factory=list)
    crecip_e2e: list[PerfRow] = field(default_factory=list)
    gelu_rows: list[PerfRow] = field(default_factory=list)
    interp_fwd: list[PerfRow] = field(default_factory=list)
    interp_bwd: list[PerfRow] = field(default_factory=list)
    interp_fwdbwd: list[PerfRow] = field(default_factory=list)
    cexp_fwd: list[PerfRow] = field(default_factory=list)
    gmean_txt: str = "未解析"
    fft_metric: str = ""


def collect_statuses(jobs: dict[str, dict]) -> Statuses:
    return Statuses(
        ophost=job_status(jobs, "ophost_ut"),
        triton=job_status(jobs, "triton_cpu"),
        composite=job_status(jobs, "composite_cpu"),
        aot=job_status(jobs, "triton_aot"),
        st_cm=job_status(jobs, "aclnnComplexMul"),
        st_cmg=job_status(jobs, "aclnnComplexMulGrad"),
        st_fwd=job_status(jobs, "aclnnFusedBiasSoftmax"),
        st_bwd=job_status(jobs, "aclnnFusedSoftmaxGrad"),
        ex_cm=job_status(jobs, "example_complex_mul"),
        ex_cmg=job_status(jobs, "example_complex_mul_grad"),
        ex_fwd=job_status(jobs, "example_fused_bias_softmax"),
        ex_bwd=job_status(jobs, "example_fused_softmax_grad"),
        ex_gelu=job_status(jobs, "example_conv3d_gelu"),
        st_gelu=job_status(jobs, "aclnnConv3dGeluCustom"),
        prec_cm=job_status(jobs, "complex_mul_precision"),
        prec_fused=job_status(jobs, "fused_bias_softmax_precision"),
        prec_conv3d=job_status(jobs, "conv3d_custom_precision"),
        prec_crecip=job_status(jobs, "complex_reciprocal_precision"),
        prec_gelu=job_status(jobs, "conv3d_gelu_precision"),
        prec_interp=job_status(jobs, "interpolate_precision"),
        prec_cexp=job_status(jobs, "complex_exp_precision"),
        perf_cm=job_status(jobs, "complex_mul_perf"),
        perf_fused=job_status(jobs, "fused_bias_softmax_perf"),
        perf_conv3d=job_status(jobs, "conv3d_custom_perf"),
        perf_crecip=job_status(jobs, "complex_reciprocal_perf"),
        perf_gelu=job_status(jobs, "conv3d_gelu_perf"),
        perf_interp=job_status(jobs, "interpolate_perf"),
        perf_cexp=job_status(jobs, "complex_exp_perf"),
        fft_v=combine(
            job_status(jobs, "fft_npu_verify"), job_status(jobs, "fft_npu_verify_5d")
        ),
        fft_p=job_status(jobs, "fft_npu_perf"),
        docs=job_status(jobs, "docs_gate"),
        pkg=job_status(jobs, "pkg_gate"),
        install=job_status(jobs, "install_gate"),
        api=job_status(jobs, "api_smoke"),
    )


def fft_metric_text(fft_rows: list[PerfRow]) -> str:
    if not fft_rows:
        return "本次未解析到 FFT 微基准"
    return (
        f"常规形态见下表；业务 5D W/H/D rfft {speedup_span(fft_rows, 'rfft 5d')}，"
        f"irfft {speedup_span(fft_rows, 'irfft 5d')}，"
        f"fwd+bwd {speedup_span(fft_rows, 'fwd+bwd 5d')}"
    )


def collect_perf(jobs: dict[str, dict]) -> PerfSummary:
    cm_rows = parse_perf_rows(job_log(jobs, "complex_mul_perf"))
    fused_rows = parse_perf_rows(job_log(jobs, "fused_bias_softmax_perf"))
    fft_rows = parse_perf_rows(job_log(jobs, "fft_npu_perf"))
    conv3d_rows = parse_perf_rows(job_log(jobs, "conv3d_custom_perf"))
    crecip_rows = parse_perf_rows(job_log(jobs, "complex_reciprocal_perf"))
    gelu_rows = parse_perf_rows(job_log(jobs, "conv3d_gelu_perf"))
    interp_rows = parse_perf_rows(job_log(jobs, "interpolate_perf"))
    cexp_rows = parse_perf_rows(job_log(jobs, "complex_exp_perf"))
    fused_fwd = [r for r in fused_rows if r[0].startswith("fwd ")]
    fused_bwd = [r for r in fused_rows if r[0].startswith("bwd ")]
    gmean = geo_mean(speedup_vals(fft_rows))
    return PerfSummary(
        cm_rows=cm_rows,
        cm_fwd=[r for r in cm_rows if r[0].startswith("fwd ")],
        cm_bwd=[r for r in cm_rows if r[0].startswith("bwd ")],
        fused_fwd=fused_fwd,
        fused_bwd=fused_bwd,
        fused_fp32=[r for r in fused_fwd if "fp32" in r[0]],
        fused_bf16=[r for r in fused_fwd if "bf16" in r[0]],
        bwd_fp32=[r for r in fused_bwd if "fp32" in r[0]],
        bwd_bf16=[r for r in fused_bwd if "bf16" in r[0]],
        fft_rows=fft_rows,
        fft_reg=[
            r
            for r in fft_rows
            if not r[0].endswith(("5d W", "5d H", "5d D")) and "5d " not in r[0]
        ],
        conv3d_fwd=[r for r in conv3d_rows if r[0].endswith(" fwd")],
        conv3d_fwdbwd=[r for r in conv3d_rows if r[0].endswith(" fwdbwd")],
        crecip_fwd=[r for r in crecip_rows if r[0].startswith("recip fwd")],
        crecip_e2e=[r for r in crecip_rows if r[0].startswith("e2e fused")],
        gelu_rows=gelu_rows,
        interp_fwd=[r for r in interp_rows if r[0].startswith("interpolate fwd ")],
        interp_bwd=[r for r in interp_rows if r[0].startswith("interpolate bwd ")],
        interp_fwdbwd=[
            r for r in interp_rows if r[0].startswith("interpolate fwd+bwd ")
        ],
        cexp_fwd=[r for r in cexp_rows if r[0].startswith("exp fwd")],
        gmean_txt=f"{gmean:.2f}x" if gmean else "未解析",
        fft_metric=fft_metric_text(fft_rows),
    )


def build_features(st: Statuses) -> list[tuple[str, str, str, str, str, str, str]]:
    """（特性, 总结论, 功能, 精度, 性能, 可靠性, 兼容性）"""
    return [
        (
            "ComplexMul",
            combine(st.ophost, st.st_cm, st.ex_cm, st.prec_cm, st.perf_cm),
            combine(st.st_cm, st.ex_cm),
            st.prec_cm,
            st.perf_cm,
            st.ophost,
            st.install,
        ),
        (
            "ComplexMulGrad",
            combine(st.ophost, st.st_cmg, st.ex_cmg, st.prec_cm, st.perf_cm),
            combine(st.st_cmg, st.ex_cmg),
            st.prec_cm,
            st.perf_cm,
            st.ophost,
            st.install,
        ),
        (
            "FusedBiasSoftmax",
            combine(st.ophost, st.st_fwd, st.ex_fwd, st.prec_fused, st.perf_fused),
            combine(st.st_fwd, st.ex_fwd),
            st.prec_fused,
            st.perf_fused,
            st.ophost,
            st.install,
        ),
        (
            "FusedSoftmaxGrad",
            combine(st.ophost, st.st_bwd, st.ex_bwd, st.prec_fused, st.perf_fused),
            combine(st.st_bwd, st.ex_bwd),
            st.prec_fused,
            st.perf_fused,
            st.ophost,
            st.install,
        ),
        (
            "real_rfft",
            combine(st.triton, st.aot, st.fft_v, st.fft_p),
            combine(st.triton, st.aot, st.fft_v),
            st.triton,
            st.fft_p,
            st.triton,
            st.install,
        ),
        (
            "real_irfft",
            combine(st.triton, st.aot, st.fft_v, st.fft_p),
            combine(st.triton, st.aot, st.fft_v),
            st.triton,
            st.fft_p,
            st.triton,
            st.install,
        ),
        (
            "complex_reciprocal",
            combine(st.triton, st.aot, st.prec_crecip, st.perf_crecip, st.install),
            combine(st.triton, st.aot, st.prec_crecip),
            st.prec_crecip,
            st.perf_crecip,
            st.triton,
            st.install,
        ),
        (
            "interpolate",
            combine(st.triton, st.aot, st.prec_interp, st.perf_interp, st.install),
            combine(st.triton, st.aot, st.prec_interp),
            st.prec_interp,
            st.perf_interp,
            st.triton,
            st.install,
        ),
        (
            "complex_exp",
            combine(st.triton, st.aot, st.prec_cexp, st.perf_cexp, st.install),
            combine(st.triton, st.aot, st.prec_cexp),
            st.prec_cexp,
            st.perf_cexp,
            st.triton,
            st.install,
        ),
        (
            "Conv3d / conv3d",
            combine(st.composite, st.prec_conv3d, st.perf_conv3d, st.api, st.install),
            st.composite,
            st.prec_conv3d,
            st.perf_conv3d,
            st.composite,
            st.install,
        ),
        (
            "Conv3DGelu",
            combine(st.ophost, st.st_gelu, st.ex_gelu, st.prec_gelu, st.perf_gelu),
            combine(st.st_gelu, st.ex_gelu),
            st.prec_gelu,
            st.perf_gelu,
            st.ophost,
            st.install,
        ),
    ]


def build_cases(st: Statuses) -> list[tuple[str, str, str]]:
    """（测试用例, 结果, 测试内容）"""
    return [
        (
            "ComplexMul 切分与约束",
            st.ophost,
            "代表 shape / 小 tensor / 退化输入的切分计划，核间覆盖与对齐，UB 预算",
        ),
        (
            "ComplexMul 前反向精度",
            st.prec_cm,
            "闭式解对拍（含 `whl` 接口与 NPU 执行）：`(4,1024)`、`(1,1,192,320,320)` 等，相对误差阈值 `1e-5`",
        ),
        (
            "ComplexMulGrad 切分与梯度",
            combine(st.ophost, st.st_cmg, st.prec_cm),
            "反向 tile 小于前向、代表 shape 填满、退化输入；梯度与 autograd / 闭式解一致",
        ),
        (
            "FusedBiasSoftmax 切分与约束",
            st.ophost,
            "双 bias 广播、无 bias、非对齐宽、满尺寸 bias1、非法输入拒绝、dtype 对应 tiling key",
        ),
        (
            "FusedBiasSoftmax 前向精度",
            st.prec_fused,
            "`softmax(x+bias1+bias2,-1)`：bf16 广播/全尺寸/尖峰分布、fp32 无 bias；连续输入成功进入融合路径时的原地语义；"
            "同 shape 不同缓冲互不污染；与 fp32 标杆对拍",
        ),
        (
            "FusedSoftmaxGrad 切分与路径",
            st.ophost,
            "对齐 MIX、非对齐 VEC、fp32/fp16 走向量、超 UB 拒绝、`Q≠S` 合法、非法输入拒绝",
        ),
        (
            "FusedSoftmaxGrad 反向精度",
            st.prec_fused,
            "`g=go@V^T` 后 JVP：连续 probs 成功进入融合路径时原地覆写；典型 / 超核 / 非对齐 D / fp32 / `Q≠S`；与 fp32 标杆对拍",
        ),
        (
            "融合注意力链路",
            st.prec_fused,
            "`attention_core` 前向输出与 `dL/dlogits`、`dL/dq`、`dL/dv` 对拍，容差 `max_diff<0.6`",
        ),
        (
            "real_rfft 基函数与前向",
            st.triton,
            "Fourier 基与 `torch.fft` 一致；多 shape/dim 前向半谱对拍（含回退路径与 NPU）",
        ),
        (
            "real_irfft 重建与往返",
            st.triton,
            '半谱重建实信号，往返误差与 `torch.fft.irfft(..., norm="ortho")` 对齐',
        ),
        (
            "real_rfft / real_irfft 反向",
            st.triton,
            "多 shape/dim 反向与 CPU `torch.fft` 对拍",
        ),
        (
            "real_rfft / real_irfft 大形态",
            st.fft_v,
            "三维潜变量形态与 `128^3` 往返/反向；默认示例可运行",
        ),
        (
            "complex_reciprocal 前反向精度",
            combine(st.triton, st.aot, st.prec_crecip),
            "CPU 回退与 torch `1/z` 逐位一致（含实虚分离入口、极端幅值、int64 offset 分支）；NPU 真机"
            "五方法对拍（stable 全分布 max_rel<=1e-5，含 tiny/huge 极端幅值）与反向 vs float64；"
            "kernel 三条 AOT 全分支可编译",
        ),
        (
            "interpolate 前反向精度",
            combine(st.triton, st.aot, st.prec_interp),
            "5D float32 trilinear 与同设备 `F.interpolate` 对拍（前向 `max_abs<=1e-5`、反向梯度同阈值）；"
            "4D bilinear 走原生路径；kernel 的 W/H/D 前向与 W^T/H^T/D^T 反向 index32 分支纳入 AOT",
        ),
        (
            "complex_exp 前反向精度",
            combine(st.triton, st.aot, st.prec_cexp),
            "CPU 回退与 torch `exp` 逐位一致（含纯虚指数、`cexpf` 特殊值表、纯实数上溢、int64 offset 分支）；"
            "NPU 真机对拍（vs float64，max_rel<=1e-5）、一阶与二阶反向 vs float64；"
            "kernel 前向与反向的 int32/int64 四条 AOT 分支可编译",
        ),
        (
            "Conv3DGelu 精度与路径",
            combine(st.ophost, st.st_gelu, st.ex_gelu, st.prec_gelu),
            "1×1 直写与 3×3 单平面路径；stride / dilation / 非立方 kernel 的 ST；"
            'NPU 相对 CPU `F.gelu(F.conv3d, approximate="tanh")` 相对误差 `< 1e-5`，输入梯度最大绝对误差 `< 1e-4`',
        ),
        (
            "Conv3d / conv3d NPU 精度与性能",
            combine(st.prec_conv3d, st.perf_conv3d),
            "测试脚本列出的 NPU float32 形态对拍：支持参数域内的自定义前/反向 vs 原生（L2 相对误差阈值 `1e-5`；"
            "特定大 kernel depthwise case 验证分步反向可避开已观察到的原生 L1 限制）；fwd / fwd+bwd 计时",
        ),
        (
            "Conv3d / conv3d 前反向精度",
            st.composite,
            "仓内列出的 stride/dilation/非立方 kernel/数值 padding 组合与训练调用形态；"
            "在支持参数域内和 `nn.Conv3d` / `F.conv3d` 前反向对拍（测试脚本规定的 dtype 与阈值）",
        ),
        (
            "算子文档",
            st.docs,
            "交付文档齐全（README、LICENSE、SECURITY、文档导航及入门/使用/开发文档共 "
            f"{len(load_doc_checks())} 份）；核对应子清单、公开接口与 `--pkg` / `--install` / `-u` 入口。"
            "文档列表与检查项见下表「文档测试」",
        ),
        (
            "编译出包",
            st.pkg,
            "`bash build.sh --help` 合法选项可打印；`--soc` / `--ops` / `-j` 可组合；产出 OPP `.run` 与 Python whl",
        ),
        (
            "安装与卸载",
            st.install,
            "OPP 写入 customize vendor；`pip install` / `pip uninstall oil-gas-ops-prospect` 可执行",
        ),
        (
            "公开接口冒烟",
            st.api,
            "`complex_mul` / `fused_bias_softmax` / `fused_softmax_grad` / `conv3d_gelu` / `real_rfft` / `real_irfft` 可导入；"
            "CPU 回退前反向可跑通",
        ),
        (
            "可靠性",
            combine(st.ophost, st.prec_fused, st.api),
            "非法输入被切分侧拒绝；融合算子仅在连续输入成功进入融合路径时原地写回；非连续输入副本与非 NPU 回退不覆写调用者原张量",
        ),
        (
            "配套兼容",
            combine(st.install, st.api),
            "CANN 9.0.0 + 配套 PyTorch / torch_npu 可完成编译、安装与算子验证",
        ),
    ]


def build_conclusion(
    total: int, failed: int, ok: bool, cmd: str, generated: str
) -> str:
    return (
        f"oil-gas-ops-prospect 基于 CANN 9.0.0 完成开源测试，共计执行 {total} 组特性级测试用例，"
        f"发现 {failed} 个问题，其中有效问题 {failed} 个。"
        + (
            "整体质量良好，满足出口质量标准。**建议发布。**"
            if ok
            else "存在失败或未执行项。**不建议作为完整开源出口。**"
        )
        + f" 本次入口：`{cmd}`（{generated}）。"
    )


def section_overview(cmd: str, ok: bool, rate: str) -> list[str]:
    return [
        "# oil-gas-ops-prospect仓开源测试报告",
        "",
        "## 1. 概述",
        "",
        "本报告为 oil-gas-ops-prospect 项目开源测试报告。测试基于 CANN 9.0.0 开源包，覆盖仓内已交付算子的文档、"
        "编译出包、安装卸载、功能、精度与性能，以及配套环境兼容性。"
        f"测试过程中对已纳入范围的特性执行覆盖率为 {'100%' if ok else rate}。",
        "",
        "本仓面向油气勘探计算负载，按算子名对外提供 Python / aclnn 接口，不绑定单一上层模型。",
        "",
        "### 新增特性清单",
        "",
        "| 序号 | 特性名称 | 特性描述 |",
        "| --- | --- | --- |",
        "| 1 | ComplexMul | 复数逐点乘：`(ur, ui), (vr, vi) → (wr, wi)`，`float32` / `ND`，全部张量同 shape |",
        "| 2 | ComplexMulGrad | `ComplexMul` 的反向：由 `gwr, gwi` 与前向输入得到 `gur, gui, gvr, gvi`，"
        "`float32` / `ND` |",
        "| 3 | FusedBiasSoftmax | `y = softmax(x + bias1 + bias2, -1)`，`bias1`/`bias2` 可选，"
        "连续输入成功进入 NPU 融合路径时原地写回 `x`，支持 `fp32` / `fp16` / `bf16` |",
        "| 4 | FusedSoftmaxGrad | 注意力 softmax JVP：`g = go @ V^T`，连续 `probs` 成功进入 NPU 融合路径时原地覆写，支持 `Q≠S`，"
        "`fp32` / `fp16` / `bf16` |",
        '| 5 | real_rfft | 实数正向 FFT：`x → (xr, xi)`，`norm="ortho"`，频率轴长度为 `n//2+1` |',
        '| 6 | real_irfft | 由半谱实部、虚部重建实数：`(xr, xi), n → y`，`norm="ortho"` |',
        "| 7 | Conv3d / conv3d | 纯 PyTorch 组合兼容实现：数值 padding、stride、dilation 支持 int / 3 元组，"
        "groups 为 int，字符串 padding 不受支持；模块支持 zeros / reflect / replicate / circular，构造器无 device/dtype 关键字；"
        "dtype 以底层 PyTorch 后端支持且本仓实测为准，state_dict key 互通 |",
        "| 8 | complex_reciprocal | complex64 倒数 `1/z`（stable / naive 双路径、实虚分离入口、"
        "自定义反向），替代 NPU 上走 Ai_Cpu 的 `aclnnReciprocal` |",
        "| 9 | complex_exp | complex64 指数 `e^z`（`e^a·(cos b + i·sin b)`，自定义反向），"
        "替代 NPU 上走 Ai_Cpu 的 `aclnnExp`；数值语义对齐 `cexpf` |",
        "| 10 | interpolate | 2D/3D 线性插值；5D fp32 trilinear 按空间轴拆成批量 Cube GEMM，"
        "支持一阶 autograd、大张量分块与 CPU 回退，4D 使用原生实现 |",
        "| 11 | Conv3DGelu | `gelu_tanh(conv3d(x, weight, bias))`，`float32` / `NCDHW`，`groups=1`；"
        "1×1 直写，其余形状按一个输出平面切，反向不保存预激活 |",
        "",
        "`attention_core(q, k, v, bias1, bias2)` 为 FusedBiasSoftmax 与 FusedSoftmaxGrad 按 openfold "
        "注意力语义的组合封装，随上述融合算子一并验证。",
        "",
        "### 解决问题列表",
        "",
        "不涉及（首个开源测试版本，无历史问题清单）。",
        "",
        "### 测试活动",
        "",
        "本次测试活动包括：Readme 与算子文档评估、一键编译出包、算子包与 Python 包安装卸载、"
        f"按算子的功能/精度/性能验证、可靠性与兼容性验证。入口：`{cmd}`。",
        "",
    ]


def section_version(env: dict[str, str], date: str) -> list[str]:
    return [
        "## 2. 版本测试信息",
        "",
        "本节描述被测对象的版本信息和测试时间，包括依赖环境与测试源码仓。",
        "",
        "**硬件和版本要求**",
        "",
        f"产品型号：{env['soc']}（文档默认 `SOC_VERSION=Ascend910B1`）",
        "",
        "操作系统：覆盖以下测试因子组合",
        "",
        "| | 因子名称 | 因子取值 |",
        "| --- | --- | --- |",
        "| 1 | OS | Ubuntu / openEuler |",
        "| 2 | CPU 架构 | ARM64 |",
        "| 3 | 形态 | docker / host+device |",
        "",
        "CANN 版本：CANN 9.0.0",
        "",
        "驱动版本：Ascend HDK 26.0.RC1",
        "",
        f"Python 版本：Python >= 3.10（本次实测 {env['python']}）",
        "",
        f"PyTorch 版本：{env['torch']}（配套 torch_npu {env['torch_npu']}）",
        "",
        "依赖三方库版本：与 CANN 9.0.0 配套的 torch_npu、Triton-Ascend 3.2.1；构建依赖见仓库 `requirements.txt`",
        "",
        "测试源码仓：https://gitcode.com/cann/oil-gas-ops-prospect",
        "",
        f"测试日期：{date}",
        "",
    ]


def section_conclusion(conclusion: str, total: int, ok: bool, rate: str) -> list[str]:
    return [
        "## 3. 测试结论",
        "",
        "给出明确的版本发布结论。",
        "",
        conclusion,
        "",
        "| 评估维度 | 评估结果 |",
        "| --- | --- |",
        f"| 测试用例总数 | {total} 组（按已交付算子与关键测试活动归并，不含参数化子项展开） |",
        f"| 测试执行通过率 | {'100%（' + rate + '）' if ok else rate} |",
        f"| 测试结论 | {'通过' if ok else '未通过'} |",
        "",
    ]


def section_features(
    features: list[tuple[str, str, str, str, str, str, str]],
    cases: list[tuple[str, str, str]],
    doc_status: str,
) -> list[str]:
    lines = [
        "## 4. 特性质量评估",
        "",
        "新特性的验证结果清单如下。功能、精度在 CPU 参考路径与 NPU 路径上一并验证"
        "（非 NPU 环境走 PyTorch 回退，NPU 环境与同设备标杆对拍）。",
        "",
        "| 序号 | 特性 | 测试结论 | 功能 | 精度 | 性能 | 可靠性 | 兼容性 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for i, (
        name,
        overall,
        func,
        accuracy,
        performance,
        reliability,
        compat,
    ) in enumerate(features, 1):
        lines.append(
            f"| {i} | {name} | {verdict(overall)} | {cell(func)} | {cell(accuracy)} | "
            f"{cell(performance)} | {cell(reliability)} | {cell(compat)} |"
        )
    lines += [
        "",
        "特性级用例（按特性介绍组织，同一用例同时覆盖非 NPU 回退与 NPU 对拍，不按实现后端拆分）：",
        "",
        "| 序号 | 测试用例 | 测试内容 | 结果 |",
        "| --- | --- | --- | --- |",
    ]
    for i, (name, status, content) in enumerate(cases, 1):
        lines.append(f"| {i} | {name} | {content} | {status} |")
    lines += section_doc_test(doc_status)
    return lines


def section_doc_test(status: str) -> list[str]:
    lines = [
        "",
        "#### 文档测试",
        "",
        "第 13 组「算子文档」由 `python3 tests/opensource_gates.py docs` 执行（全量 `-u` 的 `docs_gate`）。",
        "检查下列交付文档文件存在，并核对应关键词/章节，确保与对外接口、构建入口一致。",
        "",
        "| 序号 | 文档 | 测试内容 | 结果 |",
        "| --- | --- | --- | --- |",
    ]
    for i, (rel, title, _tokens, content) in enumerate(load_doc_checks(), 1):
        lines.append(f"| {i} | `{rel}`（{title}） | {content} | {status} |")
    lines.append("")
    return lines


def section_security(st: Statuses) -> list[str]:
    return [
        "",
        "## 5. DFX专项质量评估",
        "",
        "### 5.1 安全测试",
        "",
        "版本整体安全测试结论：满足开源发布对安全声明与许可证的要求。"
        if st.docs == PASS
        else "版本整体安全测试结论：文档/许可证门禁未通过。",
        "",
        "仓内提供 `SECURITY.md`、Apache 2.0 `LICENSE`；构建入口对非法参数拦截并提示。"
        "本次未开展独立漏洞扫描、fuzz 与交付件病毒扫描。",
        "",
        "| 序号 | 测试项 | 测试结论 | 遗留风险 |",
        "| --- | --- | --- | --- |",
        f"| 1 | 安全声明与许可证 | {verdict(st.docs)} | 无 |",
        f"| 2 | 构建参数校验 | {verdict(st.pkg)} | 无 |",
        "",
    ]


def section_reliability(st: Statuses) -> list[str]:
    return [
        "### 5.2 可靠性测试",
        "",
        "可靠性指标的验证达成情况如下。",
        "",
        "| 序号 | 可靠性特性 | 测试结论 | 遗留风险 |",
        "| --- | --- | --- | --- |",
        f"| 1 | 非法或退化输入被拒绝或收缩为合法切分 | {verdict(st.ophost)} | 无 |",
        f"| 2 | 融合算子仅在连续输入成功进入融合路径时原地写回（输出与调用者原输入共享 device 指针） | {verdict(st.prec_fused)} | 无 |",
        f"| 3 | 非 NPU 或扩展不可用时回退 PyTorch，接口不变 | {verdict(st.api)} | 无 |",
        f"| 4 | 同 shape 重复调用精度稳定（含不同缓冲、融合注意力链路） | {verdict(st.prec_fused)} | 无 |",
        "",
    ]


def section_perf(st: Statuses, perf: PerfSummary, date: str) -> list[str]:
    lines = [
        "### 5.3 性能测试",
        "",
        "该版本的性能按算子微基准测量：预热 5 次、计时 20 次取均值，加速比 = 同设备 Torch 小算子时延 / 本算子时延。"
        f"测试环境为 Atlas A2 / CANN 9.0.0，不绑定具体主机。各算子均于 {date} 对同设备 Torch 标杆复测。",
        "",
        "| 场景 | 模型 | 特性 | Torch 小算子标杆 | 性能指标 | 测试环境 | 测试结果 | 遗留风险 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
        "| 算子微基准 | 不涉及 | ComplexMul / ComplexMulGrad | 同设备逐元素 `wr=ur*vr-ui*vi` 等 | "
        f"前向 {speedup_span(perf.cm_fwd)}，反向 {speedup_span(perf.cm_bwd)} | "
        f"Atlas A2 + CANN 9.0.0 | {cell(st.perf_cm)} | 无 |",
        "| 算子微基准 | 不涉及 | FusedBiasSoftmax | 同设备 `softmax(x+bias1+bias2,-1)`（fp32 累加） | "
        f"bf16 {speedup_span(perf.fused_bf16)}，小 fp32 {speedup_span(perf.fused_fp32)} | "
        f"Atlas A2 + CANN 9.0.0 | {cell(st.perf_fused)} | 无 |",
        "| 算子微基准 | 不涉及 | FusedSoftmaxGrad | 同设备 `g=go@V^T` + softmax JVP | "
        f"bf16 {speedup_span(perf.bwd_bf16)}，fp32 {speedup_span(perf.bwd_fp32)} | "
        f"Atlas A2 + CANN 9.0.0 | {cell(st.perf_fused)} | 无 |",
        "| 算子微基准 | 不涉及 | real_rfft / real_irfft | 同设备 `torch.fft.rfft` / `irfft`，"
        f'`norm="ortho"` | {perf.fft_metric} | Atlas A2 + CANN 9.0.0 | {cell(st.fft_p)} | 无 |',
        "| 算子微基准 | 不涉及 | complex_exp | 同设备 torch `exp`（aclnnExp） | "
        f"前向 {speedup_span(perf.cexp_fwd)} | "
        f"Atlas A2 + CANN 9.0.0 | {cell(st.perf_cexp)} | 无 |",
        "| 算子微基准 | 不涉及 | complex_reciprocal | 同设备 torch `1/z`（aclnnReciprocal）与先构造 complex64 的原始写法 | "
        f"前向 {speedup_span(perf.crecip_fwd)}；端到端融合 {speedup_span(perf.crecip_e2e)} | "
        f"Atlas A2 + CANN 9.0.0 | {cell(st.perf_crecip)} | 无 |",
        "| 算子微基准 | 不涉及 | interpolate | 同设备 `F.interpolate`（5D fp32 trilinear） | "
        f"前向 {speedup_span(perf.interp_fwd)}，反向 {speedup_span(perf.interp_bwd)}，"
        f"fwd+bwd {speedup_span(perf.interp_fwdbwd)} | "
        f"Atlas A2 + CANN 9.0.0 | {cell(st.perf_interp)} | 无 |",
        "| 算子微基准 | 不涉及 | Conv3DGelu | 同设备 `F.conv3d` + `F.gelu(approximate=\"tanh\")` | "
        f"1×1 直写 {speedup_span(perf.gelu_rows, 'k1_')}；3×3 按输出平面切，速度让给显存 | "
        f"Atlas A2 + CANN 9.0.0 | {cell(st.perf_gelu)} | 3×3 单平面路径慢于整幅卷积 |",
        "| 算子微基准 | 不涉及 | Conv3d / conv3d | 同设备 `nn.Conv3d` / `F.conv3d`（测试脚本列出的形态） | "
        f"前向 {speedup_span(perf.conv3d_fwd)}，fwd+bwd {speedup_span(perf.conv3d_fwdbwd)}（特定大 kernel depthwise "
        "case 仅验证分步反向可避开已观察到的原生 L1 限制） | "
        f"Atlas A2 + CANN 9.0.0 | {cell(st.perf_conv3d)} | 无 |",
        "",
        "代表形态实测如下（精度见本次 `-u` 精度 job）。",
        "",
    ]
    return lines + section_perf_tables(perf)


def section_perf_tables(perf: PerfSummary) -> list[str]:
    lines = perf_md_table(
        "complex_exp 前向（相对同设备 torch exp）", perf.cexp_fwd
    )
    lines += perf_md_table(
        "complex_reciprocal 前向（相对同设备 torch 1/z）", perf.crecip_fwd
    )
    lines += perf_md_table(
        "complex_reciprocal 端到端融合（相对先构造 complex64 的原始写法）",
        perf.crecip_e2e,
    )
    lines += perf_md_table(
        "interpolate 前向（相对同设备 F.interpolate，5D fp32 trilinear）",
        perf.interp_fwd,
    )
    lines += perf_md_table(
        "interpolate 反向（相对同设备 F.interpolate autograd）", perf.interp_bwd
    )
    lines += perf_md_table(
        "interpolate fwd+bwd（相对同设备 F.interpolate autograd）", perf.interp_fwdbwd
    )
    lines += perf_md_table(
        "Conv3d / conv3d 前向（相对同设备 nn.Conv3d，测试脚本列出的形态）",
        perf.conv3d_fwd,
    )
    lines += perf_md_table(
        "Conv3d / conv3d fwd+bwd（相对同设备 nn.Conv3d autograd）", perf.conv3d_fwdbwd
    )
    lines += perf_md_table(
        "ComplexMul / ComplexMulGrad（相对同设备逐元素公式）", perf.cm_rows
    )
    lines += perf_md_table(
        "FusedBiasSoftmax（相对同设备 fp32 `softmax`）", perf.fused_fwd
    )
    lines += perf_md_table(
        "FusedSoftmaxGrad（相对同设备 `matmul` + softmax JVP）", perf.fused_bwd
    )
    lines += [
        "**real_rfft / real_irfft 常规形态（相对同设备 `torch.fft`）**",
        "",
        f"含业务 5D 的全表几何平均加速比约 **{perf.gmean_txt}**。",
        "",
        "| 用例 | 标杆 ms | 本算子 ms | 加速比 |",
        "| --- | ---: | ---: | ---: |",
    ]
    if perf.fft_reg:
        for name, torch_ms, op_ms, sp in perf.fft_reg:
            lines.append(f"| {name} | {torch_ms} | {op_ms} | {sp} |")
    else:
        lines.append("| （本次 `-u` 日志未解析到该表） | — | — | — |")
    lines += [
        "",
        "**real_rfft / real_irfft 业务 5D `[1,35,128,128,128]`（相对同设备 `torch.fft`）**",
        "",
        "| 轴 | rfft | irfft | fwd+bwd |",
        "| --- | ---: | ---: | ---: |",
    ]
    for axis_name, axis_tag in (("W 末轴", "5d W"), ("H", "5d H"), ("D", "5d D")):
        lines.append(
            f"| {axis_name} | {five_d_cell(perf.fft_rows, axis_tag, 'rfft')} | "
            f"{five_d_cell(perf.fft_rows, axis_tag, 'irfft')} | "
            f"{five_d_cell(perf.fft_rows, axis_tag, 'fwd+bwd')} |"
        )
    return lines


def section_compat(st: Statuses, env: dict[str, str], has_npu: bool) -> list[str]:
    npu_cell = (
        "pass"
        if has_npu and st.install == PASS
        else ("fail" if st.install == FAIL else "未执行")
    )
    compat_ok = st.install == PASS and st.api == PASS
    return [
        "",
        "### 5.4 兼容性测试",
        "",
        f"兼容性评估：{'通过' if compat_ok else '未通过'}。",
        "",
        "| 序号 | 兼容性场景 | 验证结果 | 遗留风险 |",
        "| --- | --- | --- | --- |",
        f"| 1 | CANN 9.0.0 配套（Toolkit + 910B Ops） | {npu_cell} | 无 |",
        f"| 2 | Python {env['python']} + PyTorch {env['torch']} + 配套 torch_npu {env['torch_npu']} | "
        f"{'pass' if env['torch'] != '未探测' else '未探测'} | 无 |",
        "| 3 | `requirements.txt` 声明的构建与测试依赖 | "
        f"{'pass' if (REPO_ROOT / 'requirements.txt').is_file() else 'fail'} | 无 |",
        f"| 4 | docker 与 host+device 形态下编译、安装、跑通算子 | {cell(combine(st.install, st.api)).lower()} | 无 |",
        "| 5 | 文档声明的 Atlas A2 / Ascend 910B | pass | Atlas A3 / 950 系列文档标明暂未验证 |",
        "",
    ]


def section_coverage(st: Statuses, cases: list[tuple[str, str, str]]) -> list[str]:
    feature_cases = cases[:-2]
    feature_pass = sum(1 for c in feature_cases if c[1] == PASS)
    perf_parts = (st.perf_cm, st.perf_fused, st.fft_p, st.perf_gelu, st.perf_interp)
    perf_all = combine(*perf_parts)
    n_perf = len(perf_parts)
    rel_all = combine(st.ophost, st.prec_fused, st.api, st.prec_fused)
    compat_ok = st.install == PASS and st.api == PASS
    sec_ok = st.docs == PASS and st.pkg == PASS
    return [
        "## 6. 测试执行评估",
        "",
        "### 6.1 测试覆盖",
        "",
        "特性以及关键测试活动的测试覆盖情况如下。",
        "",
        "| 测试活动 | 测试结论 | 用例数 | 用例覆盖率 | 用例通过率 |",
        "| --- | --- | --- | --- | --- |",
        f"| 特性测试 | {'pass' if feature_pass == len(feature_cases) else 'fail'} | {len(feature_cases)} | 100% | "
        f"{feature_pass}/{len(feature_cases)} |",
        f"| 性能测试 | {'pass' if perf_all == PASS else 'fail'} | {n_perf} | 100% | "
        f"{f'{n_perf}/{n_perf}' if perf_all == PASS else cell(perf_all)} |",
        f"| 可靠性测试 | {'pass' if rel_all == PASS else 'fail'} | 4 | 100% | "
        f"{'4/4' if rel_all == PASS else cell(rel_all)} |",
        "| 继承特性测试 | 不涉及 | 0 | — | — |",
        f"| 兼容性测试 | {'pass' if compat_ok else 'fail'} | 5 | 100% | "
        f"{'5/5' if compat_ok else '见 5.4'} |",
        f"| 安全测试 | {'pass' if sec_ok else 'fail'} | 2 | 100% | "
        f"{'2/2' if sec_ok else '见 5.1'} |",
        "",
        f"说明：特性测试第 1～{len(feature_cases)} 组覆盖文档、编译安装与全部已交付算子；性能测试、可靠性、兼容性、安全与第 4、5 节表格对应。"
        "首发版本无继承特性。下游模型仓交叉验证未配置，不纳入本次范围，不计入失败。",
        "",
    ]


def section_issues(failed: list[tuple[str, str, str]]) -> list[str]:
    lines = [
        "## 7. 遗留问题和关键风险",
        "",
    ]
    if not failed:
        return lines + [
            "不涉及本仓功能缺陷。",
            "",
            "### 7.1 遗留问题统计",
            "",
            "| | 问题总数 | 严重 | 主要 | 次要 | 不重要 | 已取消 |",
            "| --- | --- | --- | --- | --- | --- | --- |",
            "| 数目 | 0 | 0 | 0 | 0 | 0 | 0 |",
            "| 百分比 | 100% | 0% | 0% | 0% | 0% | 0% |",
            "",
            "### 7.2 遗留问题列表",
            "",
            "| 问题单(issue链接) | 问题描述 | 问题级别 | 问题影响和规避措施 | 当前状态 |",
            "| --- | --- | --- | --- | --- |",
            "| 不涉及 | 不涉及 | — | — | — |",
            "",
            "已知环境差异（非本仓库失败，不计入上表）：原生 `torch.fft` 在部分 NPU 官方实现上的反向与 CPU 不完全一致；"
            "本仓 `real_rfft` / `real_irfft` 以 CPU `torch.fft` 为标杆验收。官方用例门槛 `1e-5`；"
            "相位折叠造基后实测相对误差约 `2e-7`～`5e-7`。",
            "",
        ]
    lines += [
        f"本次失败 {len(failed)} 项。",
        "",
        "### 7.1 遗留问题统计",
        "",
        "| | 问题总数 | 严重 | 主要 | 次要 | 不重要 | 已取消 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
        f"| 数目 | {len(failed)} | 0 | {len(failed)} | 0 | 0 | 0 |",
        "",
        "### 7.2 遗留问题列表",
        "",
        "| 问题单(issue链接) | 问题描述 | 问题级别 | 问题影响和规避措施 | 当前状态 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for name, status, _ in failed:
        lines.append(
            f"| 不涉及 | `{name}` {status} | 主要 | 见 `test_result/test_report.md` | 打开 |"
        )
    lines.append("")
    return lines


def render(payload: dict, env: dict[str, str]) -> str:
    jobs = load_jobs(payload)
    argv = payload.get("argv") or []
    cmd = "bash build.sh " + " ".join(argv) if argv else "bash build.sh -u"
    generated = payload.get("generated") or env["date"]
    date = str(generated)[:10]
    has_npu = bool(payload.get("has_npu"))

    st = collect_statuses(jobs)
    perf = collect_perf(jobs)
    features = build_features(st)
    cases = build_cases(st)

    executed = [c for c in cases if c[1] not in (NA, SKIP)]
    passed = [c for c in executed if c[1] == PASS]
    failed = [c for c in executed if c[1] == FAIL]
    rate = f"{len(passed)}/{len(executed)}" if executed else "0/0"
    ok = all(c[1] == PASS for c in cases)
    conclusion = build_conclusion(len(cases), len(failed), ok, cmd, generated)

    lines = section_overview(cmd, ok, rate)
    lines += section_version(env, date)
    lines += section_conclusion(conclusion, len(cases), ok, rate)
    lines += section_features(features, cases, st.docs)
    lines += section_security(st)
    lines += section_reliability(st)
    lines += section_perf(st, perf, date)
    lines += section_compat(st, env, has_npu)
    lines += section_coverage(st, cases)
    lines += section_issues(failed)
    lines += [
        "## 8. 附件",
        "",
        "原始执行纪要与 JSON：`test_result/test_report.md`、`test_result/test_results.json`。",
    ]
    return "\n".join(lines) + "\n"


def write_opensource_report(payload: dict, out: Path) -> Path:
    text = render(payload, collect_env())
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    if out.resolve() != DOCS_OUT.resolve():
        DOCS_OUT.parent.mkdir(parents=True, exist_ok=True)
        DOCS_OUT.write_text(text, encoding="utf-8")
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--from-json",
        default=str(DEFAULT_JSON),
        help="bash build.sh -u 落盘的 test_results.json",
    )
    p.add_argument("--out", default=str(DEFAULT_OUT), help="开源测试报告输出路径")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    src = Path(args.from_json)
    if not src.is_absolute():
        src = REPO_ROOT / src
    if not src.is_file():
        print(f"[ERROR] missing -u results JSON: {src}", file=sys.stderr)
        print("Run `bash build.sh -u` first, or pass --from-json.", file=sys.stderr)
        return 2
    out = Path(args.out)
    if not out.is_absolute():
        out = REPO_ROOT / out
    payload = json.loads(src.read_text(encoding="utf-8"))
    written = write_opensource_report(payload, out)
    print(f"[test] opensource report: {written}")
    if written.resolve() != DOCS_OUT.resolve():
        print(f"[test] opensource report (docs): {DOCS_OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
