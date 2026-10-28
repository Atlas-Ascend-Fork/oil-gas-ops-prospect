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

"""开源报告 13～16 组对应的轻量门禁：文档、出包入口、安装痕迹、公开接口冒烟。"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

# 文档测试清单：与 docs/README.md 导航及仓库根交付文档对齐。
# 每项为 (相对路径, 标题, 必须出现的关键词, 开源报告「测试内容」)。
DOC_CHECKS: list[tuple[str, str, tuple[str, ...], str]] = [
    (
        "README.md",
        "项目入口",
        ("编译与安装", "bash build.sh --pkg", "bash build.sh --install", "bash build.sh -u"),
        "仓名与两类后端交付物；三种构建入口；测试入口 `-u`",
    ),
    (
        "LICENSE",
        "许可证",
        ("Apache License", "Version 2.0"),
        "Apache License 2.0 全文",
    ),
    (
        "SECURITY.md",
        "安全声明",
        ("安全声明", "公网地址声明", "opp/vendors/customize"),
        "运行/构建安全建议；公网地址声明表",
    ),
    (
        "CONTRIBUTING.md",
        "贡献指南",
        ("提交前本地自检", "bash build.sh -u", "oil-gas-engineering"),
        "贡献流程、提交前自检与 PR/CI 要求",
    ),
    (
        "CHANGELOG.md",
        "变更记录",
        ("变更记录", "## [Unreleased]"),
        "语义化版本变更记录，含 Unreleased",
    ),
    (
        "docs/README.md",
        "文档导航",
        ("快速入门", "算子列表", "接口列表", "编译参数说明"),
        "入门/使用/开发/社区文档索引可跳转",
    ),
    (
        "docs/QUICKSTART.md",
        "快速入门",
        ("CANN", "9.0.0", "bash build.sh --pkg", "bash build.sh --install"),
        "环境前置、三种构建入口、最短调用路径",
    ),
    (
        "docs/zh/context/quick_install.md",
        "环境部署",
        ("Toolkit", "Triton-Ascend", "3.2.1", "bash build.sh --install"),
        "CANN Toolkit/Ops、Triton-Ascend 安装与环境变量",
    ),
    (
        "docs/zh/context/dir_structure.md",
        "目录结构",
        ("ascendc/", "triton-ascend/", "tests/", "build.sh"),
        "仓库各目录职责与 `build.sh` 入口说明",
    ),
    (
        "docs/zh/op_list.md",
        "算子列表",
        (
            "ComplexMul",
            "ComplexMulGrad",
            "FusedBiasSoftmax",
            "FusedSoftmaxGrad",
            "real_rfft",
            "real_irfft",
            "complex_reciprocal",
            "complex_exp",
            "interpolate",
            "Conv3d",
            "conv3d",
            "Conv3DGelu",
            "Conv3DGeluGrad",
            "数学定义",
        ),
        "12 个已交付算子的数学定义、输入输出与约束",
    ),
    (
        "docs/zh/api_list.md",
        "接口列表",
        (
            "complex_mul",
            "fused_bias_softmax",
            "fused_softmax_grad",
            "attention_core",
            "real_rfft",
            "real_irfft",
            "complex_reciprocal",
            "complex_exp",
            "interpolate",
            "Conv3d",
            "conv3d",
            "conv3d_gelu",
            "aclnnComplexMul",
            "aclnnConv3dGeluCustom",
        ),
        "Python 公开接口与 aclnn 两段式入口、环境变量",
    ),
    (
        "docs/zh/invocation/quick_op_invocation.md",
        "算子调用",
        ("bash build.sh --install", "complex_mul", "real_rfft", "aclnnComplexMul"),
        "Python / aclnn 端到端调用示例",
    ),
    (
        "docs/zh/context/build.md",
        "编译参数说明",
        ("bash build.sh --pkg", "bash build.sh --install", "-u", "ascendc/build.sh"),
        "根目录三种构建入口与 `-u` 测试选项",
    ),
    (
        "docs/zh/develop/operator_development_guide.md",
        "算子开发指南",
        ("op_host", "op_kernel", "manifest", "bash build.sh --install"),
        "新增 AscendC / Triton 算子的目录、注册、编译、封装与验收步骤",
    ),
    (
        "docs/zh/develop/precision_acceptance_template.md",
        "精度验收模板",
        ("精度验收", "ophost tiling UT", "bash build.sh -u --st"),
        "PR 精度报告模板与验收项",
    ),
    (
        "docs/zh/debug/op_debug_prof.md",
        "调试与性能分析",
        ("调试", "NPU", "bash build.sh -u"),
        "AI Core Host/Kernel 调试与 msProf 上板采集",
    ),
]


def _fail(msg: str) -> int:
    print(f"FAIL {msg}")
    return 1


def check_docs() -> int:
    missing: list[str] = []
    content_fail: list[str] = []
    for rel, _title, tokens, _content in DOC_CHECKS:
        path = REPO_ROOT / rel
        if not path.is_file():
            missing.append(rel)
            continue
        text = path.read_text(encoding="utf-8")
        absent = [token for token in tokens if token not in text]
        if absent:
            content_fail.append(f"{rel} missing {absent}")
    if missing:
        return _fail("missing " + ", ".join(missing))
    if content_fail:
        return _fail("; ".join(content_fail))
    names = [item[0] for item in DOC_CHECKS]
    print("PASS docs present and content ok: " + ", ".join(names))
    return 0


def _glob_count(directory: Path, pattern: str) -> int:
    return len(list(directory.glob(pattern))) if directory.is_dir() else 0


def check_pkg() -> int:
    # 直接执行 build.sh（自带 shebang），不经 bash 转发，避免把用户输入交给 shell。
    proc = subprocess.run(
        [str(REPO_ROOT / "build.sh"), "-h"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    text = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0:
        return _fail(f"build.sh -h rc={proc.returncode}")
    for token in ("--soc", "--ops", "-j", "--pkg", "--install"):
        if token not in text:
            return _fail(f"build.sh -h missing {token}")
    runs = _glob_count(REPO_ROOT / "output", "**/*_opp_*.run")
    wheels = _glob_count(REPO_ROOT / "output" / "wheels", "oil_gas_ops_prospect-*.whl")
    print(f"PASS build.sh -h options ok; opp_run={runs} wheel={wheels}")
    return 0


def check_install() -> int:
    cann = Path(os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/cann-9.0.0"))
    vendor = cann / "opp" / "vendors" / "customize"
    lib = vendor / "op_api" / "lib"
    if not vendor.is_dir():
        return _fail(f"OPP vendor missing: {vendor}")
    if not lib.is_dir():
        return _fail(f"OPP op_api lib missing: {lib}")
    print(f"PASS OPP vendor present: {vendor}")
    return 0


def check_api() -> int:
    spec = __import__("importlib.util").util.spec_from_file_location(
        "local_pkg", REPO_ROOT / "scripts" / "local_pkg.py"
    )
    if spec is None or spec.loader is None:
        return _fail("cannot load local_pkg.py")
    mod = __import__("importlib.util").util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.activate(REPO_ROOT)

    os.environ["ACLNN_CACHE_LIMIT"] = "0"
    os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    import torch
    from oil_gas_ops_prospect.complex_mul import complex_mul
    from oil_gas_ops_prospect.fused_bias_softmax import (
        attention_core,
        fused_bias_softmax,
        fused_softmax_grad,
    )
    from oil_gas_ops_prospect.conv3d_gelu import conv3d_gelu
    from oil_gas_ops_prospect.conv3d_custom import Conv3d
    from oil_gas_ops_prospect.conv3d_custom.functional import conv3d
    from oil_gas_ops_prospect.complex_exp import complex_exp
    from oil_gas_ops_prospect.complex_reciprocal import complex_reciprocal
    from oil_gas_ops_prospect.interpolate import interpolate
    from oil_gas_ops_prospect.real_fft import real_irfft, real_rfft

    zc = torch.randn(2, 3) + 1j * torch.randn(2, 3)
    if not torch.equal(complex_reciprocal(zc), 1 / zc):
        return _fail("complex_reciprocal cpu fallback")
    if complex_reciprocal(zc.real, -0.25).shape != zc.shape:
        return _fail("complex_reciprocal parts shape")
    ze = torch.randn(2, 3) + 1j * torch.randn(2, 3)
    if not torch.equal(complex_exp(ze), torch.exp(ze)):
        return _fail("complex_exp cpu fallback")

    xi = torch.randn(1, 2, 4, 5, 6)
    if interpolate(xi, size=(8, 10, 12), mode="trilinear").shape != (1, 2, 8, 10, 12):
        return _fail("interpolate shape")

    xc = torch.randn(1, 3, 6, 6, 6, requires_grad=True)
    wc = torch.randn(4, 3, 3, 3, 3, requires_grad=True)
    oc = conv3d(xc, wc, None, 1, 1, 1, 1)
    gc = torch.autograd.grad(oc.sum(), (xc, wc))
    if oc.shape != (1, 4, 6, 6, 6) or gc[0].shape != xc.shape or gc[1].shape != wc.shape:
        return _fail("conv3d fwd/bwd shape")
    mod = Conv3d(3, 4, 3, padding=1)
    if mod(torch.randn(1, 3, 5, 5, 5)).shape != (1, 4, 5, 5, 5):
        return _fail("Conv3d module shape")
    gx = torch.randn(1, 2, 4, 4, 4)
    gw = torch.randn(2, 2, 1, 1, 1)
    gy = conv3d_gelu(gx, gw)
    if gy.shape != (1, 2, 4, 4, 4):
        return _fail("conv3d_gelu shape")
    try:
        conv3d_gelu(gx, gw, groups=2)
    except ValueError:
        pass
    else:
        return _fail("conv3d_gelu groups!=1 should raise")

    ur = torch.randn(4, 8)
    ui, vr, vi = torch.randn_like(ur), torch.randn_like(ur), torch.randn_like(ur)
    wr, wi = complex_mul(ur, ui, vr, vi)
    if wr.shape != ur.shape or wi.shape != ur.shape:
        return _fail("complex_mul shape")

    x = torch.randn(2, 2, 8, 16)
    y = fused_bias_softmax(x, None, None)
    if y.shape != x.shape:
        return _fail("fused_bias_softmax shape")
    probs = torch.softmax(torch.randn(2, 2, 8, 16), -1)
    go = torch.randn(2, 2, 8, 8)
    values = torch.randn(2, 2, 16, 8)
    gx = fused_softmax_grad(probs, go, values)
    if gx.shape != probs.shape:
        return _fail("fused_softmax_grad shape")
    q = torch.randn(1, 2, 4, 8)
    k = torch.randn(1, 2, 4, 8)
    v = torch.randn(1, 2, 4, 8)
    out = attention_core(q, k, v, None, None)
    if out.shape != q.shape:
        return _fail("attention_core shape")

    sig = torch.randn(2, 32)
    xr, xi = real_rfft(sig, dim=-1, norm="ortho")
    rec = real_irfft(xr, xi, n=32, dim=-1, norm="ortho")
    if rec.shape != sig.shape:
        return _fail("real_irfft shape")
    print(
        "PASS public API CPU fallback: complex_mul fused_bias_softmax "
        "fused_softmax_grad attention_core real_rfft real_irfft complex_reciprocal "
        "complex_exp interpolate Conv3d conv3d conv3d_gelu"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("check", choices=("docs", "pkg", "install", "api"))
    args = p.parse_args(argv)
    dispatch = {
        "docs": check_docs,
        "pkg": check_pkg,
        "install": check_install,
        "api": check_api,
    }
    check = dispatch.get(args.check)
    if check is None:
        return _fail(f"unknown check: {args.check}")
    return check()


if __name__ == "__main__":
    sys.exit(main())
