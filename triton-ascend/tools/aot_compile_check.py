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

"""无 NPU 设备下对 Triton-Ascend 算子做 AOT 编译验证（IR 分支 + 二进制）。

用途
----
本仓库的开发/CI 环境通常没有 NPU 卡，`kernel[grid](...)` 无法执行，因此 Triton
算子的编译期回归（tl 语义错误、constexpr 分支写坏、tl.dot 形状不合法、UB/L0C
超限）没有门禁。本脚本把每个 kernel 的每个关注分支跑完整编译链路：

    TTIR  ->  Linalg(ttadapter)  ->  bishengir  ->  npubin (Ascend ELF)

编译成功即证明该分支能生成合法 kernel 二进制；失败即为编译期回归。

原理（三个关键点）
------------------
1. triton-ascend 后端的 target 可由 ``TRITON_ASCEND_ARCH`` 提供，不必查卡
   （见 ``triton/backends/ascend/utils.py: get_ascend_arch_from_env``）。
2. 但 bishengir 阶段 ``linalg_to_bin_enable_npu_compile_A2_A3`` 里硬编码调用
   ``NPUUtils().get_arch()`` -> ``rtGetSocVersion()``，无卡会失败并报
   ``rtGetSocVersion failed, 0x7bc80``。这里把它打桩为返回 env arch。
3. 走 ``triton.compile(ASTSource(...))`` 而不是 ``kernel[grid](...)``，绕开
   driver 的 device / stream 查询。

用法
----
    # 容器内，仓库根目录
    python3 triton-ascend/tools/aot_compile_check.py
    python3 triton-ascend/tools/aot_compile_check.py --filter irfft
    python3 triton-ascend/tools/aot_compile_check.py --dump-dir /tmp/ir

退出码 0 表示全部分支编译通过；triton-ascend 后端缺失时先打印
``[FAIL]`` 定位与安装说明，再原样抛出（退出码非 0，安装方式见
``docs/zh/context/quick_install.md``）。
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TRITON_ROOT = REPO_ROOT / "triton-ascend"
for _p in (TRITON_ROOT / "real_fft", TRITON_ROOT / "complex_reciprocal"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

DEFAULT_ARCH = "Ascend910B1"
os.environ.setdefault("TRITON_ASCEND_ARCH", DEFAULT_ARCH)
os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/triton_cache_aot")
# 无卡镜像可能装有 torch_npu 但缺 CANN 运行环境，torch 后端自动加载会抛错
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
ARCH = os.environ["TRITON_ASCEND_ARCH"]


def import_triton():
    """延迟导入 triton。

    必须在上面的 ``TRITON_ASCEND_ARCH`` / ``TRITON_CACHE_DIR`` 落定之后再导入
    （后端与缓存目录在导入期就会读环境变量），所以这些 import 不能提到模块顶部。
    返回 ``(triton, GPUTarget, ASTSource)``。
    """
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler.compiler import ASTSource

    return triton, GPUTarget, ASTSource


def kernel_of(module, name: str):
    """按名字取模块内的 ``@triton.jit`` kernel。

    kernel 是各算子模块的实现细节（下划线前缀），本工具是它们的编译期门禁，
    需要逐个 kernel 编译。用 ``getattr`` 按名取，避免在模块外直接书写受保护成员。
    """
    kernel = getattr(module, name, None)
    if kernel is None:
        raise AttributeError(f"{module.__name__} 没有 kernel {name}")
    return kernel


def stub_soc_runtime_queries() -> None:
    """把依赖 NPU runtime 的 arch / 核数查询打桩，使无卡环境可编译。"""
    from triton.backends.ascend import driver

    driver.NPUUtils.get_arch = lambda self: ARCH  # type: ignore[method-assign]
    driver.NPUUtils.get_aicore_num = lambda self: 20  # type: ignore[method-assign]
    driver.NPUUtils.get_aivector_core_num = lambda self: 40  # type: ignore[method-assign]


def ensure_triton_ascend_available() -> None:
    """triton / 昇腾后端可用性预检，缺失时打印易读说明后原样抛出。

    PyPI 社区版 triton 没有 ``triton.backends.ascend`` 后端，不能替代
    Triton-Ascend（见 docs/zh/context/quick_install.md）。此时本脚本无法
    执行编译链路；先打印定位与修复说明，再重新抛出原始异常，保持
    与直接崩溃一致的失败语义（退出码非 0）。
    """
    try:
        import triton  # noqa: F401
    except ModuleNotFoundError:
        print("[FAIL] AOT 编译检查无法执行：未安装 triton。")
        print("       Triton 算子的编译期回归需在装有 Triton-Ascend 的环境执行，")
        print("       安装方式见 docs/zh/context/quick_install.md。")
        raise
    try:
        import triton.backends.ascend  # noqa: F401
    except ModuleNotFoundError as exc:
        print(f"[FAIL] AOT 编译检查无法执行：当前 triton {getattr(triton, '__version__', '?')}"
              f" 无昇腾后端（缺失模块: {exc.name}）。")
        print("       PyPI 社区版 triton 不能替代 Triton-Ascend。安装方式：")
        print("       pip uninstall -y triton triton-ascend && \\")
        print("       pip install triton-ascend==3.2.1 "
              "--extra-index-url=https://mirrors.huaweicloud.com/ascend/repos/pypi")
        raise


def build_src(ast_source_cls, kernel, ptr_types: list[str], meta: dict):
    """按 kernel.arg_names 顺序拼 signature；meta 里的名字视为 constexpr。"""
    signature: dict[str, str] = {}
    constants: dict[str, object] = {}
    ptr_iter = iter(ptr_types)
    for name in kernel.arg_names:
        if name in meta:
            signature[name] = "constexpr"
            constants[name] = meta[name]
        else:
            signature[name] = next(ptr_iter)
    leftover = list(ptr_iter)
    if leftover:
        raise ValueError(f"{kernel.__name__}: 多余的指针类型 {leftover}")
    return ast_source_cls(fn=kernel, signature=signature, constants=constants)


def p(n: int, dtype: str = "*fp32") -> list[str]:
    return [dtype] * n


def _real_fft_cases(fft) -> list[tuple[str, object, list[str], dict]]:
    """real_fft 的两条降级路径：摊平 2D GEMM 与沿轴 strided GEMM。"""
    out: list[tuple[str, object, list[str], dict]] = []

    # ---- real_fft: flatten 2D GEMM 分支（先摊平成 rows x n，再与傅里叶基做双 GEMM）----
    flat = dict(rows=16384, n=128, freq_len=65, num_blocks=128,
                tile_m=128, tile_n=128, tile_k=128)
    out += [
        ("real_fft/rfft.fwd  [flatten 2D GEMM]",
         kernel_of(fft, "_rfft_matmul_kernel"), p(5), flat),
        ("real_fft/irfft.fwd [flatten 2D GEMM]",
         kernel_of(fft, "_irfft_matmul_kernel"), p(5), flat),
        ("real_fft/rfft.bwd  [flatten 2D GEMM]",
         kernel_of(fft, "_rfft_backward_matmul_kernel"), p(5), flat),
        ("real_fft/irfft.bwd [flatten 2D GEMM]",
         kernel_of(fft, "_irfft_backward_matmul_kernel"), p(5), flat),
    ]

    # ---- real_fft: axis strided GEMM 分支（沿任意轴直算，免 permute）----
    ax_k = dict(n_lead=128, n=128, freq_len=65, n_tail=128, num_blocks=128,
                tile_b=64, tile_f=64, tile_k=64)
    ax_t = dict(n_lead=128, n=128, freq_len=65, n_tail=128, num_blocks=128,
                tile_b=64, tile_f=64, tile_t=64)
    out += [
        ("real_fft/rfft.fwd  [axis strided GEMM]",
         kernel_of(fft, "_rfft_axis_matmul_kernel"), p(5), ax_k),
        ("real_fft/rfft.bwd  [axis strided GEMM]",
         kernel_of(fft, "_rfft_axis_backward_kernel"), p(5), ax_k),
        ("real_fft/irfft.fwd [axis strided GEMM]",
         kernel_of(fft, "_irfft_axis_matmul_kernel"), p(5), ax_t),
        ("real_fft/irfft.bwd [axis strided GEMM]",
         kernel_of(fft, "_irfft_axis_backward_kernel"), p(5), ax_t),
    ]

    return out


def _complex_reciprocal_cases(cr) -> list[tuple[str, object, list[str], dict]]:
    """complex_reciprocal 的三条 kernel：交错前向、实虚分离前向、反向。"""
    out: list[tuple[str, object, list[str], dict]] = []

    fwd_ptrs = p(2) + ["i32"]
    out += [
        ("complex_reciprocal/recip.fwd [stable]",
         kernel_of(cr, "_complex_reciprocal_kernel"), fwd_ptrs,
         dict(block_size=2048, index64=False, stable=True)),
        ("complex_reciprocal/recip.fwd [stable, index64]",
         kernel_of(cr, "_complex_reciprocal_kernel"), fwd_ptrs,
         dict(block_size=2048, index64=True, stable=True)),
        ("complex_reciprocal/recip.fwd [naive]",
         kernel_of(cr, "_complex_reciprocal_kernel"), fwd_ptrs,
         dict(block_size=2048, index64=False, stable=False)),
    ]

    of_ptrs = p(3) + ["i32", "fp32", "fp32"]
    for xr_t, xi_t in ((True, True), (True, False), (False, True)):
        tag = f"xr={'tensor' if xr_t else 'scalar'},xi={'tensor' if xi_t else 'scalar'}"
        out.append((
            f"complex_reciprocal/of.fwd   [{tag}]",
            kernel_of(cr, "_complex_reciprocal_of_kernel"), of_ptrs,
            dict(block_size=2048, index64=False, stable=True, xr_is_tensor=xr_t, xi_is_tensor=xi_t),
        ))
    out.append((
        "complex_reciprocal/of.fwd   [tensor,tensor naive index64]",
        kernel_of(cr, "_complex_reciprocal_of_kernel"), of_ptrs,
        dict(block_size=2048, index64=True, stable=False, xr_is_tensor=True, xi_is_tensor=True),
    ))

    grad_ptrs = p(5) + ["i32"]
    out += [
        ("complex_reciprocal/grad    [interleaved]",
         kernel_of(cr, "_complex_reciprocal_grad_kernel"), grad_ptrs,
         dict(block_size=2048, index64=False, interleaved_output=True, write_gxi=False)),
        ("complex_reciprocal/grad    [planar gxr+gxi]",
         kernel_of(cr, "_complex_reciprocal_grad_kernel"), grad_ptrs,
         dict(block_size=2048, index64=False, interleaved_output=False, write_gxi=True)),
        ("complex_reciprocal/grad    [planar gxr, index64]",
         kernel_of(cr, "_complex_reciprocal_grad_kernel"), grad_ptrs,
         dict(block_size=2048, index64=True, interleaved_output=False, write_gxi=False)),
    ]

    return out


def cases() -> list[tuple[str, object, list[str], dict]]:
    """(用例名, kernel, 指针类型列表, constexpr 字典)，覆盖各算子的主要 IR 分支。"""
    import triton_complex_reciprocal as cr
    import triton_fft_real as fft

    return _real_fft_cases(fft) + _complex_reciprocal_cases(cr)


def _slugify(name: str) -> str:
    return name.replace("/", ".").replace(" ", "").replace("[", "_").replace("]", "")


def _dump_stage(dump: Path | None, slug: str, stage: str, body) -> str | None:
    """记录一级 IR 的大小，并在需要时落盘。非 str/bytes 的产物跳过。"""
    if isinstance(body, str):
        if dump:
            (dump / f"{slug}.{stage}").write_text(body)
    elif isinstance(body, bytes):
        if dump:
            (dump / f"{slug}.{stage}").write_bytes(body)
    else:
        return None
    return f"{len(body)}B"


def _report_stages(compiled, name: str, dump: Path | None) -> None:
    slug = _slugify(name)
    stages = {}
    for stage, body in compiled.asm.items():
        size = _dump_stage(dump, slug, stage, body)
        if size is not None:
            stages[stage] = size
    meta = compiled.metadata
    info = " ".join(
        f"{k}={getattr(meta, k)}" for k in ("mix_mode", "parallel_mode") if hasattr(meta, k)
    )
    print(f"[ OK ] {name}")
    print(f"       {stages}  {info}")


def compile_cases(selected, dump: Path | None) -> list[str]:
    """逐个编译用例，返回失败用例名列表。"""
    triton_mod, gpu_target_cls, ast_source_cls = import_triton()
    target = gpu_target_cls("npu", ARCH, 0)
    stub_soc_runtime_queries()

    print(f"triton={triton_mod.__version__}  target={target}")
    if dump:
        print(f"IR dump -> {dump}")
    print()

    ok = 0
    failures: list[str] = []
    for name, kernel, ptrs, meta in selected:
        try:
            src = build_src(ast_source_cls, kernel, ptrs, meta)
            _report_stages(triton_mod.compile(src, target=target), name, dump)
            ok += 1
        except Exception as exc:  # noqa: BLE001
            print(f"[FAIL] {name}: {type(exc).__name__}: {str(exc)[:400]}")
            traceback.print_exc(limit=2)
            failures.append(name)

    print(f"\n合计: OK={ok} FAIL={len(failures)}")
    for name in failures:
        print(f"  FAILED  {name}")
    return failures


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dump-dir", default="", help="导出各级 IR 到该目录（默认不导出）")
    ap.add_argument("--filter", default="", help="只跑名字包含该子串的用例")
    ap.add_argument("--list", action="store_true", help="只列出用例，不编译")
    return ap.parse_args(argv)


def main() -> int:
    args = parse_args()

    ensure_triton_ascend_available()

    selected = [c for c in cases() if not args.filter or args.filter in c[0]]
    if args.list:
        for name, *_ in selected:
            print(name)
        return 0
    if not selected:
        print(f"没有用例匹配 --filter {args.filter!r}", file=sys.stderr)
        return 2

    dump = Path(args.dump_dir) if args.dump_dir else None
    if dump:
        dump.mkdir(parents=True, exist_ok=True)

    return 1 if compile_cases(selected, dump) else 0


if __name__ == "__main__":
    sys.exit(main())
