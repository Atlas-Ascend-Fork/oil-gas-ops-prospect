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

"""统一测试编排：UT / ST / 精度 / 性能，并写出 Markdown 报告。"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SUITES_PATH = REPO_ROOT / "tests" / "suites.json"

PASS = "PASS"
FAIL = "FAIL"
SKIP = "SKIP"


LOG_CAP = 80000


def clip_log(text: str) -> str:
    perf_lines = [ln for ln in text.splitlines() if "[perf]" in ln]
    body = text if len(text) <= LOG_CAP else text[-LOG_CAP:]
    if not perf_lines:
        return body
    block = "\n".join(perf_lines)
    if block in body:
        return body
    return block + "\n\n" + body


@dataclass
class JobResult:
    name: str
    suite: str
    status: str
    duration_s: float
    detail: str = ""
    log_tail: str = ""
    log: str = ""


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def npu_available() -> bool:
    code = """
import sys
try:
    import torch
    import torch_npu  # noqa: F401
except Exception:
    sys.exit(2)
sys.exit(0 if (hasattr(torch, "npu") and torch.npu.is_available()) else 2)
"""
    proc = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True)
    if proc.returncode == 0:
        return True
    # torch 探测失败时，只要本机有 davinci 节点就仍尝试上板套件。
    return Path("/dev/davinci0").exists()


def run_cmd(cmd: list[str], cwd: Path | None = None, env: dict[str, str] | None = None,
            timeout: int | None = None) -> tuple[int, str]:
    merged = os.environ.copy()
    if env:
        merged.update(env)
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd or REPO_ROOT),
            env=merged,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    except FileNotFoundError as exc:
        return 127, f"command not found: {cmd[0]} ({exc})"
    chunks: list[str] = []
    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            chunks.append(line)
        code = proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        return 124, "".join(chunks).strip() + "\n[timeout]"
    return code, "".join(chunks).strip()


def tail(text: str, n: int = 40) -> str:
    lines = text.splitlines()
    return "\n".join(lines[-n:])


def make_result(name: str, suite: str, status: str, duration_s: float,
                detail: str = "", out: str = "") -> JobResult:
    return JobResult(name, suite, status, duration_s, detail, tail(out), clip_log(out))


def extract_notes(text: str) -> str:
    keys = []
    for pat in (
        r"\[  PASSED  \] \d+ tests",
        r"\[  FAILED  \] \d+ tests",
        r"mismatch_ratio=[0-9.]+",
        r"rel_err[^\n]*",
        r"speedup=[0-9.]+x",
        r"\[perf\][^\n]*",
        r"\[vs complex_mul\][^\n]*",
        r"ALL PASS",
        r"\d+ FAILS",
        r"all \d+ ST cases pass",
        r"\[SKIP\][^\n]*",
        r"PASS",
        r"FAIL[^\n]*",
    ):
        found = re.findall(pat, text)
        keys.extend(found[:3])
    # 去重保序
    seen = set()
    uniq = []
    for item in keys:
        if item not in seen:
            seen.add(item)
            uniq.append(item)
    return "; ".join(uniq[:8])


def filter_ops(items: list[dict], ops: set[str] | None) -> list[dict]:
    if not ops:
        return items
    return [it for it in items if it.get("op") in ops]


def run_ophost(jobs: int, gtest_filter: str, noexec: bool, cov: bool) -> JobResult:
    build_dir = REPO_ROOT / "ascendc" / "_work" / "ophost_ut"
    src = REPO_ROOT / "tests" / "ascendc" / "ophost"
    cache = build_dir / "CMakeCache.txt"
    if cache.is_file():
        cached_src = ""
        for line in cache.read_text(errors="ignore").splitlines():
            if line.startswith("CMAKE_HOME_DIRECTORY:"):
                cached_src = line.split("=", 1)[-1].strip()
                break
        if cached_src and Path(cached_src) != src:
            print(f"[test] ophost cache src changed ({cached_src} -> {src}), wiping {build_dir}")
            shutil.rmtree(build_dir)
    cmake_flags = ["-DCMAKE_BUILD_TYPE=Release"]
    if cov:
        cmake_flags += ["-DCMAKE_CXX_FLAGS=--coverage", "-DCMAKE_EXE_LINKER_FLAGS=--coverage"]
    t0 = time.perf_counter()
    code, out = run_cmd(["cmake", "-S", str(src), "-B", str(build_dir), *cmake_flags])
    if code != 0:
        return make_result("ophost_ut", "ut", FAIL, time.perf_counter() - t0, "cmake failed", out)
    code, out2 = run_cmd(["cmake", "--build", str(build_dir), "-j", str(jobs)])
    out += "\n" + out2
    if code != 0:
        return make_result("ophost_ut", "ut", FAIL, time.perf_counter() - t0, "build failed", out)
    if noexec:
        return make_result("ophost_ut", "ut", SKIP, time.perf_counter() - t0, "--noexec: compiled only", out)
    exe = build_dir / "ophost_ut"
    xml = REPO_ROOT / "test_result" / "ophost_ut.xml"
    xml.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(exe), f"--gtest_output=xml:{xml}"]
    if gtest_filter:
        cmd.append(f"--gtest_filter={gtest_filter}")
    code, out3 = run_cmd(cmd)
    out += "\n" + out3
    status = PASS if code == 0 else FAIL
    return make_result("ophost_ut", "ut", status, time.perf_counter() - t0,
                       extract_notes(out3) or status, out)


def run_triton(cov: bool, noexec: bool) -> JobResult:
    t0 = time.perf_counter()
    if noexec:
        return make_result("triton_cpu", "ut", SKIP, 0.0, "--noexec")
    args = [sys.executable, "-m", "pytest", str(REPO_ROOT / "tests" / "triton"), "-q"]
    if cov:
        probe = subprocess.run([sys.executable, "-c", "import pytest_cov"], capture_output=True)
        if probe.returncode == 0:
            args += ["--cov", "--cov-report=term-missing"]
    code, out = run_cmd(args)
    return make_result("triton_cpu", "ut", PASS if code == 0 else FAIL, time.perf_counter() - t0,
                       extract_notes(out) or ("passed" if code == 0 else "pytest failed"), out)


def run_composite(cov: bool, noexec: bool) -> JobResult:
    t0 = time.perf_counter()
    if noexec:
        return make_result("composite_cpu", "ut", SKIP, 0.0, "--noexec")
    args = [sys.executable, "-m", "pytest", str(REPO_ROOT / "tests" / "composite"), "-q"]
    if cov:
        probe = subprocess.run([sys.executable, "-c", "import pytest_cov"], capture_output=True)
        if probe.returncode == 0:
            args += ["--cov", "--cov-report=term-missing"]
    code, out = run_cmd(args)
    return make_result("composite_cpu", "ut", PASS if code == 0 else FAIL, time.perf_counter() - t0,
                       extract_notes(out) or ("passed" if code == 0 else "pytest failed"), out)


def run_aot(noexec: bool) -> JobResult:
    t0 = time.perf_counter()
    if noexec:
        return make_result("triton_aot", "ut", SKIP, 0.0, "--noexec")
    code, out = run_cmd([sys.executable, str(REPO_ROOT / "triton-ascend" / "tools" / "aot_compile_check.py")])
    return make_result("triton_aot", "ut", PASS if code == 0 else FAIL, time.perf_counter() - t0,
                       extract_notes(out) or status_word(code), out)


def status_word(code: int) -> str:
    return PASS if code == 0 else FAIL


def run_script(name: str, suite: str, cmd: list[str], cwd: Path | None = None) -> JobResult:
    t0 = time.perf_counter()
    code, out = run_cmd(cmd, cwd=cwd)
    return make_result(name, suite, PASS if code == 0 else FAIL, time.perf_counter() - t0,
                       extract_notes(out) or status_word(code), out)


def maybe_skip_npu(name: str, suite: str, require_npu: bool, has_npu: bool) -> JobResult | None:
    if has_npu:
        return None
    if require_npu:
        return JobResult(name, suite, FAIL, 0.0, "NPU required but not available")
    return JobResult(name, suite, SKIP, 0.0, "no NPU, skipped")


def write_report(path: Path, results: list[JobResult], argv: list[str], has_npu: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    passed = sum(1 for r in results if r.status == PASS)
    failed = sum(1 for r in results if r.status == FAIL)
    skipped = sum(1 for r in results if r.status == SKIP)
    total_s = sum(r.duration_s for r in results)
    generated = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %z")
    lines = [
        "# 测试报告",
        "",
        f"- 生成时间：{generated}",
        f"- 命令：`bash build.sh {' '.join(argv)}`" if argv else "- 命令：`scripts/run_tests.py`",
        f"- NPU：{'可用' if has_npu else '不可用（NPU 套件将 skip 或失败）'}",
        f"- 合计：{len(results)} 项，**通过 {passed}**，**失败 {failed}**，跳过 {skipped}，耗时 {total_s:.1f}s",
        "",
        "## 总览",
        "",
        "| 套件 | 用例 | 结果 | 耗时 (s) | 摘要 |",
        "| --- | --- | --- | ---: | --- |",
    ]
    for r in results:
        mark = {"PASS": "✅ PASS", "FAIL": "❌ FAIL", "SKIP": "⏭️ SKIP"}[r.status]
        detail = r.detail.replace("|", "\\|")
        lines.append(f"| {r.suite} | `{r.name}` | {mark} | {r.duration_s:.1f} | {detail} |")
    lines += ["", "## 明细", ""]
    for r in results:
        lines += [f"### {r.name}", "", f"- 套件：`{r.suite}`", f"- 结果：**{r.status}**", f"- 耗时：{r.duration_s:.1f}s"]
        if r.detail:
            lines.append(f"- 摘要：{r.detail}")
        if r.log_tail:
            lines += ["", "```text", r.log_tail[:8000], "```", ""]
        else:
            lines.append("")
    if failed:
        lines += ["## 结论", "", f"**失败 {failed} 项。** 请根据上方明细修复后再跑。", ""]
    else:
        lines += ["## 结论", "", "全部执行项通过（跳过项不计入失败）。", ""]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_results_json(path: Path, results: list[JobResult], argv: list[str], has_npu: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "argv": argv,
        "has_npu": has_npu,
        "jobs": [
            {
                "name": r.name,
                "suite": r.suite,
                "status": r.status,
                "duration_s": round(r.duration_s, 3),
                "detail": r.detail,
                "log_tail": r.log_tail,
                "log": r.log,
            }
            for r in results
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def generate_opensource_report(results_json: Path, out: Path) -> int:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "opensource_report.py"),
        "--from-json",
        str(results_json),
        "--out",
        str(out),
    ]
    proc = subprocess.run(cmd, cwd=REPO_ROOT)
    return proc.returncode


def _attach_flag_values(argv: list[str], flags: tuple[str, ...]) -> list[str]:
    """把 ``--flag -u`` 收成 ``--flag=-u``，避免 argparse 把以 ``-`` 开头的值当成新选项。"""
    out: list[str] = []
    i = 0
    while i < len(argv):
        cur = argv[i]
        if cur in flags and i + 1 < len(argv) and "=" not in cur:
            nxt = argv[i + 1]
            if nxt.startswith("-"):
                out.append(f"{cur}={nxt}")
                i += 2
                continue
        out.append(cur)
        i += 1
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suites", default="ophost,triton,composite,aot,st,precision,perf,examples,fft,gate",
                   help="comma-separated: ophost,triton,composite,aot,st,precision,perf,examples,fft,gate")
    p.add_argument("--ops", default="", help="filter operator names, comma-separated")
    p.add_argument("--jobs", type=int, default=os.cpu_count() or 8)
    p.add_argument("--gtest-filter", default="")
    p.add_argument("--noexec", action="store_true")
    p.add_argument("--cov", action="store_true")
    p.add_argument("--require-npu", action="store_true",
                   help="NPU suites fail instead of skip when no device")
    p.add_argument("--report", default="", help="markdown report path")
    p.add_argument("--results-json", default="",
                   help="persist job results JSON (default: test_result/test_results.json)")
    p.add_argument("--opensource-report", default="",
                   help="community opensource markdown generated from this -u JSON")
    p.add_argument("--orig-argv", default="", help="original build.sh argv for the report header")
    raw = list(sys.argv[1:] if argv is None else argv)
    return p.parse_args(_attach_flag_values(
        raw,
        ("--orig-argv", "--gtest-filter", "--report", "--results-json",
         "--opensource-report", "--suites", "--ops", "--jobs"),
    ))


@dataclass
class SuiteContext:
    """各套件共用的运行参数。"""

    args: argparse.Namespace
    catalog: dict
    ops: set[str]
    has_npu: bool

    @property
    def noexec(self) -> bool:
        return bool(self.args.noexec)

    @property
    def require_npu(self) -> bool:
        return bool(self.args.require_npu)


@dataclass
class ReportPaths:
    report: Path
    latest: Path
    results_json: Path
    opensource: Path | None


def _repo_path(value: str, default: Path) -> Path:
    """相对路径按仓库根解析，空值回落到 default。"""
    path = Path(value) if value else default
    return path if path.is_absolute() else REPO_ROOT / path


def resolve_report_paths(args: argparse.Namespace) -> ReportPaths:
    report = _repo_path(args.report, REPO_ROOT / "test_result" / f"test_report_{now_stamp()}.md")
    results_json = _repo_path(args.results_json, REPO_ROOT / "test_result" / "test_results.json")
    opensource = _repo_path(args.opensource_report, REPO_ROOT) if args.opensource_report else None
    return ReportPaths(report, REPO_ROOT / "test_result" / "test_report.md", results_json, opensource)


def ops_mismatch(name: str, suite: str, label: str, items: list[dict], ops: set[str]) -> JobResult | None:
    if ops and not items:
        return JobResult(name, suite, FAIL, 0.0, f"no {label} match --ops={','.join(sorted(ops))}")
    return None


def run_st_case(ctx: SuiteContext, item: dict) -> JobResult:
    name = item["name"]
    if ctx.noexec:
        return JobResult(name, "st", SKIP, 0.0, "--noexec")
    skip = maybe_skip_npu(name, "st", ctx.require_npu, ctx.has_npu)
    if skip is not None:
        print(f"[test] ST {name} {skip.status}")
        return skip
    print(f"[test] ST {name}")
    return run_script(name, "st", ["bash", str(REPO_ROOT / item["run"])])


def run_example_case(ctx: SuiteContext, item: dict) -> JobResult:
    name = f"example_{item['op']}"
    if ctx.noexec:
        return JobResult(name, "examples", SKIP, 0.0, "--noexec")
    skip = maybe_skip_npu(name, "examples", ctx.require_npu, ctx.has_npu)
    if skip is not None:
        return skip
    print(f"[test] example {item['op']}")
    return run_script(name, "examples", ["bash", str(REPO_ROOT / item["run"])])


def run_npu_python_case(ctx: SuiteContext, item: dict, mode: str, built_ops: set[str]) -> JobResult:
    name = f"{item['op']}_{mode}"
    if ctx.noexec:
        return JobResult(name, mode, SKIP, 0.0, "--noexec")
    skip = maybe_skip_npu(name, mode, ctx.require_npu, ctx.has_npu)
    if skip is not None:
        print(f"[test] {name} {skip.status}")
        return skip
    cmd = [sys.executable, str(REPO_ROOT / item["script"]), "--mode", mode]
    if item["op"] not in built_ops:
        cmd.append("--build-ext")
        built_ops.add(item["op"])
    print(f"[test] {name}")
    return run_script(name, mode, cmd, cwd=REPO_ROOT / Path(item["script"]).parent)


def run_npu_script_case(ctx: SuiteContext, name: str, suite: str, cmd: list[str]) -> JobResult:
    if ctx.noexec:
        return JobResult(name, suite, SKIP, 0.0, "--noexec")
    skip = maybe_skip_npu(name, suite, ctx.require_npu, ctx.has_npu)
    if skip is not None:
        print(f"[test] {name} {skip.status}")
        return skip
    print(f"[test] {name}")
    return run_script(name, suite, cmd)


# 各 collect_* 直接往 results 里追加，异常中断时 finally 仍能把已完成的 job 写进报告。
def collect_st(ctx: SuiteContext, results: list[JobResult]) -> None:
    items = filter_ops(ctx.catalog["st"], ctx.ops or None)
    mismatch = ops_mismatch("st", "st", "ST cases", items, ctx.ops)
    if mismatch is not None:
        results.append(mismatch)
    for item in items:
        results.append(run_st_case(ctx, item))


def collect_examples(ctx: SuiteContext, results: list[JobResult]) -> None:
    items = filter_ops(ctx.catalog["examples"], ctx.ops or None)
    mismatch = ops_mismatch("examples", "examples", "examples", items, ctx.ops)
    if mismatch is not None:
        results.append(mismatch)
    for item in items:
        results.append(run_example_case(ctx, item))


def collect_npu_python(ctx: SuiteContext, modes: list[str], results: list[JobResult]) -> None:
    built_ops: set[str] = set()
    for mode in modes:
        items = filter_ops(ctx.catalog["npu_python"], ctx.ops or None)
        mismatch = ops_mismatch(mode, mode, "NPU python cases", items, ctx.ops)
        if mismatch is not None:
            results.append(mismatch)
        for item in items:
            results.append(run_npu_python_case(ctx, item, mode, built_ops))


def collect_fft(ctx: SuiteContext, results: list[JobResult]) -> None:
    fft_dir = REPO_ROOT / "tests" / "triton" / "real_fft"
    verify = fft_dir / "verify_triton_fft_real_128_3d.py"
    bench = fft_dir / "bench_opensource_perf.py"
    fft_jobs = [
        ("fft_npu_verify", [sys.executable, str(verify), "--size", "128", "--rounds", "1"]),
        ("fft_npu_verify_5d", [sys.executable, str(verify), "--run-spectral-3d-smoke", "--rounds", "1"]),
        ("fft_npu_perf", [sys.executable, str(bench)]),
    ]
    for name, cmd in fft_jobs:
        results.append(run_npu_script_case(ctx, name, "fft", cmd))


def collect_gate(ctx: SuiteContext, results: list[JobResult]) -> None:
    gates = REPO_ROOT / "tests" / "opensource_gates.py"
    for name, check in (
        ("docs_gate", "docs"),
        ("pkg_gate", "pkg"),
        ("install_gate", "install"),
        ("api_smoke", "api"),
    ):
        if ctx.noexec:
            results.append(JobResult(name, "gate", SKIP, 0.0, "--noexec"))
            continue
        print(f"[test] {name}")
        results.append(run_script(name, "gate", [sys.executable, str(gates), check]))


def run_suites(ctx: SuiteContext, wanted: list[str], results: list[JobResult]) -> None:
    if "ophost" in wanted:
        print("[test] ophost UT")
        results.append(run_ophost(ctx.args.jobs, ctx.args.gtest_filter, ctx.noexec, ctx.args.cov))
    if "triton" in wanted:
        print("[test] triton CPU pytest")
        results.append(run_triton(ctx.args.cov, ctx.noexec))
    if "composite" in wanted:
        print("[test] composite CPU pytest")
        results.append(run_composite(ctx.args.cov, ctx.noexec))
    if "aot" in wanted:
        print("[test] triton AOT")
        results.append(run_aot(ctx.noexec))
    if "st" in wanted:
        collect_st(ctx, results)
    if "examples" in wanted:
        collect_examples(ctx, results)
    collect_npu_python(ctx, [m for m in ("precision", "perf") if m in wanted], results)
    if "fft" in wanted and not ctx.ops:
        collect_fft(ctx, results)
    if "gate" in wanted:
        collect_gate(ctx, results)


def finalize_reports(paths: ReportPaths, results: list[JobResult], args: argparse.Namespace,
                     has_npu: bool) -> None:
    orig = args.orig_argv.split() if args.orig_argv else sys.argv[1:]
    write_results_json(paths.results_json, results, orig, has_npu)
    print(f"[test] results json: {paths.results_json}")
    write_report(paths.report, results, orig, has_npu)
    paths.latest.parent.mkdir(parents=True, exist_ok=True)
    paths.latest.write_text(paths.report.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"[test] report: {paths.report}")
    print(f"[test] report (latest): {paths.latest}")
    if paths.opensource is None:
        return
    gen_rc = generate_opensource_report(paths.results_json, paths.opensource)
    if gen_rc != 0:
        print(f"[test] opensource report failed rc={gen_rc}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    wanted = [s.strip() for s in args.suites.split(",") if s.strip()]
    ops = {x.strip() for x in args.ops.replace(",", " ").split() if x.strip() and x.strip() != "all"}
    ctx = SuiteContext(args=args, catalog=json.loads(SUITES_PATH.read_text()), ops=ops,
                       has_npu=npu_available())
    paths = resolve_report_paths(args)
    results: list[JobResult] = []

    try:
        run_suites(ctx, wanted, results)
    finally:
        finalize_reports(paths, results, args, ctx.has_npu)

    failed = sum(1 for r in results if r.status == FAIL)
    print(f"[test] done: pass={sum(r.status==PASS for r in results)} "
          f"fail={failed} skip={sum(r.status==SKIP for r in results)}")
    if paths.opensource is not None and not paths.opensource.is_file():
        return 1
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
