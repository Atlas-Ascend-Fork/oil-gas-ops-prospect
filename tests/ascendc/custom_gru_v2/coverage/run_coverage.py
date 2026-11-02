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

"""Coverage collection script for CustomGRU v2.

Strategy:
- NPU environment: run all tests once (CPU fallback + NPU kernel paths both
  exercised — CPU fallback tests force CPU tensors explicitly), target 100%.
- CPU-only environment: NPU tests auto-skipped via @npu_only(), target 80%.

Uses `coverage run -m pytest` instead of `pytest --cov` to avoid import-time
conflicts between pytest-cov instrumentation and torch_npu monkey-patching.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main():
    testplan_dir = Path(__file__).resolve().parent.parent
    unit_dir = testplan_dir / "unit"
    integration_dir = testplan_dir / "integration"

    has_npu = False
    try:
        import torch_npu  # noqa: F401
        import torch
        if torch.npu.is_available() and torch.npu.device_count() > 0:
            try:
                from oil_gas_ops_prospect.custom_gru_v2 import _load_extension
                _load_extension()
                has_npu = True
            except (RuntimeError, ImportError):
                has_npu = False
    except (ImportError, AttributeError):
        has_npu = False

    if has_npu:
        print("[coverage] NPU detected — running full coverage (CPU fallback + NPU paths)")
        target = 100
    else:
        print("[coverage] No NPU — running CPU-only coverage")
        print("[coverage] Note: NPU-specific paths cannot be covered without NPU hardware.")
        print("[coverage] For 100% coverage, run on NPU environment.")
        target = 80

    run_cmd = [
        sys.executable, "-m", "coverage", "run",
        "-m", "pytest",
        str(unit_dir), str(integration_dir),
        "-v", "--tb=short",
    ]
    print(f"[coverage] Target: {target}%")
    print(f"[coverage] Run: {' '.join(run_cmd)}\n")
    result = subprocess.run(run_cmd, cwd=str(testplan_dir))

    if result.returncode != 0:
        print("\n[FAIL] Tests failed during coverage run.")
        return result.returncode

    report_cmd = [
        sys.executable, "-m", "coverage", "report",
        "--include=*/custom_gru_v2.py",
        "--show-missing",
        f"--fail-under={target}",
    ]
    print(f"\n[coverage] Report: {' '.join(report_cmd)}\n")
    result = subprocess.run(report_cmd, cwd=str(testplan_dir))

    html_cmd = [
        sys.executable, "-m", "coverage", "html",
        "--include=*/custom_gru_v2.py",
        "-d", str(testplan_dir / "coverage" / "htmlcov"),
    ]
    subprocess.run(html_cmd, cwd=str(testplan_dir))

    if result.returncode == 0:
        print(f"\n[PASS] Coverage target ({target}%) met.")
    else:
        print(f"\n[FAIL] Coverage below target ({target}%).")
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
