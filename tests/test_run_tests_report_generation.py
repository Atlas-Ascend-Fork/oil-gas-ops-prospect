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

from __future__ import annotations

import json
from pathlib import Path

from scripts import run_tests


def _prepare_runner(tmp_path: Path, monkeypatch) -> Path:
    root = tmp_path / "repo"
    (root / "tests").mkdir(parents=True)
    (root / "tests" / "suites.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(run_tests, "REPO_ROOT", root)
    monkeypatch.setattr(run_tests, "SUITES_PATH", root / "tests" / "suites.json")
    monkeypatch.setattr(run_tests, "npu_available", lambda: False)
    monkeypatch.setattr(run_tests, "run_suites", lambda _ctx, _wanted, _results: None)
    return root


def test_report_generation_failure_fails_even_when_old_report_exists(tmp_path, monkeypatch):
    root = _prepare_runner(tmp_path, monkeypatch)
    stale_report = tmp_path / "community-report.md"
    stale_report.write_text("previous successful report\n", encoding="utf-8")
    monkeypatch.setattr(run_tests, "generate_opensource_report", lambda _src, _out: 1)

    rc = run_tests.main([
        "--suites=",
        f"--results-json={tmp_path / 'results.json'}",
        f"--report={tmp_path / 'execution.md'}",
        f"--opensource-report={stale_report}",
    ])

    assert rc == 1
    assert stale_report.read_text(encoding="utf-8") == "previous successful report\n"
    assert (root / "test_result" / "test_report.md").is_file()
    assert json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))["jobs"] == []


def test_successful_report_generation_keeps_zero_exit(tmp_path, monkeypatch):
    _prepare_runner(tmp_path, monkeypatch)
    report = tmp_path / "community-report.md"

    def generate(_src: Path, out: Path) -> int:
        out.write_text("fresh report\n", encoding="utf-8")
        return 0

    monkeypatch.setattr(run_tests, "generate_opensource_report", generate)
    rc = run_tests.main([
        "--suites=",
        f"--results-json={tmp_path / 'results.json'}",
        f"--report={tmp_path / 'execution.md'}",
        f"--opensource-report={report}",
    ])

    assert rc == 0
    assert report.read_text(encoding="utf-8") == "fresh report\n"
