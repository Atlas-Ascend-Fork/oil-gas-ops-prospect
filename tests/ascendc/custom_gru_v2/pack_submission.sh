#!/bin/bash
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

# Pack CustomGRUV2 submission zip for review.
# Includes ALL files: operator (host/kernel), Python wrapper, C++ bindings, testplan.
# Each item is existence-checked; missing items cause sys.exit(1).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR" && git rev-parse --show-toplevel)"
COMMIT="$(cd "$REPO_ROOT" && git rev-parse HEAD)"
OUT="/tmp/oil-gas-ops-prospect-${COMMIT}-ascendc-operators-custom_gru_v2.zip"
cd "$REPO_ROOT"
python3 -c "
import zipfile, os, sys
zippath = '${OUT}'
items = [
    'ascendc/operators/custom_gru_v2/',
    'ascendc/pybind/custom_gru_v2.py',
    'ascendc/pybind/csrc/custom_gru_v2/bindings.cpp',
]
missing = []
for d in items:
    if not os.path.exists(d):
        missing.append(d)
if missing:
    for m in missing:
        print(f'ERROR: item not found: {m}', file=sys.stderr)
    sys.exit(1)
with zipfile.ZipFile(zippath, 'w', zipfile.ZIP_DEFLATED) as zf:
    for d in items:
        if os.path.isdir(d):
            for root, dirs2, files in os.walk(d):
                for f in files:
                    fp = os.path.join(root, f)
                    if '__pycache__' not in fp and '.pytest_cache' not in fp \
                            and '.coverage' not in os.path.basename(fp) \
                            and 'htmlcov/' not in fp \
                            and 'results_prof/' not in fp \
                            and 'results_sanitizer/' not in fp \
                            and 'results_msopst/' not in fp \
                            and 'mindstudio_sanitizer_log/' not in fp \
                            and ('.log' not in os.path.basename(fp) \
                                 or 'npu_reports/' in fp):
                        zf.write(fp)
        elif os.path.isfile(d):
            zf.write(d)
        else:
            print(f'ERROR: unreachable — item not found after pre-check: {d}', file=sys.stderr)
            sys.exit(1)
print(f'Created {zippath} ({os.path.getsize(zippath)} bytes)')
with zipfile.ZipFile(zippath, 'r') as zf:
    for info in zf.infolist():
        print(f'  {info.file_size:>8}  {info.filename}')
"
