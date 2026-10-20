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

import sys
import unittest
from pathlib import Path

ASCENDC_ROOT = Path(__file__).resolve().parents[1] / "ascendc"
sys.path.insert(0, str(ASCENDC_ROOT))

from pybind_extensions import _filter_extension_dirs  # noqa: E402


EXTENSIONS = [Path("complex_mul"), Path("fused_bias_softmax")]


class PybindExtensionSelectionTests(unittest.TestCase):
    def test_unfiltered_build_keeps_all_pybind_families(self):
        self.assertEqual(_filter_extension_dirs(EXTENSIONS, None), EXTENSIONS)

    def test_incomplete_complex_mul_family_is_not_built(self):
        self.assertEqual(_filter_extension_dirs(EXTENSIONS, {"complex_mul"}), [])

    def test_complete_complex_mul_family_keeps_its_extension(self):
        self.assertEqual(
            _filter_extension_dirs(EXTENSIONS, {"complex_mul", "complex_mul_grad"}),
            [Path("complex_mul")],
        )

    def test_incomplete_softmax_family_is_not_built(self):
        self.assertEqual(_filter_extension_dirs(EXTENSIONS, {"fused_bias_softmax"}), [])

    def test_complete_softmax_family_keeps_its_extension(self):
        self.assertEqual(
            _filter_extension_dirs(
                EXTENSIONS, {"fused_bias_softmax", "fused_softmax_grad"}
            ),
            [Path("fused_bias_softmax")],
        )

    def test_unknown_operator_selection_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "missing_operator"):
            _filter_extension_dirs(EXTENSIONS, {"missing_operator"})


if __name__ == "__main__":
    unittest.main()
