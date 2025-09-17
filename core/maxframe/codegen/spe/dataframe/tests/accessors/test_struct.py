# Copyright 1999-2025 Alibaba Group Holding Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pandas as pd
import pyarrow as pa
import pytest

from ...... import dataframe as md
from ......utils import ARROW_DTYPE_NOT_SUPPORTED
from ....core import SPECodeContext
from ...accessors.struct_ import SeriesStructMethodAdapter

pytestmark = pytest.mark.skipif(
    ARROW_DTYPE_NOT_SUPPORTED, reason="Arrow Dtype is not supported"
)


def _run_generated_code(
    code: str, ctx: SPECodeContext, input_val: pd.DataFrame
) -> dict:
    local_vars = ctx.constants.copy()
    local_vars["var_0"] = input_val
    import_code = "import pandas as pd\nimport numpy as np\n"
    exec(import_code + code, local_vars, local_vars)
    return local_vars


@pytest.fixture
def series():
    version_type = pa.struct(
        [
            ("major", pa.int64()),
            ("minor", pa.int64()),
        ]
    )
    return md.Series(
        [
            {"version": {"major": 1, "minor": 5}, "project": "pandas"},
            {"version": {"major": 2, "minor": 1}, "project": "pandas"},
            {"version": {"major": 1, "minor": 26}, "project": "numpy"},
        ],
        dtype=pd.ArrowDtype(
            pa.struct([("version", version_type), ("project", pa.string())])
        ),
    )


def test_field(series):
    s1 = series.struct.field(["version", "minor"])
    context = SPECodeContext()
    adapter = SeriesStructMethodAdapter()
    results = adapter.generate_code(s1.op, context)

    expected_results = [
        """
var_1 = var_0.struct.field(['version', 'minor'])
"""
    ]
    assert results == expected_results
    local_vars = _run_generated_code(results[0], context, series.op.data)
    expected_series = pd.Series(
        [5, 1, 26], dtype=pd.ArrowDtype(pa.int64()), name="minor"
    )
    pd.testing.assert_series_equal(expected_series, local_vars["var_1"])
