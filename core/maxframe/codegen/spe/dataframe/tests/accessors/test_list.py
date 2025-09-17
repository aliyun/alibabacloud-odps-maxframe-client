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

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from ...... import dataframe as md
from ......lib.dtypes_extension import list_
from ......utils import ARROW_DTYPE_NOT_SUPPORTED
from ....core import SPECodeContext
from ...accessors.list_ import SeriesListMethodAdapter

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
def pd_df_1():
    return pd.DataFrame(
        {
            "A": pd.Series(
                [[1, 2, 3], [4, 5, 6, 7], None],
                index=[1, 2, 3],
                dtype=list_(pa.int32()),
            ),
            "B": pd.Series([1, 2, 3], index=[1, 2, 3], dtype=np.dtype("int64")),
        },
    )


@pytest.fixture
def md_df_1(pd_df_1):
    return md.DataFrame(pd_df_1)


def test_getitem(md_df_1, pd_df_1):
    s1 = md_df_1["A"].list.get(0)
    context = SPECodeContext()
    adapter = SeriesListMethodAdapter()
    results = adapter.generate_code(s1.op, context)

    expected_results = [
        """
def _inner_get(data):
    try:
        return data[0]
    except IndexError:
        if True:
            return None
        else:
            raise

var_1 = var_0.map(_inner_get, na_action="ignore").astype(const_0)
var_1.name = None
"""
    ]
    assert results == expected_results
    local_vars = _run_generated_code(results[0], context, pd_df_1["A"])
    expected_series = pd.Series(
        [1, 4, None],
        index=[1, 2, 3],
        dtype=pd.ArrowDtype(pa.int32()),
    )
    pd.testing.assert_series_equal(expected_series, local_vars["var_1"])


def test_getitem_with_index_error(md_df_1, pd_df_1):
    s1 = md_df_1["A"].list[3]
    context = SPECodeContext()
    adapter = SeriesListMethodAdapter()
    results = adapter.generate_code(s1.op, context)

    expected_results = [
        """
def _inner_get(data):
    try:
        return data[3]
    except IndexError:
        if False:
            return None
        else:
            raise

var_1 = var_0.map(_inner_get, na_action="ignore").astype(const_0)
var_1.name = None
"""
    ]
    assert results == expected_results
    with pytest.raises(IndexError):
        _run_generated_code(results[0], context, pd_df_1["A"])


def test_length(md_df_1, pd_df_1):
    s1 = md_df_1["A"].list.len()
    context = SPECodeContext()
    adapter = SeriesListMethodAdapter()
    results = adapter.generate_code(s1.op, context)

    expected_results = [
        """
var_1 = var_0.map(len, na_action="ignore").astype(const_0)
var_1.name = None
"""
    ]
    assert results == expected_results
    local_vars = _run_generated_code(results[0], context, pd_df_1["A"])
    expected_series = pd.Series(
        [3, 4, pd.NA], index=[1, 2, 3], name=None, dtype=pd.ArrowDtype(pa.int64())
    )
    pd.testing.assert_series_equal(expected_series, local_vars["var_1"])
