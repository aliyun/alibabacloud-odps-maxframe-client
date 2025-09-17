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
from ......lib.dtypes_extension import dict_
from ......utils import ARROW_DTYPE_NOT_SUPPORTED
from ....core import SPECodeContext
from ...accessors.dict_ import SeriesDictMethodAdapter

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
                [[("k1", 1), ("k2", 2)], [("k1", 11)], None],
                index=[1, 2, 3],
                dtype=dict_(pa.string(), pa.int32()),
            ),
            "B": pd.Series([1, 2, 3], index=[1, 2, 3], dtype=np.dtype("int64")),
        },
    )


@pytest.fixture
def md_df_1(pd_df_1):
    return md.DataFrame(pd_df_1)


def test_getitem(md_df_1, pd_df_1):
    s1 = md_df_1["A"].dict["k1"]
    context = SPECodeContext()
    adapter = SeriesDictMethodAdapter()
    results = adapter.generate_code(s1.op, context)

    expected_results = [
        """
def _inner_get(data):
    found = False
    for tup in data:
        if tup[0] == 'k1':
            found = True
            return tup[1]
    if not found:
        if False:
            return None
        else:
            raise KeyError('k1')

var_1 = var_0.map(_inner_get, na_action="ignore").astype(const_0)
var_1.name = 'k1'
"""
    ]
    assert results == expected_results
    local_vars = _run_generated_code(results[0], context, pd_df_1["A"])
    expected_series = pd.Series(
        [1, 11, None],
        index=[1, 2, 3],
        name="k1",
        dtype=pd.ArrowDtype(pa.int32()),
    )
    pd.testing.assert_series_equal(expected_series, local_vars["var_1"])


@pytest.mark.skipif(ARROW_DTYPE_NOT_SUPPORTED, reason="Arrow Dtype is not supported")
def test_getitem_with_default_value(md_df_1, pd_df_1):
    s1 = md_df_1["A"].dict.get("k2", 9)
    context = SPECodeContext()
    adapter = SeriesDictMethodAdapter()
    results = adapter.generate_code(s1.op, context)

    expected_results = [
        """
def _inner_get(data):
    found = False
    for tup in data:
        if tup[0] == 'k2':
            found = True
            return tup[1]
    if not found:
        if True:
            return 9
        else:
            raise KeyError('k2')

var_1 = var_0.map(_inner_get, na_action="ignore").astype(const_0)
var_1.name = 'k2'
"""
    ]
    assert results == expected_results
    local_vars = _run_generated_code(results[0], context, pd_df_1["A"])
    expected_series = pd.Series(
        [2, 9, None],
        index=[1, 2, 3],
        name="k2",
        dtype=pd.ArrowDtype(pa.int32()),
    )
    pd.testing.assert_series_equal(expected_series, local_vars["var_1"])


def test_getitem_with_key_error(md_df_1, pd_df_1):
    s1 = md_df_1["A"].dict["k2"]
    context = SPECodeContext()
    adapter = SeriesDictMethodAdapter()
    results = adapter.generate_code(s1.op, context)

    expected_results = [
        """
def _inner_get(data):
    found = False
    for tup in data:
        if tup[0] == 'k2':
            found = True
            return tup[1]
    if not found:
        if False:
            return None
        else:
            raise KeyError('k2')

var_1 = var_0.map(_inner_get, na_action="ignore").astype(const_0)
var_1.name = 'k2'
"""
    ]
    assert results == expected_results
    with pytest.raises(KeyError):
        _run_generated_code(results[0], context, pd_df_1["A"])


def test_setitem(md_df_1, pd_df_1):
    s1 = md_df_1["A"]
    s1.dict["k2"] = 9
    context = SPECodeContext()
    adapter = SeriesDictMethodAdapter()
    results = adapter.generate_code(s1.op, context)

    expected_results = [
        """
def _inner_set(row):
    found = False
    value = list()
    for tup in row:
        if tup[0] == 'k2':
            value.append((tup[0], 9))
            found = True
        else:
            value.append(tup)
    if not found:
        value.append(('k2', 9))
    return value

var_1 = var_0.map(_inner_set, na_action="ignore").astype(const_0)
"""
    ]
    assert results == expected_results
    local_vars = _run_generated_code(results[0], context, pd_df_1["A"])
    expected_series = pd.Series(
        [
            [("k1", 1), ("k2", 9)],
            [("k1", 11), ("k2", 9)],
            None,
        ],
        index=[1, 2, 3],
        name="A",
        dtype=pd.ArrowDtype(pa.map_(pa.string(), pa.int32())),
    )
    pd.testing.assert_series_equal(expected_series, local_vars["var_1"])


@pytest.mark.skipif(ARROW_DTYPE_NOT_SUPPORTED, reason="Arrow Dtype is not supported")
def test_length(md_df_1, pd_df_1):
    s1 = md_df_1["A"].dict.len()
    context = SPECodeContext()
    adapter = SeriesDictMethodAdapter()
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
        [2, 1, pd.NA], index=[1, 2, 3], name=None, dtype=pd.ArrowDtype(pa.int64())
    )
    pd.testing.assert_series_equal(expected_series, local_vars["var_1"])


def test_remove_with_ignore_key_error(md_df_1, pd_df_1):
    s1 = md_df_1["A"].dict.remove("k2", ignore_key_error=True)
    context = SPECodeContext()
    adapter = SeriesDictMethodAdapter()
    results = adapter.generate_code(s1.op, context)

    expected_results = [
        """
def _inner_remove(value):
    row = list()
    found = False
    for tup in value:
        if tup[0] == 'k2':
            found = True
        else:
            row.append(tup)
    if not found and not True:
        raise KeyError('k2')
    return row

var_1 = var_0.map(_inner_remove, na_action="ignore").astype(const_0)
"""
    ]
    assert results == expected_results
    local_vars = _run_generated_code(results[0], context, pd_df_1["A"])
    expected_series = pd.Series(
        [[("k1", 1)], [("k1", 11)], None],
        index=[1, 2, 3],
        name="A",
        dtype=pd.ArrowDtype(pa.map_(pa.string(), pa.int32())),
    )
    pd.testing.assert_series_equal(expected_series, local_vars["var_1"])


@pytest.mark.skipif(ARROW_DTYPE_NOT_SUPPORTED, reason="Arrow Dtype is not supported")
def test_remove_with_key_error(md_df_1, pd_df_1):
    s1 = md_df_1["A"].dict.remove("k2", ignore_key_error=False)
    context = SPECodeContext()
    adapter = SeriesDictMethodAdapter()
    results = adapter.generate_code(s1.op, context)

    expected_results = [
        """
def _inner_remove(value):
    row = list()
    found = False
    for tup in value:
        if tup[0] == 'k2':
            found = True
        else:
            row.append(tup)
    if not found and not False:
        raise KeyError('k2')
    return row

var_1 = var_0.map(_inner_remove, na_action="ignore").astype(const_0)
"""
    ]
    assert results == expected_results
    with pytest.raises(KeyError):
        _run_generated_code(results[0], context, pd_df_1["A"])


def test_contains(md_df_1, pd_df_1):
    s1 = md_df_1["A"].dict.contains("k2")
    context = SPECodeContext()
    adapter = SeriesDictMethodAdapter()
    results = adapter.generate_code(s1.op, context)

    expected_results = [
        """
var_1 = var_0.map(lambda x: any('k2' in tup[0] for tup in x), na_action="ignore").astype(const_0)
var_1.name = None
"""
    ]
    assert results == expected_results
    local_vars = _run_generated_code(results[0], context, pd_df_1["A"])
    expected_series = pd.Series(
        [True, False, None],
        index=[1, 2, 3],
        name=None,
        dtype=pd.ArrowDtype(pa.bool_()),
    )
    pd.testing.assert_series_equal(expected_series, local_vars["var_1"])
