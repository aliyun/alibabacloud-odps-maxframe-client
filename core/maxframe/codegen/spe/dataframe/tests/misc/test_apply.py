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
import pytest

from ...... import dataframe as md
from ....core import SPECodeContext
from ...misc import DataFrameApplyAdapter


@pytest.fixture
def df1():
    return md.DataFrame(
        [
            [1, 10, 101],
            [9, 90, 909],
            [2, 20, 202],
        ],
        index=pd.MultiIndex.from_arrays(
            [[1, 9, 2], [11, 99, 22], [111, 999, 222]],
            names=("idx_0", "idx_1", "idx_2"),
        ),
        columns=list("ABC"),
    )


@pytest.fixture
def s1():
    return md.Series([5, 2, 3, 1], index=[5, 2, 3, 1], name="A")


def test_series_apply(s1):
    s = s1.apply(lambda x: x + 1, convert_dtype=True)
    adapter = DataFrameApplyAdapter()
    context = SPECodeContext()
    results = adapter.generate_code(s.op, context)
    assert len(context.get_udfs()) == 1
    f = context.get_udfs()[0]
    expected_results = [
        f"var_1_applied = var_0.apply(func={f.name}, convert_dtype=True)",
        "var_1_renamed = var_1_applied.rename('A')",
        "var_1 = var_1_renamed.astype(np.dtype('int64'))",
    ]
    assert results == expected_results


def test_apply_with_series_dtype_and_name(s1):
    s = s1.apply(
        lambda x: x + 1,
        convert_dtype=False,
        name="X",
        dtype="float32",
        output_type="series",
    )
    adapter = DataFrameApplyAdapter()
    context = SPECodeContext()
    results = adapter.generate_code(s.op, context)
    assert len(context.get_udfs()) == 1
    f = context.get_udfs()[0]
    expected_results = [
        f"var_1_applied = var_0.apply(func={f.name}, convert_dtype=False)",
        "var_1_renamed = var_1_applied.rename('X')",
        "var_1 = var_1_renamed.astype(np.dtype('float32'))",
    ]
    assert results == expected_results


def test_apply_with_dataframe_dtypes(df1):
    df = df1.apply(
        lambda x: pd.Series([str(x), x + 1]),
        convert_dtype=False,
        dtypes=pd.Series(["object", "float64"], index=["X", "Y"]),
        output_type="dataframe",
    )
    adapter = DataFrameApplyAdapter()
    context = SPECodeContext()
    results = adapter.generate_code(df.op, context)
    assert len(context.get_udfs()) == 1
    f = context.get_udfs()[0]
    expected_results = [
        f"var_1_applied = var_0.apply(func={f.name}, axis=0, raw=False, result_type=None)",
        "var_1_applied.columns = ['X', 'Y']",
        "var_1 = var_1_applied.astype({'X': np.dtype('object'), 'Y': np.dtype('float64')})",
    ]
    assert results == expected_results


def test_dataframe_apply_column(df1):
    df = df1.apply(lambda x: x + 1, axis=0)
    adapter = DataFrameApplyAdapter()
    context = SPECodeContext()
    results = adapter.generate_code(df.op, context)
    assert len(context.get_udfs()) == 1
    f = context.get_udfs()[0]
    expected_results = [
        f"var_1_applied = var_0.apply(func={f.name}, axis=0, raw=False, result_type=None)",
        "var_1_applied.columns = ['A', 'B', 'C']",
        "var_1 = var_1_applied.astype("
        "{'A': np.dtype('int64'),"
        " 'B': np.dtype('int64'),"
        " 'C': np.dtype('int64')})",
    ]
    assert results == expected_results


def test_dataframe_apply_row(df1):
    df = df1.apply(lambda x: x + 1, 1, True, "expand")
    adapter = DataFrameApplyAdapter()
    context = SPECodeContext()
    results = adapter.generate_code(df.op, context)
    assert len(context.get_udfs()) == 1
    f = context.get_udfs()[0]
    expected_results = [
        f"var_1_applied = var_0.apply(func={f.name}, axis=1, raw=True, result_type='expand')",
        "var_1_applied.columns = ['A', 'B', 'C']",
        "var_1 = var_1_applied.astype("
        "{'A': np.dtype('int64'),"
        " 'B': np.dtype('int64'),"
        " 'C': np.dtype('int64')})",
    ]
    assert results == expected_results
