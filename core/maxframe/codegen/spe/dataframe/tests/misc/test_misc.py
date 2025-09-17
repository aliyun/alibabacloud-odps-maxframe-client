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
import pytest

from ...... import dataframe as md
from ....core import SPECodeContext
from ...misc import (
    CaseWhenOperatorAdapter,
    DataFrameAstypeAdapter,
    DataFrameCheckMonotonicAdapter,
    DataFrameCutAdapter,
    DataFrameDescribeAdapter,
    DataFrameDiffAdapter,
    DataFrameDropAdapter,
    DataFrameDuplicatedAdapter,
    DataFrameExplodeAdapter,
    DataFrameIsinAdapter,
    DataFrameMapAdapter,
    DataFrameMemoryUsageAdapter,
    DataFrameShiftAdapter,
    DataFrameToNumericAdapter,
)


@pytest.fixture
def df1():
    return md.DataFrame(
        np.random.randn(6, 4),
        index=md.Index(["1", "2", "3", "4", "5", "6"], name="test_idx"),
        columns=list("ABCD"),
    )


@pytest.fixture
def midx_df1():
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


def test_astype(df1):
    v1 = df1.astype({"A": "string"})
    context = SPECodeContext()
    results = DataFrameAstypeAdapter().generate_code(v1.op, context)
    assert results[0] == "var_1 = var_0.astype({'A': 'string'})"


def test_case_when(df1):
    s = df1.A.case_when([(df1.B > 50, df1.A + 10), (df1.C < 500, 3)])
    context = SPECodeContext()
    results = CaseWhenOperatorAdapter().generate_code(s.op, context)
    expected_results = [
        "var_1 = var_0.case_when(caselist=[(var_2, var_4), (var_3, 3)])"
    ]
    assert results == expected_results


def test_check_monotonic():
    context = SPECodeContext()

    v0 = md.Series([1, 2, 3, 4, 5, 6])
    v1 = v0.is_monotonic_increasing
    results = DataFrameCheckMonotonicAdapter().generate_code(v1.op, context)
    assert results[0] == "var_1 = var_0.is_monotonic_increasing"

    v2 = v0.is_monotonic_decreasing
    results = DataFrameCheckMonotonicAdapter().generate_code(v2.op, context)
    assert results[0] == "var_2 = var_0.is_monotonic_decreasing"


def test_cut():
    context = SPECodeContext()

    v0 = md.Series([1, 2, 3, 4, 5, 6])
    v1 = md.cut(v0, 3)
    results = DataFrameCutAdapter().generate_code(v1.op, context)
    assert results[0] == (
        "var_0 = pd.cut(var_1, 3, skip_none=None, right=True, labels=None, "
        "retbins=False, precision=3, include_lowest=False, duplicates='raise', "
        "ordered=True)"
    )


def test_describe(df1):
    df = df1.describe(include="all")
    context = SPECodeContext()
    results = DataFrameDescribeAdapter().generate_code(df.op, context)
    assert (
        results[0]
        == "var_1 = var_0.describe(percentiles=[0.25, 0.5, 0.75], include='all')"
    )


def test_diff(df1):
    df = df1.diff(periods=1)
    context = SPECodeContext()
    results = DataFrameDiffAdapter().generate_code(df.op, context)
    assert results[0] == "var_1 = var_0.diff(periods=1, axis=0)"


def test_drop(midx_df1):
    df = midx_df1.drop(index=(1, 11, 111), columns=["A", "C"], errors="ignore")
    adapter = DataFrameDropAdapter()
    context = SPECodeContext()
    results = adapter.generate_code(df.op, context)
    expected_results = [
        "var_1 = var_0.drop(index=(1, 11, 111), columns=['A', 'C'], level=None,"
        " errors='ignore')"
    ]
    assert results == expected_results

    del midx_df1["A"]
    results = adapter.generate_code(midx_df1.op, context)
    expected_results = [
        "var_2 = var_0.drop(index=None, columns='A', level=None, errors='raise')"
    ]
    assert results == expected_results


def test_duplicated(df1):
    df = df1.duplicated(subset=["A", "B"], keep=False)
    context = SPECodeContext()
    results = DataFrameDuplicatedAdapter().generate_code(df.op, context)
    assert results[0] == "var_1 = var_0.duplicated(subset=['A', 'B'], keep=False)"


def test_explode():
    context = SPECodeContext()

    v0 = md.Series([[1, 2, 3], [5, 6]])
    v1 = v0.explode(ignore_index=True)
    results = DataFrameExplodeAdapter().generate_code(v1.op, context)
    assert results[0] == "var_1 = var_0.explode(None, ignore_index=True)"


def test_isin(df1):
    filter_dict = {"A": [0, 1], "B": [2, 3], "E": [99, 100]}
    df = df1.isin(filter_dict)
    context = SPECodeContext()
    results = DataFrameIsinAdapter().generate_code(df.op, context)
    expected_results = [
        "var_1 = var_0.isin({'A': [0, 1], 'B': [2, 3], 'E': [99, 100]})"
    ]
    assert results == expected_results


def test_map(df1):
    v0 = md.Series([1, 2, 3, 4, 5, 6])
    v1 = v0.map(lambda x: x * 2 + 1, dtype="int64")
    context = SPECodeContext()
    results = DataFrameMapAdapter().generate_code(v1.op, context)
    assert len(context.get_udfs()) == 1
    f = context.get_udfs()[0]
    expected_results = [f"var_1 = var_0.map({f.name})"]
    assert results == expected_results


def test_memory_usage(df1):
    context = SPECodeContext()
    v1 = df1.memory_usage(index=False)
    results = DataFrameMemoryUsageAdapter().generate_code(v1.op, context)
    assert results[0] == "var_1 = var_0.memory_usage(index=False, deep=False)"


def test_shift(df1):
    df = df1.shift(periods=1)
    context = SPECodeContext()
    results = DataFrameShiftAdapter().generate_code(df.op, context)
    assert results[0] == "var_1 = var_0.shift(periods=1, axis=0)"


def test_to_numeric():
    v0 = md.Series([1, 2, 3, 4, 5, 6])
    v1 = md.to_numeric(v0, downcast="float")
    context = SPECodeContext()
    results = DataFrameToNumericAdapter().generate_code(v1.op, context)
    assert (
        results[0] == "var_1 = pd.to_numeric(var_0, errors='raise', downcast='float')"
    )
