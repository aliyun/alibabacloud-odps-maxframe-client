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

import pytest

from ..... import dataframe as md
from ...core import SPECodeContext
from ..reduction import (
    DataFrameAggregateAdapter,
    DataFrameKurtosisAdapter,
    DataFrameModeAdapter,
    DataFrameNuniqueAdapter,
    DataFrameReductionAdapter,
    DataFrameUniqueAdapter,
    DataFrameVarAdapter,
)


@pytest.fixture
def df1():
    return md.DataFrame(
        [[1, 10], [2, 20], [3, 30], [4, 40]], index=list("abcd"), columns=list("AB")
    )


@pytest.fixture
def s1():
    return md.Series([1, 2, 3, 4])


def test_sum(df1):
    df = df1.sum(skipna=False)
    results = DataFrameReductionAdapter().generate_code(df.op, SPECodeContext())
    expected_results = ["var_1 = var_0.sum(axis=0, skipna=False, min_count=0)"]
    assert results == expected_results


def test_kurtosis(df1):
    df = df1.kurt(bias=False, fisher=True)
    results = DataFrameKurtosisAdapter().generate_code(df.op, SPECodeContext())
    expected_results = [
        "var_1 = var_0.kurt(axis=0, skipna=True, bias=False, fisher=True)"
    ]
    assert results == expected_results


def test_dataframe_nunique(df1):
    df = df1.nunique()
    results = DataFrameNuniqueAdapter().generate_code(df.op, SPECodeContext())
    expected_results = ["var_1 = var_0.nunique(axis=0, dropna=True)"]
    assert results == expected_results


def test_series_nunique(s1):
    s = s1.nunique()
    results = DataFrameNuniqueAdapter().generate_code(s.op, SPECodeContext())
    expected_results = ["var_1 = var_0.nunique(dropna=True)"]
    assert results == expected_results


def test_var(df1):
    df = df1.var(ddof=2)
    results = DataFrameVarAdapter().generate_code(df.op, SPECodeContext())
    expected_results = ["var_1 = var_0.var(axis=0, skipna=True, ddof=2)"]
    assert results == expected_results


def test_unique(s1):
    df = md.unique(s1)
    context = SPECodeContext()
    results = DataFrameUniqueAdapter().generate_code(df.op, context)
    assert "pd" in context.imports["pandas"]
    expected_results = ["var_1 = pd.unique(var_0)"]
    assert results == expected_results


def test_median(df1):
    df = df1.median(skipna=False)
    results = DataFrameReductionAdapter().generate_code(df.op, SPECodeContext())
    expected_results = ["var_1 = var_0.median(axis=0, skipna=False)"]
    assert results == expected_results

    df = df1.median(axis=1, numeric_only=True)
    results = DataFrameReductionAdapter().generate_code(df.op, SPECodeContext())
    expected_results = ["var_1 = var_0.median(axis=1, skipna=True, numeric_only=True)"]
    assert results == expected_results


def test_mode(df1, s1):
    df = df1.mode(dropna=False, numeric_only=True)
    results = DataFrameModeAdapter().generate_code(df.op, SPECodeContext())
    expected_results = ["var_1 = var_0.mode(axis=0, numeric_only=True, dropna=False)"]
    assert results == expected_results

    s = s1.mode()
    results = DataFrameModeAdapter().generate_code(s.op, SPECodeContext())
    expected_results = ["var_1 = var_0.mode(dropna=True)"]
    assert results == expected_results


def test_aggregate(df1):
    df = df1.agg(["sum", "mean"])
    context = SPECodeContext()
    results = DataFrameAggregateAdapter().generate_code(df.op, context)
    expected_results = ["var_1 = var_0.agg(['sum', 'mean'], axis=0)"]
    assert results == expected_results
