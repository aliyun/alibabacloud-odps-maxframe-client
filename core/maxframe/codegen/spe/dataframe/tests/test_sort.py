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

from ..... import dataframe as md
from ....core import EngineAcceptance
from ...core import SPECodeContext
from ..sort import DataFrameSortIndexAdapter, DataFrameSortValuesAdapter


@pytest.fixture
def multi_index():
    arrays = [
        ["a1", "a2", "a3"],
        ["b1", "b2", "b3"],
        ["c1", "c2", "c3"],
        ["d1", "d2", "d3"],
        ["e1", "e2", "e3"],
    ]
    return pd.MultiIndex.from_arrays(
        arrays, names=("idx_a", "idx_b", "idx_c", "idx_d", "idx_e")
    )


@pytest.fixture
def df1():
    return md.DataFrame(
        np.random.randn(6, 4),
        index=md.Index(["1", "2", "3", "4", "5", "6"], name="test_idx"),
        columns=list("ABCD"),
    )


@pytest.fixture
def df2(multi_index):
    return md.DataFrame(
        np.random.randn(3, 4),
        index=multi_index,
        columns=list("ABCD"),
    )


@pytest.fixture
def s1():
    return md.Series([1, 2, 3], index=list("abc"))


@pytest.fixture
def s2(multi_index):
    return md.Series([1, 2, 3], index=multi_index)


def test_series_sort_values(s1):
    s = s1.sort_values(na_position="last", ascending=False, inplace=False)
    adapter = DataFrameSortValuesAdapter()
    context = SPECodeContext()
    assert adapter.accepts(s.op) == EngineAcceptance.ACCEPT
    results = adapter.generate_code(s.op, context)
    expected_results = [
        "var_1 = var_0.sort_values(axis=0, ascending=False, inplace=False,"
        " na_position='last', ignore_index=False)"
    ]
    assert results == expected_results


def test_dataframe_sort_by_values_single_column(df1):
    s = df1.sort_values(by="A", na_position="first", ascending=True)
    adapter = DataFrameSortValuesAdapter()
    context = SPECodeContext()
    assert adapter.accepts(s.op) == EngineAcceptance.ACCEPT
    results = adapter.generate_code(s.op, context)
    expected_results = [
        "var_1 = var_0.sort_values(axis=0, ascending=True, inplace=False,"
        " na_position='first', ignore_index=False, by=['A'])"
    ]
    assert results == expected_results


def test_dataframe_sort_by_values_multi_columns(df1):
    df1.sort_values(by=["A", "B"], inplace=True)
    adapter = DataFrameSortValuesAdapter()
    context = SPECodeContext()
    assert adapter.accepts(df1.op) == EngineAcceptance.ACCEPT
    results = adapter.generate_code(df1.op, context)
    expected_results = [
        "var_1 = var_0.sort_values(axis=0, ascending=True, inplace=True,"
        " na_position='last', ignore_index=False, by=['A', 'B'])"
    ]
    assert results == expected_results


def test_series_sort_index_by_single(s2):
    s = s2.sort_index(
        level=1, na_position="first", sort_remaining=False, ignore_index=True
    )
    adapter = DataFrameSortIndexAdapter()
    context = SPECodeContext()
    assert adapter.accepts(s.op) == EngineAcceptance.ACCEPT
    results = adapter.generate_code(s.op, context)
    expected_results = [
        "var_1 = var_0.sort_index(axis=0, ascending=True, inplace=False,"
        " na_position='first', ignore_index=True, level=[1], sort_remaining=False)"
    ]
    assert results == expected_results


def test_series_sort_index_by_multiple(s2):
    s = s2.sort_index(level=[1, "idx_d"])
    adapter = DataFrameSortIndexAdapter()
    context = SPECodeContext()
    assert adapter.accepts(s.op) == EngineAcceptance.ACCEPT
    results = adapter.generate_code(s.op, context)
    expected_results = [
        "var_1 = var_0.sort_index(axis=0, ascending=True, inplace=False,"
        " na_position='last', ignore_index=False, level=[1, 'idx_d'],"
        " sort_remaining=True)"
    ]
    assert results == expected_results


def test_dataframe_sort_index_by_single(df2):
    s = df2.sort_index(level="idx_b", na_position="first")
    adapter = DataFrameSortIndexAdapter()
    context = SPECodeContext()
    assert adapter.accepts(s.op) == EngineAcceptance.ACCEPT
    results = adapter.generate_code(s.op, context)
    expected_results = [
        "var_1 = var_0.sort_index(axis=0, ascending=True, inplace=False,"
        " na_position='first', ignore_index=False, level=['idx_b'],"
        " sort_remaining=True)"
    ]
    assert results == expected_results


def test_dataframe_sort_index_by_all(df2):
    s = df2.sort_index()
    adapter = DataFrameSortIndexAdapter()
    context = SPECodeContext()
    assert adapter.accepts(s.op) == EngineAcceptance.ACCEPT
    results = adapter.generate_code(s.op, context)
    expected_results = [
        "var_1 = var_0.sort_index(axis=0, ascending=True, inplace=False,"
        " na_position='last', ignore_index=False, level=None, sort_remaining=True)"
    ]
    assert results == expected_results


def test_dataframe_nlargest(df1):
    df = df1.nlargest(10, "A", keep="last")
    adapter = DataFrameSortValuesAdapter()
    context = SPECodeContext()
    assert adapter.accepts(df.op) == EngineAcceptance.ACCEPT
    results = adapter.generate_code(df.op, context)
    expected_results = ["var_1 = var_0.nlargest(10, keep='last', columns='A')"]
    assert results == expected_results


def test_series_nsmallest(s2):
    s = s2.nsmallest(10, keep="last")
    adapter = DataFrameSortValuesAdapter()
    context = SPECodeContext()
    assert adapter.accepts(s.op) == EngineAcceptance.ACCEPT
    results = adapter.generate_code(s.op, context)
    expected_results = ["var_1 = var_0.nsmallest(10, keep='last')"]
    assert results == expected_results
