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

from ....core import SPECodeContext
from ...indexing import DataFrameIlocGetItemAdapter, DataFrameIlocSetItemAdapter


def test_series_iloc_single_row(s1):
    df = s1.iloc[2]
    context = SPECodeContext()
    results = DataFrameIlocGetItemAdapter().generate_code(df.op, context)
    expected_results = ["var_1 = var_0.iloc[2]"]
    assert results == expected_results


def test_series_iloc_set_single_row(s1):
    s1.iloc[2] = 2
    context = SPECodeContext()
    results = DataFrameIlocSetItemAdapter().generate_code(s1.op, context)
    expected_results = ["var_1 = var_0.copy()", "var_1.iloc[2] = 2"]
    assert results == expected_results


def test_series_iloc_multi_rows(s1):
    df = s1.iloc[:2]
    context = SPECodeContext()
    results = DataFrameIlocGetItemAdapter().generate_code(df.op, context)
    expected_results = ["var_1 = var_0.iloc[slice(None, 2, None)]"]
    assert results == expected_results


def test_series_head(s1):
    df = s1.head(-3)
    context = SPECodeContext()
    results = DataFrameIlocGetItemAdapter().generate_code(df.op, context)
    expected_results = ["var_1 = var_0.iloc[slice(0, -3, None)]"]
    assert results == expected_results


def test_dataframe_iloc_only_columns(df1):
    df = df1.iloc[:, [0, 1]]
    context = SPECodeContext()
    results = DataFrameIlocGetItemAdapter().generate_code(df.op, context)
    expected_results = ["var_1 = var_0.iloc[slice(None, None, None), [0, 1]]"]
    assert results == expected_results


def test_dataframe_iloc_set_only_column(df1):
    df1.iloc[:, 0] = 5
    context = SPECodeContext()
    results = DataFrameIlocSetItemAdapter().generate_code(df1.op, context)
    expected_results = [
        "var_1 = var_0.copy()",
        "var_1.iloc[slice(None, None, None), 0] = 5",
    ]
    assert results == expected_results


def test_dataframe_iloc_only_rows(df1):
    df = df1.iloc[1]
    context = SPECodeContext()
    results = DataFrameIlocGetItemAdapter().generate_code(df.op, context)
    expected_results = ["var_1 = var_0.iloc[1, slice(None, None, None)]"]
    assert results == expected_results


def test_dataframe_iloc_rows_and_columns(df1):
    df = df1.iloc[1:4:2, [True, False, False, True]]
    context = SPECodeContext()
    results = DataFrameIlocGetItemAdapter().generate_code(df.op, context)
    expected_results = [
        "var_1 = var_0.iloc[slice(1, 4, 2), [True, False, False, True]]"
    ]
    assert results == expected_results


def test_dataframe_head(df1):
    df = df1.head(3)
    context = SPECodeContext()
    results = DataFrameIlocGetItemAdapter().generate_code(df.op, context)
    expected_results = [
        "var_1 = var_0.iloc[slice(0, 3, None), slice(None, None, None)]"
    ]
    assert results == expected_results
