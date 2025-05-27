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
from ...indexing import DataFrameResetIndexAdapter


def test_dataframe_reset_and_drop(df1):
    # index
    df1 = df1.reset_index(drop=True)
    context = SPECodeContext()
    adapter = DataFrameResetIndexAdapter()
    results = adapter.generate_code(df1.op, context)
    expected_results = [
        "var_1 = var_0.reset_index(level=None, drop=True, inplace=False, col_level=0, col_fill='')"
    ]
    assert results == expected_results


def test_series_reset_and_drop(s1):
    # index
    s1 = s1.reset_index(drop=True)
    context = SPECodeContext()
    adapter = DataFrameResetIndexAdapter()
    results = adapter.generate_code(s1.op, context)
    expected_results = [
        "var_1 = var_0.reset_index(level=None, drop=True, inplace=False, name=0)"
    ]
    assert results == expected_results


def test_series_reset_and_name(s1):
    # index
    s1 = s1.reset_index(name="gen_by_reset")
    context = SPECodeContext()
    adapter = DataFrameResetIndexAdapter()
    results = adapter.generate_code(s1.op, context)
    expected_results = [
        "var_1 = var_0.reset_index(level=None, drop=False, inplace=False, name='gen_by_reset')"
    ]
    assert results == expected_results


def test_series_reset_and_name_and_drop(s1):
    # index
    s1 = s1.reset_index(name="gen_by_reset", drop=True)
    context = SPECodeContext()
    adapter = DataFrameResetIndexAdapter()
    results = adapter.generate_code(s1.op, context)
    # should ignore `name`
    expected_results = [
        "var_1 = var_0.reset_index(level=None, drop=True, inplace=False, name='gen_by_reset')"
    ]
    assert results == expected_results


def test_dataframe_reset(df1):
    df1 = df1.reset_index()
    context = SPECodeContext()
    adapter = DataFrameResetIndexAdapter()
    results = adapter.generate_code(df1.op, context)
    expected_results = [
        "var_1 = var_0.reset_index(level=None, drop=False, inplace=False, col_level=0, col_fill='')"
    ]
    assert results == expected_results


def test_dataframe_reset_index_multi_level_column(df2):
    col_fill = "fill_auto_gen"
    reset_df2 = df2.reset_index(level=["test_idx"], col_fill=col_fill, col_level=1)
    context = SPECodeContext()
    adapter = DataFrameResetIndexAdapter()
    results = adapter.generate_code(reset_df2.op, context)
    expected_results = [
        "var_1 = var_0.reset_index(level=['test_idx'], drop=False, inplace=False, col_level=1, col_fill='fill_auto_gen')"
    ]
    assert results == expected_results
