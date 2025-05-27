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

from ...... import dataframe as md
from .....core import EngineAcceptance
from ....core import SPECodeContext
from ...missing import DataFrameFillNAAdapter


@pytest.fixture
def df1():
    return md.DataFrame(
        [
            [None, 10.0, None, "1000"],
            [2, None, False, "2000"],
            [3, None, None, "3000"],
            [4, None, True, None],
        ],
        index=md.Index([1, 2, 3, 4], name="test_idx"),
        columns=list("ABCD"),
    )


@pytest.fixture
def df2():
    return md.DataFrame(
        [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]],
        index=[4, 3, 2, 1],
        columns=list("ABCE"),
    )


def test_fill_na_with_scalar(df1):
    adapter = DataFrameFillNAAdapter()
    context = SPECodeContext()
    df = df1.fillna(3)
    assert adapter.accepts(df.op) == EngineAcceptance.ACCEPT
    results = adapter.generate_code(df.op, context)
    expected_results = [
        "var_1 = var_0.fillna(value=3, method=None, axis=0, limit=None, downcast=None)"
    ]
    assert results == expected_results


def test_dataframe_fill_na_with_dict(df1):
    adapter = DataFrameFillNAAdapter()
    context = SPECodeContext()
    df = df1.fillna({"A": 1, "C": True, "D": "filled"})
    assert adapter.accepts(df.op) == EngineAcceptance.ACCEPT
    results = adapter.generate_code(df.op, context)
    expected_results = [
        "var_1 = var_0.fillna(value={'A': 1, 'C': True, 'D': 'filled'},"
        " method=None, axis=0, limit=None, downcast=None)"
    ]
    assert results == expected_results


def test_fill_na_with_dataframe(df1, df2):
    adapter = DataFrameFillNAAdapter()
    context = SPECodeContext()
    df = df1.fillna(df2)
    assert adapter.accepts(df.op) == EngineAcceptance.ACCEPT
    results = adapter.generate_code(df.op, context)
    expected_results = [
        "var_1 = var_0.fillna(value=var_2, method=None, axis=0, limit=None,"
        " downcast=None)"
    ]
    assert results == expected_results


def test_fill_na_in_column(df1):
    adapter = DataFrameFillNAAdapter()
    context = SPECodeContext()
    df = df1.fillna(method="bfill", axis=1)
    assert adapter.accepts(df.op) == EngineAcceptance.ACCEPT
    results = adapter.generate_code(df.op, context)
    expected_results = [
        "var_1 = var_0.fillna(value=None, method='bfill', axis=1, limit=None,"
        " downcast=None)"
    ]
    assert results == expected_results
