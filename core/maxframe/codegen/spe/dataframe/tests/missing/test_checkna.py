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
from .....core import EngineAcceptance
from ....core import SPECodeContext
from ....dataframe.missing import DataFrameCheckNAAdapter


@pytest.fixture
def df1():
    return md.DataFrame(
        [
            [1, 2, np.Inf, pd.NA],
            [4, 5, np.NaN, pd.NaT],
            [3, None, None, "3000"],
            [4, None, True, None],
        ],
        index=md.Index([1, 2, 3, 4], name="test_idx"),
        columns=list("ABCD"),
    )


@pytest.fixture
def ds1():
    return md.Series(pd.Series((1, 2, 3, None, None, None, None)))


@pytest.fixture
def index1():
    return md.Index(pd.Index([1, 2, None, 4]))


@pytest.mark.parametrize(
    "check_func, expected_condition",
    [("isna", "isna"), ("isnull", "isna"), ("notna", "notna"), ("notnull", "notna")],
)
def test_seires(ds1, check_func, expected_condition):
    adapter = DataFrameCheckNAAdapter()
    context = SPECodeContext()
    func = getattr(ds1, check_func)
    ds = func()
    assert adapter.accepts(ds.op) == EngineAcceptance.ACCEPT
    results = adapter.generate_code(ds.op, context)
    expected_results = [f"var_1 = var_0.{expected_condition}()"]
    assert results == expected_results


@pytest.mark.parametrize(
    "check_func, expected_condition",
    [("isna", "isna"), ("isnull", "isna"), ("notna", "notna"), ("notnull", "notna")],
)
def test_maxframe(df1, check_func, expected_condition):
    adapter = DataFrameCheckNAAdapter()
    context = SPECodeContext()
    func = getattr(df1, check_func)
    df = func()
    assert adapter.accepts(df.op) == EngineAcceptance.ACCEPT
    results = adapter.generate_code(df.op, context)
    expected_results = [f"var_1 = var_0.{expected_condition}()"]
    assert results == expected_results


@pytest.mark.parametrize(
    "check_func, expected_condition",
    [
        ("isna", "isna"),
        ("notna", "notna"),
    ],
)
def test_index_isna(index1, check_func, expected_condition):
    adapter = DataFrameCheckNAAdapter()
    context = SPECodeContext()
    func = getattr(index1, check_func)
    di = func()
    assert adapter.accepts(di.op) == EngineAcceptance.ACCEPT
    results = adapter.generate_code(di.op, context)
    expected_results = [f"var_1 = var_0.{expected_condition}()"]
    assert results == expected_results
