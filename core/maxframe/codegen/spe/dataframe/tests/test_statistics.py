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
import pytest

from ..... import dataframe as md
from ....core import EngineAcceptance
from ...core import SPECodeContext
from ..statistics import DataFrameCorrAdapter, DataFrameQuantileAdapter


@pytest.fixture
def df1():
    return md.DataFrame(
        np.random.randn(6, 4),
        index=md.Index(["1", "2", "3", "4", "5", "6"], name="test_idx"),
        columns=list("ABCD"),
    )


@pytest.fixture
def df2():
    return md.DataFrame(
        np.random.randn(6, 4),
        index=md.Index(["6", "5", "4", "4", "5", "6"], name="test_idx"),
        columns=list("ABCD"),
    )


def test_corr(df1, df2):
    adapter = DataFrameCorrAdapter()
    context = SPECodeContext()

    v1 = df1.corr(method="pearson")
    assert adapter.accepts(v1.op) == EngineAcceptance.ACCEPT
    results = adapter.generate_code(v1.op, context)
    expected_results = ["var_1 = var_0.corr(method='pearson', min_periods=1)"]
    assert results == expected_results

    v2 = df1.corrwith(df2, method="pearson")
    assert adapter.accepts(v2.op) == EngineAcceptance.ACCEPT
    results = adapter.generate_code(v2.op, context)
    expected_results = [
        "var_2 = var_0.corrwith(var_3, axis=0, drop=False, method='pearson')"
    ]
    assert results == expected_results


def test_quantile(df1):
    adapter = DataFrameQuantileAdapter()
    context = SPECodeContext()
    v1 = df1.quantile(q=0.75)
    assert adapter.accepts(v1.op) == EngineAcceptance.ACCEPT
    results = adapter.generate_code(v1.op, context)
    expected_results = [
        "var_1 = var_0.quantile(q=0.75, axis=0, numeric_only=True, interpolation='linear')"
    ]
    assert results == expected_results
