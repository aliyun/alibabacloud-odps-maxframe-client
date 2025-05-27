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
from ..window import (
    DataFrameEwmAggAdapter,
    DataFrameExpandingAggAdapter,
    DataFrameRollingAggAdapter,
)


@pytest.fixture
def df1():
    return md.DataFrame(
        np.random.randn(6, 4),
        index=md.Index(["1", "2", "3", "4", "5", "6"], name="test_idx"),
        columns=list("ABCD"),
    )


def test_expanding_agg(df1):
    adapter = DataFrameExpandingAggAdapter()
    context = SPECodeContext()
    v1 = df1.expanding(1).sum()
    assert adapter.accepts(v1.op) == EngineAcceptance.ACCEPT
    results = adapter.generate_code(v1.op, context)
    expected_results = ["var_1 = var_0.expanding(min_periods=1, axis=0).sum()"]
    assert results == expected_results


def test_ewm_agg(df1):
    adapter = DataFrameEwmAggAdapter()
    context = SPECodeContext()
    v1 = df1.ewm(alpha=0.5, adjust=False).mean()
    assert adapter.accepts(v1.op) == EngineAcceptance.ACCEPT
    results = adapter.generate_code(v1.op, context)
    expected_results = [
        "var_1 = var_0.rolling(alpha=0.5, min_periods=1, adjust=False, axis=0, "
        "ignore_na=False).mean()"
    ]
    assert results == expected_results


def test_rolling_agg(df1):
    adapter = DataFrameRollingAggAdapter()
    context = SPECodeContext()
    v1 = df1.rolling(5, min_periods=1).mean()
    assert adapter.accepts(v1.op) == EngineAcceptance.ACCEPT
    results = adapter.generate_code(v1.op, context)
    expected_results = [
        "var_1 = var_0.rolling(5, min_periods=1, center=False, axis=0).mean()"
    ]
    assert results == expected_results
