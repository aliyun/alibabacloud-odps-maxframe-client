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
from ...missing import DataFrameDropNAAdapter


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


def test_drop_na(df1):
    adapter = DataFrameDropNAAdapter()
    context = SPECodeContext()
    df = df1.dropna(thresh=2)
    assert adapter.accepts(df.op) == EngineAcceptance.ACCEPT
    results = adapter.generate_code(df.op, context)
    expected_results = ["var_1 = var_0.dropna(axis=0, thresh=2)"]
    assert results == expected_results

    df = df1.dropna(thresh=2, ignore_index=True)
    assert adapter.accepts(df.op) == EngineAcceptance.ACCEPT
    results = adapter.generate_code(df.op, context)
    expected_results = ["var_2 = var_0.dropna(axis=0, thresh=2).reset_index(drop=True)"]
    assert results == expected_results
