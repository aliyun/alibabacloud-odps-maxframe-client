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

import pandas as pd
import pytest

from ...... import dataframe as md
from ....core import SPECodeContext
from ...misc import DropDuplicatesOperatorAdapter


@pytest.fixture
def df1():
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


@pytest.fixture
def s1():
    return md.Series([5, 2, 3, 1], index=[5, 2, 3, 1], name="A")


@pytest.fixture
def multi_index():
    arrays = [
        ["a1", "a2", "a3"],
        ["b1", "b2", "b3"],
        ["c1", "c2", "c3"],
        ["d1", "d2", "d3"],
        ["e1", "e2", "e3"],
    ]
    return md.Index(
        pd.MultiIndex.from_arrays(
            arrays, names=("idx_a", "idx_b", "idx_c", "idx_d", "idx_e")
        )
    )


def test_dataframe_drop_duplicates(df1):
    adapter = DropDuplicatesOperatorAdapter()
    context = SPECodeContext()

    df = df1.drop_duplicates(["A", "B"], keep=False)
    results = adapter.generate_code(df.op, context)
    expected_results = [
        "var_1 = var_0.drop_duplicates(subset=['A', 'B'], keep=False, "
        "ignore_index=False)"
    ]
    assert results == expected_results


def test_series_drop_duplicates(s1):
    adapter = DropDuplicatesOperatorAdapter()
    context = SPECodeContext()

    s = s1.drop_duplicates(keep="first")
    results = adapter.generate_code(s.op, context)
    expected_results = [
        "var_1 = var_0.drop_duplicates(keep='first', ignore_index=False)"
    ]
    assert results == expected_results


def test_index_drop_duplicates(multi_index):
    adapter = DropDuplicatesOperatorAdapter()
    context = SPECodeContext()

    df = multi_index.drop_duplicates(keep="first")
    results = adapter.generate_code(df.op, context)
    expected_results = ["var_1 = var_0.drop_duplicates(keep='first')"]
    assert results == expected_results
