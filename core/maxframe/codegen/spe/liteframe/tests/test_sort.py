# Copyright 1999-2026 Alibaba Group Holding Ltd.
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

from maxframe.codegen.spe.core import SPECodeContext
from maxframe.codegen.spe.liteframe.sort import LiteFrameSortAdapter
from maxframe.liteframe.operators.sort import LiteFrameSort


def test_sort_adapter_code_generation(make_liteframe):
    """LiteFrameSortAdapter should generate _run_sort call."""
    df = pd.DataFrame({"a": [3, 1, 2], "b": [30, 10, 20]})
    data = make_liteframe(df)
    op = LiteFrameSort(by=["a"], ascending=True, na_position="last")
    out = op(data)
    op = out.op
    adapter = LiteFrameSortAdapter()
    context = SPECodeContext()
    result = adapter.generate_code(op, context)
    assert len(result) == 1
    assert "LiteFrameSortAdapter._run_sort" in result[0]


@pytest.mark.parametrize(
    "df, ascending, nrows, keep_kind, expected_a, expected_len",
    [
        # basic ascending / descending (full frame)
        (
            pd.DataFrame({"a": [3, 1, 2], "b": [30, 10, 20]}),
            True,
            None,
            "head",
            [1, 2, 3],
            3,
        ),
        (
            pd.DataFrame({"a": [3, 1, 2], "b": [30, 10, 20]}),
            False,
            None,
            "head",
            [3, 2, 1],
            3,
        ),
        # nrows + keep_kind=head/first/last on a 5-row frame
        (
            pd.DataFrame({"a": [3, 1, 2, 5, 4], "b": [30, 10, 20, 50, 40]}),
            True,
            2,
            "head",
            [1, 2],
            2,
        ),
        (
            pd.DataFrame({"a": [3, 1, 2, 5, 4], "b": [30, 10, 20, 50, 40]}),
            True,
            3,
            "first",
            [1, 2, 3],
            3,
        ),
        (
            pd.DataFrame({"a": [3, 1, 2, 5, 4], "b": [30, 10, 20, 50, 40]}),
            False,
            3,
            "last",
            [5, 4, 3],
            3,
        ),
        # keep_kind='all' includes all ties at the boundary value
        (
            pd.DataFrame({"a": [1, 2, 2, 2, 3], "b": [10, 20, 30, 40, 50]}),
            True,
            2,
            "all",
            [1, 2, 2, 2],
            4,
        ),
    ],
    ids=["basic", "descending", "nrows_head", "keep_first", "keep_last", "keep_all"],
)
def test_sort_run_sort(df, ascending, nrows, keep_kind, expected_a, expected_len):
    """_run_sort covers ascending/descending, nrows head/first/last/all-keep."""
    result = LiteFrameSortAdapter._run_sort(
        df,
        by=["a"],
        ascending=ascending,
        na_position="last",
        nrows=nrows,
        keep_kind=keep_kind,
    )
    assert len(result) == expected_len
    assert list(result["a"]) == expected_a
