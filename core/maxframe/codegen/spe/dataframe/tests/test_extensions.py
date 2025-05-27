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

from ..... import dataframe as md
from ...core import SPECodeContext
from ..extensions import DataFrameApplyChunkAdapter, DataFrameReshuffleAdapter


@pytest.fixture
def df1():
    return md.DataFrame(
        [
            [1, 10, 100, 1000],
            [2, 20, 200, 2000],
            [3, 30, 300, 3000],
            [7, 70, 700, 7000],
        ],
        index=md.Index([1, 2, 3, 7], name="test_idx"),
        columns=list("ABCD"),
    )


def test_dataframe_reshuffle(df1):
    context = SPECodeContext()
    adapter = DataFrameReshuffleAdapter()

    res = df1.mf.reshuffle()
    results = adapter.generate_code(res.op, context)
    expected_results = ["var_1 = var_0.sample(frac=1)"]
    assert results == expected_results

    res = df1.mf.reshuffle(ignore_index=True)
    results = adapter.generate_code(res.op, context)
    expected_results = ["var_2 = var_0.sample(frac=1, ignore_index=True)"]
    assert results == expected_results

    res = df1.mf.reshuffle(group_by="A")
    results = adapter.generate_code(res.op, context)
    expected_results = ["var_3 = var_0.sort_values(['A'], ascending=True)"]
    assert results == expected_results

    res = df1.mf.reshuffle(group_by="A", sort_by="B")
    results = adapter.generate_code(res.op, context)
    expected_results = ["var_4 = var_0.sort_values(['A', 'B'], ascending=True)"]
    assert results == expected_results

    res = df1.mf.reshuffle(group_by="A", sort_by="B", ignore_index=True)
    results = adapter.generate_code(res.op, context)
    expected_results = [
        "var_5 = var_0.sort_values(['A', 'B'], ascending=True, ignore_index=True)"
    ]
    assert results == expected_results


def test_dataframe_apply_chunk(df1):
    adapter = DataFrameApplyChunkAdapter()

    context = SPECodeContext()
    res = df1.mf.apply_chunk(lambda x: x + 1)
    results = adapter.generate_code(res.op, context)
    expected_results = [f"var_1 = {context.get_udfs()[0].name}(var_0)"]
    assert results == expected_results

    context = SPECodeContext()
    res = df1.mf.apply_chunk(lambda x: x + 1, batch_rows=3)
    results = adapter.generate_code(res.op, context)
    expected_results = [
        f"var_0_batches = [{context.get_udfs()[0].name}(var_0.iloc[i : i + 3])"
        " for i in range(0, len(var_0), 3)]",
        "if len(var_0_batches) > 1:",
        "    var_1 = pd.concat(var_0_batches)",
        "else:",
        "    var_1 = var_0_batches[0]",
    ]
    assert results == expected_results
