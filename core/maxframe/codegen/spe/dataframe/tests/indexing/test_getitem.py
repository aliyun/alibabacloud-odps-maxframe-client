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

from .... import SPECodeGenerator
from ....core import SPECodeContext
from ...indexing import DataFrameIndexAdapter, SeriesIndexAdapter


@pytest.mark.skip(reason="Mars bug: https://github.com/mars-project/mars/issues/3356")
def test_series_index_with_selecting_attributes(s1):
    s = s1.b
    results = SeriesIndexAdapter().generate_code(s.op, SPECodeContext())
    expected_results = []
    assert results == expected_results


@pytest.mark.skip(reason="Mars bug: https://github.com/mars-project/mars/issues/3357")
def test_series_index_with_multi_index_and_selecting_single_row(s2):
    s = s2[("m1", "n1")]
    results = SeriesIndexAdapter().generate_code(s.op, SPECodeContext())
    assert results


@pytest.mark.skip(reason="Mars bug: https://github.com/mars-project/mars/issues/3357")
def test_series_index_with_multi_index_and_selecting_multi_row(s2):
    s = s2[("m1", "n1")]
    results = SeriesIndexAdapter().generate_code(s.op, SPECodeContext())
    assert results


def test_series_index_with_selecting_single_row(s1):
    s = s1["b"]
    results = SeriesIndexAdapter().generate_code(s.op, SPECodeContext())
    expected_results = ["var_1 = var_0['b']"]
    assert results == expected_results


def test_series_index_with_selecting_multi_row(s1):
    s = s1[["a", "b"]]
    context = SPECodeContext()
    results = SeriesIndexAdapter().generate_code(s.op, context)
    expected_results = ["var_1 = var_0[['a', 'b']]"]
    assert results == expected_results


def test_dataframe_index_with_selecting_attributes(df1):
    df = df1.A
    results = DataFrameIndexAdapter().generate_code(df.op, SPECodeContext())
    expected_results = ["var_1 = var_0['A']"]
    assert results == expected_results


def test_dataframe_index_with_selecting_column(df1):
    df = df1["B"]
    results = DataFrameIndexAdapter().generate_code(df.op, SPECodeContext())
    expected_results = ["var_1 = var_0['B']"]
    assert results == expected_results


def test_dataframe_index_with_simple_filter(df1):
    v1 = df1["A"]
    v2 = v1 > 0
    v3 = v1 + 1
    v4 = v3 > 0
    v5 = v2 & v4
    v6 = df1["B"]
    v7 = v1 + v6
    v8 = df1["C"]
    v9 = v7 > v8
    v10 = v5 | v9
    # df1[(df1["A"] > 0) & (df1["A"] + 1 > 0) & (df1["A"] + df1["B"] > df1["C"])]
    df = df1[v10]
    graph = df.build_graph()
    generator = SPECodeGenerator("1")
    context = SPECodeContext()
    results = []
    skipped = True
    for v in graph.topological_iter():
        if skipped:
            if v.key == df1.key:
                skipped = False
            continue
        r = generator.get_op_adapter(type(v.op))().generate_code(v.op, context)
        results.extend(r)
    out_vars = [
        context.get_output_tileable_variable(v.op.outputs[0])
        for v in [df1, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, df]
    ]
    idx1 = results.index(f"{out_vars[1]} = {out_vars[0]}['A']")
    idx2 = results.index(f"{out_vars[2]} = {out_vars[1]}.gt(0)")
    idx3 = results.index(f"{out_vars[3]} = {out_vars[1]}.add(1)")
    idx4 = results.index(f"{out_vars[4]} = {out_vars[3]}.gt(0)")
    idx5 = results.index(f"{out_vars[5]} = {out_vars[2]} & {out_vars[4]}")
    idx6 = results.index(f"{out_vars[6]} = {out_vars[0]}['B']")
    idx7 = results.index(f"{out_vars[7]} = {out_vars[1]}.add({out_vars[6]})")
    idx8 = results.index(f"{out_vars[8]} = {out_vars[0]}['C']")
    idx9 = results.index(f"{out_vars[9]} = {out_vars[7]}.gt({out_vars[8]})")
    idx10 = results.index(f"{out_vars[10]} = {out_vars[5]} | {out_vars[9]}")
    idx_out = results.index(f"{out_vars[11]} = {out_vars[0]}[{out_vars[10]}]")
    assert idx1 < idx2
    assert idx1 < idx3
    assert idx3 < idx4
    assert idx2 < idx5
    assert idx4 < idx5
    assert idx1 < idx7
    assert idx6 < idx7
    assert idx7 < idx9
    assert idx8 < idx9
    assert idx5 < idx10
    assert idx9 < idx10
    assert idx10 < idx_out
