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

import functools
import operator

import numpy as np
import pandas as pd

import maxframe.dataframe as md
import maxframe.tensor as mt
from maxframe.codegen.spe.core import SPECodeContext
from maxframe.codegen.spe.dataframe.arithmetic import (
    DataFrameArithmeticTreeOpAdapter,
    DataFrameBinOpAdapter,
    DataFrameUnaryFuncAdapter,
)
from maxframe.dataframe.arithmetic.bitwise_and import DataFrameTreeAnd
from maxframe.dataframe.arithmetic.bitwise_or import DataFrameTreeOr


def test_common_unary():
    s1 = mt.random.randint(0, 100, size=(5, 2))
    v1 = md.DataFrame(s1, columns=["A", "B"], chunk_size=1)
    df = v1.abs()
    adapter = DataFrameUnaryFuncAdapter()
    context = SPECodeContext()
    results = adapter.generate_code(df.op, context)
    expected_results = ["var_1 = np.abs(var_0)"]
    assert results == expected_results


def test_round():
    adapter = DataFrameUnaryFuncAdapter()
    context = SPECodeContext()

    s1 = mt.random.randint(0, 100, size=(5, 2))
    v1 = md.DataFrame(s1, columns=["A", "B"], chunk_size=1)

    df = v1.round({"A": 0, "B": 1})
    results = adapter.generate_code(df.op, context)
    expected_results = ["var_1 = var_0.round({'A': 0, 'B': 1})"]
    assert results == expected_results

    df = v1.round(1)
    results = adapter.generate_code(df.op, context)
    expected_results = ["var_2 = var_0.round(1)"]
    assert results == expected_results


def test_binary_with_axis():
    s1 = mt.random.randint(0, 100, size=(5, 2))
    s2 = mt.random.randint(0, 100, size=(5, 2))
    v1 = md.DataFrame(s1, columns=["A", "B"], chunk_size=1)
    v2 = md.DataFrame(s2, columns=["A", "B"], chunk_size=1)
    df = v1.add(v2, axis="columns")
    adapter = DataFrameBinOpAdapter()
    context = SPECodeContext()
    results = adapter.generate_code(df.op, context)
    expected_results = ["var_2 = var_0.add(var_1, axis=1)"]
    assert results == expected_results


def test_binary_without_axis():
    s1 = mt.random.randint(0, 100, size=(5, 2))
    v1 = md.DataFrame(s1, columns=["A", "B"], chunk_size=1)
    v2 = md.DataFrame(s1, columns=["A", "B"], chunk_size=1)
    df = v1["A"].add(v2["B"])
    adapter = DataFrameBinOpAdapter()
    context = SPECodeContext()
    results = adapter.generate_code(df.op, context)
    expected_results = ["var_2 = var_0.add(var_1)"]
    assert results == expected_results


def test_tree_and():
    # Fixed, pairwise-distinct inputs: SPE maps tileables by key, so two
    # data-identical series collapse to one generated variable.  Random
    # booleans occasionally coincide (P ~ 3/1024), flaking the exact-string
    # assertion below — distinct data keeps the allocation deterministic.
    s1 = pd.Series([True] * 10)
    s2 = pd.Series([False] * 10)
    s3 = pd.Series([i % 2 == 0 for i in range(10)])
    md_s1 = md.Series(s1, chunk_size=5)
    md_s2 = md.Series(s2, chunk_size=5)
    md_s3 = md.Series(s3, chunk_size=5)

    op = DataFrameTreeAnd(combine_size=5)
    result = op(md_s1, md_s2, md_s3)

    adapter = DataFrameArithmeticTreeOpAdapter()
    context = SPECodeContext()
    codes = adapter.generate_code(result.op, context)
    expected = "var_3 = functools.reduce(operator.and_, [var_0, var_1, var_2])"
    assert codes == [expected]


def test_tree_or():
    # Fixed, pairwise-distinct inputs: see test_tree_and — data-identical
    # series share a tileable key and collapse to one generated variable.
    s1 = pd.Series([True] * 10)
    s2 = pd.Series([False] * 10)
    s3 = pd.Series([i % 2 == 0 for i in range(10)])
    md_s1 = md.Series(s1, chunk_size=5)
    md_s2 = md.Series(s2, chunk_size=5)
    md_s3 = md.Series(s3, chunk_size=5)

    op = DataFrameTreeOr(combine_size=5)
    result = op(md_s1, md_s2, md_s3)

    adapter = DataFrameArithmeticTreeOpAdapter()
    context = SPECodeContext()
    codes = adapter.generate_code(result.op, context)
    expected = "var_3 = functools.reduce(operator.or_, [var_0, var_1, var_2])"
    assert codes == [expected]


def test_tree_and_execution():
    s1 = pd.Series(np.random.choice([True, False], size=10))
    s2 = pd.Series(np.random.choice([True, False], size=10))
    s3 = pd.Series(np.random.choice([True, False], size=10))
    md_s1 = md.Series(s1, chunk_size=5)
    md_s2 = md.Series(s2, chunk_size=5)
    md_s3 = md.Series(s3, chunk_size=5)

    op = DataFrameTreeAnd(combine_size=5)
    result = op(md_s1, md_s2, md_s3)

    adapter = DataFrameArithmeticTreeOpAdapter()
    context = SPECodeContext()
    codes = adapter.generate_code(result.op, context)

    # Bind inputs by their actual generated names and extract the output
    # variable from the generated code instead of hardcoding ``var_3``, so
    # the test is robust to the context's variable-allocation strategy.
    input_vars = [context.get_input_tileable_variable(inp) for inp in result.op.inputs]
    ns = {input_vars[i]: v for i, v in enumerate([s1, s2, s3])}
    ns["functools"] = functools
    ns["operator"] = operator
    exec("\n".join(codes), ns)
    expected = functools.reduce(operator.and_, [s1, s2, s3])
    res_var = codes[0].split(" =")[0]
    pd.testing.assert_series_equal(ns[res_var], expected)
