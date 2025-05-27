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

import maxframe.dataframe as md
import maxframe.tensor as mt

from ...core import SPECodeContext
from ..arithmetic import DataFrameBinOpAdapter, DataFrameUnaryFuncAdapter


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
