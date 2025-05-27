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

from ....core import SPECodeContext
from ...indexing import DataFrameSetitemAdapter


def test_dataframe_setitem_with_scalar_to_one_index(df1):
    df1["A"] = np.intc(100)
    context = SPECodeContext()
    adapter = DataFrameSetitemAdapter()
    results = adapter.generate_code(df1.op, context)
    expected_results = ["var_0['A'] = const_0", "var_1 = var_0"]
    assert context.constants["const_0"] == 100
    assert results == expected_results


def test_dataframe_setitem_with_scalar_to_multi_indexes(df1):
    df1[["A", "E"]] = 100
    context = SPECodeContext()
    adapter = DataFrameSetitemAdapter()
    results = adapter.generate_code(df1.op, context)
    expected_results = ["var_0[['A', 'E']] = 100", "var_1 = var_0"]
    assert results == expected_results


def test_dataframe_setitem_with_non_scalar(df1):
    df1[["A", "E"]] = df1[["A", "B"]]
    context = SPECodeContext()
    adapter = DataFrameSetitemAdapter()
    results = adapter.generate_code(df1.op, context)
    expected_results = ["var_0[['A', 'E']] = var_1", "var_2 = var_0"]
    assert results == expected_results
