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
from ...indexing import DataFrameSetAxisAdapter


def test_dataframe_set_axis_with_np_index(df1):
    df1.index = np.arange(1, 7)
    context = SPECodeContext()
    adapter = DataFrameSetAxisAdapter()
    results = adapter.generate_code(df1.op, context)
    expected_results = ["var_0 = var_0.set_axis(const_0, axis=0)"]
    assert results == expected_results


def test_dataframe_set_axis_with_array(df1):
    df1.columns = list("WXYZ")
    context = SPECodeContext()
    adapter = DataFrameSetAxisAdapter()
    results = adapter.generate_code(df1.op, context)
    expected_results = ["var_0 = var_0.set_axis(const_0, axis=1)"]
    assert results == expected_results


def test_series_set_axis_with_np_index(s1):
    s1.index = np.arange(1, 4)
    context = SPECodeContext()
    adapter = DataFrameSetAxisAdapter()
    results = adapter.generate_code(s1.op, context)
    expected_results = ["var_0 = var_0.set_axis(const_0, axis=0)"]
    assert results == expected_results
