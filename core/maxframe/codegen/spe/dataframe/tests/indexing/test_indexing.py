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

from ....core import SPECodeContext
from ...indexing import DataFrameAlignAdapter, DataFrameInsertAdapter


def test_dataframe_align(df1, df2):
    context = SPECodeContext()
    adapter = DataFrameAlignAdapter()

    aligned1, aligned2 = df1.align(df2, join="outer")
    results = adapter.generate_code(aligned1.op, context)
    expected_results = ["var_1, var_2 = var_0.align(var_3, join='outer', fill_axis=0)"]
    assert results == expected_results


def test_dataframe_insert(df1, df2):
    context = SPECodeContext()
    adapter = DataFrameInsertAdapter()

    df1.insert(1, "newcol", [4, 9, 2, 3, 5, 7])
    results = adapter.generate_code(df1.op, context)
    expected_results = [
        "var_1 = var_0.insert(loc=1, column='newcol', "
        "value=[4, 9, 2, 3, 5, 7], allow_duplicates=False)"
    ]
    assert results == expected_results
