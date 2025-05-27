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
from ...indexing import DataFrameRenameAdapter


def test_dataframe_rename(df1):
    context = SPECodeContext()
    adapter = DataFrameRenameAdapter()

    df1 = df1.rename(columns={"A": "x"})
    results = adapter.generate_code(df1.op, context)
    expected_results = ["var_1 = var_0.rename(columns={'A': 'x'}, errors='ignore')"]
    assert results == expected_results

    df1 = df1.rename(index=lambda x: x + "_")
    results = adapter.generate_code(df1.op, context)
    expected_results = [
        f"var_2 = var_1.rename(index={context.get_udfs()[0].name}, " "errors='ignore')"
    ]
    assert results == expected_results


def test_series_rename(s1):
    s1.name = "series_name"
    context = SPECodeContext()
    adapter = DataFrameRenameAdapter()
    results = adapter.generate_code(s1.op, context)
    expected_results = ["var_1 = var_0.rename(index='series_name')"]
    assert results == expected_results


def test_index_rename(idx1):
    idx1.set_names("renamed", level=0, inplace=True)
    context = SPECodeContext()
    adapter = DataFrameRenameAdapter()
    results = adapter.generate_code(idx1.op, context)
    expected_results = ["var_1 = var_0.set_names('renamed', level=0)"]
    assert results == expected_results
