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

from ....core import SPECodeContext
from ...missing import DataFrameReplaceAdapter


def test_dataframe_replace():
    df1 = md.DataFrame(
        {"A": [0, 1, 2, 3, 4], "B": [5, 6, 7, 8, 9], "C": ["a", "b", "c", "d", "e"]}
    )
    df = df1.replace(to_replace=0, value=5)
    adapter = DataFrameReplaceAdapter()
    context = SPECodeContext()
    results = adapter.generate_code(df.op, context)
    expected_results = [
        "var_1 = var_0.replace(to_replace=0, value=5, limit=None, regex=False)"
    ]
    assert results == expected_results


def test_series_replace():
    s1 = md.Series([1, 2, 3, 4, 5])
    s = s1.replace(to_replace=1, value=5)

    adapter = DataFrameReplaceAdapter()
    context = SPECodeContext()
    results = adapter.generate_code(s.op, context)
    expected_results = [
        "var_1 = var_0.replace(to_replace=1, value=5, limit=None, regex=False)"
    ]
    assert results == expected_results
