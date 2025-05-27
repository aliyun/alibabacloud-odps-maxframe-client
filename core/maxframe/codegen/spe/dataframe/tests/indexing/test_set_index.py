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

from ....core import SPECodeContext
from ...indexing import DataFrameSetIndexAdapter


@pytest.mark.parametrize(
    "keys,drop,append,verify_integrity",
    [
        ("A", True, False, False),
        (["C", "D"], True, False, False),
        (["C", "D"], True, True, False),
        (["C", "D"], True, False, True),
    ],
)
def test_dataframe_set_index(df1, keys, drop, append, verify_integrity):
    _df1 = df1.set_index(
        keys=keys, drop=drop, append=append, verify_integrity=verify_integrity
    )
    context = SPECodeContext()
    adapter = DataFrameSetIndexAdapter()
    results = adapter.generate_code(_df1.op, context)

    expected_results = [
        f"var_1 = var_0.set_index(keys={repr(keys)}, drop={drop}, append={append}, inplace=False, verify_integrity={verify_integrity})"
    ]
    assert results == expected_results
