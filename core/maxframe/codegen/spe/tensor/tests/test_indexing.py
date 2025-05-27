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

from ..... import tensor as mt
from ...core import SPECodeContext
from ..indexing import TensorIndexAdapter, TensorIndexSetValueAdapter


def test_getitem():
    s1 = mt.random.rand(10, 20, 30)
    result = s1[:, 1, 2:3]

    adapter = TensorIndexAdapter()
    context = SPECodeContext()
    results = adapter.generate_code(result.op, context)
    expected_results = [
        "var_0 = var_1[tuple([slice(None, None, None), 1, slice(2, 3, None)])]"
    ]
    assert results == expected_results


def test_setitem():
    s1 = mt.random.rand(10, 20, 30)
    s1[:, 1, 2:3] = 1.0

    adapter = TensorIndexSetValueAdapter()
    context = SPECodeContext()
    results = adapter.generate_code(s1.op, context)
    expected_results = [
        "var_0 = var_1.copy()",
        "var_0[tuple((slice(None, None, None), 1, slice(2, 3, None)))] = 1.0",
    ]
    assert results == expected_results
