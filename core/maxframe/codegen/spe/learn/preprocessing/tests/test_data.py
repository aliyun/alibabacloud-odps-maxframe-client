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

from ...... import tensor as mt
from ......learn.preprocessing import normalize
from ....core import SPECodeContext
from .._data import TensorNormalizeAdapter


def test_normalize():
    t1 = mt.array([[1, 2, 3], [4, 5, 6]])
    result = normalize(t1, norm="l1")

    adapter = TensorNormalizeAdapter()
    context = SPECodeContext()
    results = adapter.generate_code(result.op, context)
    expected_results = [
        "var_1 = sk_normalize(var_0, norm='l1', axis=1, return_norm=False)"
    ]
    assert results == expected_results
