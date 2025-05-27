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
from ..sort import TensorSortAdapter


def test_sort():
    t1 = mt.array([1, 4, 5, 3])
    t2 = mt.sort(t1, stable=True)
    adapter = TensorSortAdapter()
    context = SPECodeContext()
    context.get_input_tileable_variable(t1.op.outputs[0])
    results = adapter.generate_code(t2.op, context)
    expected_results = ["var_1 = np.sort(var_0, axis=0, kind='STABLE', order=None)"]
    assert results == expected_results

    t2, t3 = mt.sort(t1, stable=True, return_index=True)
    adapter = TensorSortAdapter()
    context = SPECodeContext()
    context.get_input_tileable_variable(t1.op.outputs[0])
    results = adapter.generate_code(t2.op, context)
    expected_results = [
        "var_1 = np.sort(var_0, axis=0, kind='STABLE', order=None)",
        "var_2 = np.argsort(var_0, axis=0, kind='STABLE', order=None)",
    ]
    assert results == expected_results

    t2 = mt.argsort(t1, stable=True)
    adapter = TensorSortAdapter()
    context = SPECodeContext()
    context.get_input_tileable_variable(t1.op.outputs[0])
    results = adapter.generate_code(t2.op, context)
    expected_results = [
        "var_1 = np.argsort(var_0, axis=0, kind='STABLE', order=None)",
    ]
    assert results == expected_results
