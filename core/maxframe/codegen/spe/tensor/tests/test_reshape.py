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
from ..reshape import TensorReshapeAdapter


def test_reshape_C():
    t1 = mt.arange(6).reshape((3, 2))
    t2 = mt.reshape(t1, (2, 3))
    adapter = TensorReshapeAdapter()
    context = SPECodeContext()
    context.get_input_tileable_variable(t1.op.outputs[0])
    results = adapter.generate_code(t2.op, context)
    expected_results = ["var_1 = np.reshape(var_0, (2, 3), 'C')"]
    assert results == expected_results


def test_reshape_F():
    t1 = mt.arange(6).reshape((3, 2))
    t2 = mt.reshape(t1, (2, 3), "F")
    adapter = TensorReshapeAdapter()
    context = SPECodeContext()
    context.get_input_tileable_variable(t1.op.outputs[0])
    results = adapter.generate_code(t2.op, context)
    expected_results = ["var_1 = np.reshape(var_0, (2, 3), 'F')"]
    assert results == expected_results
