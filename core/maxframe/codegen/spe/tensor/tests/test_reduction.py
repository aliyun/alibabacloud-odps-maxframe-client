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
from ..reduction import (
    TensorCumReductionAdapter,
    TensorReductionAdapter,
    TensorVarAdapter,
)


def test_reduction():
    s1 = mt.random.rand(1000, 5)
    result = mt.argmax(s1, axis=1)

    adapter = TensorReductionAdapter()
    context = SPECodeContext()
    results = adapter.generate_code(result.op, context)
    expected_results = ["var_1 = np.argmax(var_0, axis=1)"]
    assert results == expected_results


def test_cum_reduction():
    s1 = mt.random.rand(1000, 5)
    result = mt.cumsum(s1, axis=1)

    adapter = TensorCumReductionAdapter()
    context = SPECodeContext()
    results = adapter.generate_code(result.op, context)
    expected_results = ["var_1 = np.cumsum(var_0, axis=1)"]
    assert results == expected_results


def test_var():
    s1 = mt.random.rand(1000, 5)
    result = mt.var(s1, ddof=0.1)

    adapter = TensorVarAdapter()
    context = SPECodeContext()
    results = adapter.generate_code(result.op, context)
    expected_results = ["var_1 = np.var(var_0, ddof=0.1)"]
    assert results == expected_results


def test_nanvar():
    s1 = mt.random.rand(1000, 5)
    result = mt.nanvar(s1, ddof=0.1)

    adapter = TensorVarAdapter()
    context = SPECodeContext()
    results = adapter.generate_code(result.op, context)
    expected_results = ["var_1 = np.nanvar(var_0, ddof=0.1)"]
    assert results == expected_results
