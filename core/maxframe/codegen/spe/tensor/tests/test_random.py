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
from ..random import TensorRandAdapter


def test_rand_multi_dimension():
    s1 = mt.random.rand(1000, 5)
    adapter = TensorRandAdapter()
    context = SPECodeContext()
    results = adapter.generate_code(s1.op, context)
    expected_results = ["var_0 = np.random.rand(1000, 5)"]
    assert results == expected_results


def test_rand_no_dimension():
    s1 = mt.random.rand()
    adapter = TensorRandAdapter()
    context = SPECodeContext()
    results = adapter.generate_code(s1.op, context)
    expected_results = ["var_0 = np.random.rand()"]
    assert results == expected_results


def test_rand_backward_compatibility_multi_dimension():
    s1 = mt.random.rand(10, 5, 20)
    s1.op.size = None
    adapter = TensorRandAdapter()
    context = SPECodeContext()
    results = adapter.generate_code(s1.op, context)
    expected_results = ["var_0 = np.random.rand(10, 5, 20)"]
    assert results == expected_results


def test_rand_backward_compatibility_no_dimension():
    s1 = mt.random.rand()
    s1.op.size = None
    adapter = TensorRandAdapter()
    context = SPECodeContext()
    results = adapter.generate_code(s1.op, context)
    expected_results = ["var_0 = np.random.rand()"]
    assert results == expected_results
