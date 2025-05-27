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
from ..merge import TensorConcatenateAdapter


def test_concatenate():
    s1 = mt.random.rand(1000, 5)
    s2 = mt.random.rand(1000, 5)
    adapter = TensorConcatenateAdapter()
    s3 = mt.concatenate([s1, s2], axis=1)
    context = SPECodeContext()
    results = adapter.generate_code(s3.op, context)
    expected_results = ["var_2 = np.concatenate((var_0, var_1), axis=1)"]
    assert results == expected_results
