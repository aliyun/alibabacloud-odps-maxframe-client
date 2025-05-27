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
from .....tensor.special import expit
from ...core import SPECodeContext
from ..special import TensorSpecialUnaryOpAdapter


def test_special_unary():
    result = expit(mt.random.rand(10, 10))

    adapter = TensorSpecialUnaryOpAdapter()
    context = SPECodeContext()
    results = adapter.generate_code(result.op, context)
    expected_results = ["var_1 = spspecial.expit(var_0)"]
    assert results == expected_results
