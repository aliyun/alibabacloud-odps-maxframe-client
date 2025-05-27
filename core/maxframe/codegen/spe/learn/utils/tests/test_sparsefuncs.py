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

import numpy as np

from ...... import tensor as mt
from ......learn.utils import count_nonzero
from ....core import SPECodeContext
from ..sparsefuncs import LearnCountNonzeroAdapter


def test_sparsefuncs():
    t1 = mt.array([1, 2, 3, 4])
    t1._shape = (np.nan,)
    result = count_nonzero(t1)

    adapter = LearnCountNonzeroAdapter()
    context = SPECodeContext()
    results = adapter.generate_code(result.op, context)
    expected_results = [
        "var_1 = sk_count_nonzero(var_0, axis=None, sample_weight=None)"
    ]
    assert results == expected_results
