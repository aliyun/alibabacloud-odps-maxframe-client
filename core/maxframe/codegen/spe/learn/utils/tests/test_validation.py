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
from ......learn.utils import check_consistent_length
from ....core import SPECodeContext
from ..validation import CheckConsistentLengthAdapter


def test_check_consistent_length():
    t1 = mt.array([1, 2, 3, 4])
    t1._shape = (np.nan,)
    result = check_consistent_length(t1)

    adapter = CheckConsistentLengthAdapter()
    context = SPECodeContext()
    results = adapter.generate_code(result[0].op, context)
    expected_results = ["var_1 = check_consistent_length(var_0)"]
    assert results == expected_results

    t2 = mt.array([1, 2, 3, 5])
    t2._shape = (np.nan,)

    result = check_consistent_length(t1, t2, ref_length=4)
    adapter = CheckConsistentLengthAdapter()
    context = SPECodeContext()
    results = adapter.generate_code(result[0].op, context)
    expected_results = [
        "var_2, var_3 = check_consistent_length(var_0, var_1, ref_length=4)"
    ]
    assert results == expected_results
