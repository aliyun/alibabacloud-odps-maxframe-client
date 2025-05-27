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
from ......learn.utils.checks import (
    assert_all_finite,
    check_non_negative_then_return_value,
)
from ....core import SPECodeContext
from ..checks import AssertAllFiniteAdapter, CheckNonNegativeAdapter


def test_assert_all_finite():
    t1 = mt.array([[2, mt.nan, mt.inf], [mt.inf, mt.nan, 2]])
    result = assert_all_finite(t1)

    adapter = AssertAllFiniteAdapter()
    context = SPECodeContext()
    results = adapter.generate_code(result.op, context)
    expected_results = [
        "var_1 = sk_assert_all_finite(var_0, allow_nan=False, check_only=True)"
    ]
    assert results == expected_results


def test_check_non_negative():
    t1 = mt.array([2, -1, 2])
    result = check_non_negative_then_return_value(t1, t1, "check")

    adapter = CheckNonNegativeAdapter()
    context = SPECodeContext()
    results = adapter.generate_code(result.op, context)
    expected_results = [
        "sk_check_non_negative(var_0, whom='check')",
        "var_1 = var_0",
    ]
    assert results == expected_results
