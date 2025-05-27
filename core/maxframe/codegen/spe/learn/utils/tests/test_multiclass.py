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
from ......learn.utils.multiclass import is_multilabel, type_of_target, unique_labels
from ....core import SPECodeContext
from ..multiclass import IsMultilabelAdapter, TypeOfTargetAdapter, UniqueLabelsAdapter


def test_is_multilabel():
    t1 = mt.array([1, 2, 3, 4])
    result = is_multilabel(t1)

    adapter = IsMultilabelAdapter()
    context = SPECodeContext()
    results = adapter.generate_code(result.op, context)
    expected_results = ["var_1 = sk_is_multilabel(var_0)"]
    assert results == expected_results


def test_type_of_target():
    t1 = mt.array(["a", "b", "a"])
    result = type_of_target(t1)

    adapter = TypeOfTargetAdapter()
    context = SPECodeContext()
    results = adapter.generate_code(result.op, context)
    expected_results = ["var_1 = sk_type_of_target(var_0)"]
    assert results == expected_results


def test_unique_labels():
    t1 = mt.array([1, 2, 3, 4])
    t2 = mt.array([2, 2, 3, 4])
    result = unique_labels(t1, t2)

    adapter = UniqueLabelsAdapter()
    context = SPECodeContext()
    results = adapter.generate_code(result.op, context)
    expected_results = ["var_2 = sk_unique_labels(var_0, var_1)"]
    assert results == expected_results
