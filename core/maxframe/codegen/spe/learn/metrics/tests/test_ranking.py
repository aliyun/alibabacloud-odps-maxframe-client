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
from ......learn.metrics import auc, roc_auc_score, roc_curve
from ....core import SPECodeContext
from .._ranking import AucOpAdapter, RocAucScoreAdapter, RocCurveAdapter


def test_auc():
    t1 = mt.array([1, 1, 2, 2])
    t2 = mt.array([0.1, 0.4, 0.35, 0.8])
    result = auc(t1, t2)

    adapter = AucOpAdapter()
    context = SPECodeContext()
    results = adapter.generate_code(result.op, context)
    expected_results = ["var_2 = sk_auc(var_0, var_1)"]
    assert results == expected_results


def test_roc_auc_score():
    t1 = mt.array([0, 0, 1, 1])
    t2 = mt.array([0.1, 0.4, 0.35, 0.8])
    result = roc_auc_score(t1, t2)

    adapter = RocAucScoreAdapter()
    context = SPECodeContext()
    results = adapter.generate_code(result.op, context)
    expected_results = [
        "var_2 = sk_roc_auc_score("
        "var_0, var_1, average='macro', multi_class='raise')"
    ]
    assert results == expected_results


def test_roc_curve():
    t1 = mt.array([0, 0, 1, 1])
    t2 = mt.array([0.1, 0.4, 0.35, 0.8])
    result = roc_curve(t1, t2)

    adapter = RocCurveAdapter()
    context = SPECodeContext()
    results = adapter.generate_code(result[0].op, context)
    expected_results = [
        "var_2, var_3, var_4 = sk_roc_curve(" "var_0, var_1, drop_intermediate=True)"
    ]
    assert results == expected_results
