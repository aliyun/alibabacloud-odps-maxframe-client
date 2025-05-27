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
from ......learn.metrics import (
    accuracy_score,
    multilabel_confusion_matrix,
    precision_recall_fscore_support,
)
from ......learn.metrics._check_targets import _check_targets
from ....core import SPECodeContext
from .._classification import (
    AccuracyScoreAdapter,
    CheckTargetsAdapter,
    MultiLabelConfusionMatrixAdapter,
    PrecisionRecallFScoreSupportAdapter,
)


def test_check_targets():
    t1 = mt.array([1, 2, 3, 4])
    t2 = mt.array([1, 2, 3, 5])
    result = _check_targets(t1, t2)

    adapter = CheckTargetsAdapter()
    context = SPECodeContext()
    results = adapter.generate_code(result[0].op, context)
    expected_results = ["var_2, var_3, var_4 = sk_check_targets(var_0, var_1)"]
    assert results == expected_results


def test_accuracy_score():
    t1 = mt.array([1, 2, 3, 4])
    t2 = mt.array([1, 2, 3, 5])
    result = accuracy_score(t1, t2)

    adapter = AccuracyScoreAdapter()
    context = SPECodeContext()
    results = adapter.generate_code(result.op, context)
    expected_results = ["var_2 = sk_accuracy_score(var_0, var_1, normalize=True)"]
    assert results == expected_results


def test_multilabel_confusion_matrix():
    y_true = mt.array([[1, 0, 1], [0, 1, 0]])
    y_pred = mt.array([[1, 0, 0], [0, 1, 1]])
    result = multilabel_confusion_matrix(y_true, y_pred)

    adapter = MultiLabelConfusionMatrixAdapter()
    context = SPECodeContext()
    results = adapter.generate_code(result.op, context)
    expected_results = [
        "var_2 = sk_multilabel_confusion_matrix(var_0, var_1, samplewise=False)"
    ]
    assert results == expected_results


def test_precision_recall_fscore_support():
    y_true = mt.array([[1, 0, 1], [0, 1, 0]])
    y_pred = mt.array([[1, 0, 0], [0, 1, 1]])

    results = precision_recall_fscore_support(y_true, y_pred)
    adapter = PrecisionRecallFScoreSupportAdapter()
    context = SPECodeContext()
    results = adapter.generate_code(results[0].op, context)
    expected_results = [
        "var_2, var_3, var_4, var_5 = sk_precision_recall_fscore_support("
        "var_0, var_1, beta=1.0, pos_label=1, "
        "warn_for=('precision', 'recall', 'f-score'), zero_division='warn')"
    ]
    assert results == expected_results

    results = precision_recall_fscore_support(y_true, y_pred, average="macro")
    adapter = PrecisionRecallFScoreSupportAdapter()
    context = SPECodeContext()
    results = adapter.generate_code(results[0].op, context)
    expected_results = [
        "var_2, var_3, var_4, _na = sk_precision_recall_fscore_support("
        "var_0, var_1, beta=1.0, pos_label=1, average='macro', "
        "warn_for=('precision', 'recall', 'f-score'), zero_division='warn')"
    ]
    assert results == expected_results
