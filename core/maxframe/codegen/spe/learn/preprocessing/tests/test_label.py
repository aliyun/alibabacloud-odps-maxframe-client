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
from ......learn.preprocessing._label import _label_binarize, label_binarize
from ....core import SPECodeContext
from .._label import LabelBinarizeOpAdapter


def test_label_binarize():
    arr = mt.array([0, 1, 2])
    result = label_binarize(arr, classes=[0, 1, 2])

    adapter = LabelBinarizeOpAdapter()
    context = SPECodeContext()
    results = adapter.generate_code(result.op, context)
    expected_results = [
        "var_1 = sk_label_binarize(var_0, classes=[0, 1, 2], neg_label=0, "
        "pos_label=1, sparse_output=False)"
    ]
    assert results == expected_results

    result = _label_binarize(arr, n_classes=3)

    adapter = LabelBinarizeOpAdapter()
    context = SPECodeContext()
    results = adapter.generate_code(result.op, context)
    expected_results = [
        "var_1 = sk_label_binarize(var_0, classes=np.arange(3), neg_label=0, "
        "pos_label=1, sparse_output=False)"
    ]
    assert results == expected_results
