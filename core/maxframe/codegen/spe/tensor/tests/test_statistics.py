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
from ..statistics import TensorBinCountAdapter, TensorHistogramAdapter


def test_bincount():
    w = mt.array([0.3, 0.5, 0.2, 0.7, 1.0, -0.6])  # weights
    x = mt.array([0, 1, 1, 2, 2, 2])
    result = mt.bincount(x, weights=w)

    adapter = TensorBinCountAdapter()
    context = SPECodeContext()
    results = adapter.generate_code(result.op, context)
    expected_results = ["var_0 = np.bincount(var_1, weights=var_2, minlength=0)"]
    assert results == expected_results


def test_histogram():
    arr = mt.arange(5)
    result, _bins = mt.histogram(arr, bins=3)

    adapter = TensorHistogramAdapter()
    context = SPECodeContext()
    results = adapter.generate_code(result.op, context)
    expected_results = [
        "var_1, _ = np.histogram(var_0, bins=var_2, range=None, density=None, "
        "weights=None)"
    ]
    assert results == expected_results
