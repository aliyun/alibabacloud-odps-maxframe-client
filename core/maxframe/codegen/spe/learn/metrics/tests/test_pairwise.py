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
from ......learn.metrics.pairwise import euclidean_distances
from ....core import SPECodeContext
from ..pairwise import EuclideanDistancesAdapter


def test_euclidean_distances():
    x = mt.random.rand(10, 3)

    d = euclidean_distances(
        x,
        X_norm_squared=mt.random.rand(10, 1),
    )

    adapter = EuclideanDistancesAdapter()
    context = SPECodeContext()
    results = adapter.generate_code(d.op, context)
    expected_results = [
        "var_0 = sklearn.metrics.pairwise.euclidean_distances("
        "var_1, var_1, squared=False, X_norm_squared=var_2)"
    ]
    assert results == expected_results
