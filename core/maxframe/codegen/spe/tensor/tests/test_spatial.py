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
from ..spatial import TensorCDistAdapter


def test_cdist():
    xa = mt.random.rand(100, 10)
    xb = mt.random.rand(89, 10)
    VI = mt.random.rand(10, 10)
    dist = mt.spatial.distance.cdist(xa, xb, metric="mahalanobis", VI=VI)

    adapter = TensorCDistAdapter()
    context = SPECodeContext()
    results = adapter.generate_code(dist.op, context)
    expected_results = [
        "var_0 = scipy.spatial.distance.cdist("
        "var_1, var_2, metric='mahalanobis', VI=var_3)"
    ]
    assert results == expected_results
