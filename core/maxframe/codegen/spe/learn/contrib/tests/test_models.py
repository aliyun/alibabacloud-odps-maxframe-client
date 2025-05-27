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

import pytest

try:
    import lightgbm
except ImportError:
    lightgbm = None

from ...... import tensor as mt
from ......learn.contrib.lightgbm import Dataset, train
from ....core import SPECodeContext
from ..models import ModelApplyChunkAdapter


@pytest.mark.skipif(lightgbm is None, reason="requires lightgbm to run the test")
def test_spe_lgbm_feature_importance():
    X = mt.random.rand(100, 10)
    y = mt.random.rand(100)
    ds = Dataset(X, y)
    model = train({}, ds, num_boost_round=10)
    fe = model.feature_importance()

    adapter = ModelApplyChunkAdapter()
    context = SPECodeContext()
    results = adapter.generate_code(fe.op, context)
    assert results == [
        "var_1 = const_0(var_0, importance_type='split', iteration=None)"
    ]
