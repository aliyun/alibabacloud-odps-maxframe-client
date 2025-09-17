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

import mock
import pytest

try:
    import xgboost
except ImportError:
    xgboost = None

from ...... import tensor as mt
from ......learn.contrib.xgboost import DMatrix, train
from ....core import SPECodeContext
from ..xgboost import ToDMatrixAdapter, XGBPredictAdapter, XGBTrainAdapter

pytestmark = pytest.mark.skipif(
    xgboost is None, reason="requires xgboost to run the test"
)


def test_spe_xgb_dmatrix():
    X = mt.random.rand(100, 10)
    y = mt.random.rand(100)
    ds = DMatrix(X, y)
    adapter = ToDMatrixAdapter()
    context = SPECodeContext()
    results = adapter.generate_code(ds.op, context)
    assert results == ["var_0 = xgboost.DMatrix(data=var_1, label=var_2)"]


def test_spe_xgb_train_script():
    X = mt.random.rand(100, 10)
    y = mt.random.rand(100)
    ds = DMatrix(X, y)
    model = train({}, ds, num_boost_round=10)
    adapter = XGBTrainAdapter()
    context = SPECodeContext()
    results = adapter.generate_code(model.op, context)
    assert results == [
        "start_time = time.time()",
        "var_2 = dict()",
        "logger.info('Trained data size: (%s, %s)', var_1.num_row(), var_1.num_col())",
        "var_0 = xgboost.train(params={}, dtrain=var_1, evals=[], num_boost_round=10, evals_result=var_2)",
        "logger.info('Train cost: %.2f s', time.time() - start_time)",
    ]

    with mock.patch(
        "maxframe.core.entity.executable._ExecutableMixin.execute",
        new=lambda *_, **__: None,
    ), mock.patch(
        "maxframe.core.entity.executable._ExecutableMixin.fetch",
        new=lambda *_, **__: {},
    ):
        X_valid = mt.random.rand(50, 10)
        y_valid = mt.random.rand(50)
        ds_valid = DMatrix(X_valid, y_valid)
        model = train({}, ds, evals=[(ds_valid, "vset")])
        adapter = XGBTrainAdapter()
        context = SPECodeContext()
        results = adapter.generate_code(model.op, context)
        assert results == [
            "start_time = time.time()",
            "var_3 = dict()",
            "logger.info('Trained data size: (%s, %s)', var_1.num_row(), var_1.num_col())",
            "var_0 = xgboost.train(params={}, dtrain=var_1, evals=[(var_2, 'vset')], "
            "num_boost_round=10, evals_result=var_3)",
            "logger.info('Train cost: %.2f s', time.time() - start_time)",
        ]


def test_spe_xgb_predict():
    X = mt.random.rand(100, 10)
    y = mt.random.rand(100)
    ds = DMatrix(X, y)
    model = train({}, ds)
    predicted = model.predict(ds)
    adapter = XGBPredictAdapter()
    context = SPECodeContext()
    results = adapter.generate_code(predicted.op, context)
    assert results == [
        "start_time = time.time()",
        "logger.info('Predicted data size: (%s, %s)', var_1.num_row(), var_1.num_col())",
        "var_0 = var_2.predict(var_1, pred_leaf=False, pred_contribs=False, "
        "approx_contribs=False, pred_interactions=False, validate_features=True, "
        "training=False, iteration_range=(0, 0), strict_shape=False)",
        "logger.info('Predict cost: %.2f s', time.time() - start_time)",
    ]
