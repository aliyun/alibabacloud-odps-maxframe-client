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
    import lightgbm
    from sklearn.datasets import make_classification
except ImportError:
    lightgbm = make_classification = None

from ...... import tensor as mt
from ......learn.contrib.lightgbm import Dataset, train
from ....core import SPECodeContext
from ..lightgbm import LGBMPredictAdapter, LGBMTrainAdapter, ToLGBMDatasetAdapter

pytestmark = pytest.mark.skipif(
    lightgbm is None, reason="requires lightgbm to run the test"
)


def test_spe_lgbm_dataset():
    X = mt.random.rand(100, 10)
    y = mt.random.rand(100)
    ds = Dataset(X, y)
    adapter = ToLGBMDatasetAdapter()
    context = SPECodeContext()
    results = adapter.generate_code(ds.op, context)
    assert results == [
        "var_0 = lightgbm.Dataset(data=var_1, label=var_2, free_raw_data=True)"
    ]


def test_spe_lgbm_train_script():
    X = mt.random.rand(100, 10)
    y = mt.random.rand(100)
    ds = Dataset(X, y)
    model = train({}, ds, num_boost_round=10)
    adapter = LGBMTrainAdapter()
    context = SPECodeContext()
    results = adapter.generate_code(model.op, context)
    assert results == [
        "start_time = time.time()",
        "logger.info('Trained data size: (%s, %s)', var_1.num_data(), var_1.num_feature())",
        "var_0 = LGBMTrainAdapter._run_lgbm_train(params={}, train_set=var_1, "
        "num_boost_round=10, valid_sets=[], keep_training_booster=False, timeout=120)",
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
        ds_valid = Dataset(X_valid, y_valid)
        model = train({}, ds, num_boost_round=10, valid_sets=[ds_valid])
        adapter = LGBMTrainAdapter()
        context = SPECodeContext()
        results = adapter.generate_code(model.op, context)
        assert results == [
            "start_time = time.time()",
            "var_3 = dict()",
            "logger.info('Trained data size: (%s, %s)', var_1.num_data(), var_1.num_feature())",
            "var_0 = LGBMTrainAdapter._run_lgbm_train(params={}, train_set=var_1, "
            "num_boost_round=10, valid_sets=[var_2], keep_training_booster=False, "
            "timeout=120, evals_result=var_3)",
            "logger.info('Train cost: %.2f s', time.time() - start_time)",
        ]


def test_spe_lgbm_train_run():
    X, y = make_classification(n_features=5, n_informative=3)
    ds = lightgbm.Dataset(X, y)
    model = LGBMTrainAdapter._run_lgbm_train(
        params={}, train_set=ds, num_boost_round=10
    )
    assert isinstance(model, lightgbm.Booster)

    X_valid, y_valid = make_classification(n_samples=50, n_features=5, n_informative=3)
    ds_valid = lightgbm.Dataset(X_valid, y_valid)
    eval_results = {}
    model = LGBMTrainAdapter._run_lgbm_train(
        params={},
        train_set=ds,
        valid_sets=[ds_valid],
        num_boost_round=10,
        evals_result=eval_results,
    )
    assert isinstance(model, lightgbm.Booster)
    assert len(eval_results) > 0


def test_spe_lgbm_predict():
    X = mt.random.rand(100, 10)
    y = mt.random.rand(100)
    ds = Dataset(X, y)
    model = train({}, ds, num_boost_round=10)
    predicted = model.predict(ds)
    adapter = LGBMPredictAdapter()
    context = SPECodeContext()
    results = adapter.generate_code(predicted.op, context)
    assert results == [
        "start_time = time.time()",
        "var_0 = var_2.predict(var_1, start_iteration=0, raw_score=False, "
        "pred_leaf=False, pred_contrib=False)",
        "logger.info('Predict cost: %.2f s', time.time() - start_time)",
    ]
