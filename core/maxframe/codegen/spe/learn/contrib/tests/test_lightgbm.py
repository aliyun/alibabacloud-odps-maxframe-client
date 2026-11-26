# Copyright 1999-2026 Alibaba Group Holding Ltd.
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

import types

import mock
import pytest

try:
    import lightgbm
    from sklearn.datasets import make_classification
except ImportError:
    lightgbm = make_classification = None

from maxframe import tensor as mt
from maxframe.codegen.spe.core import SPECodeContext
from maxframe.codegen.spe.learn.contrib.lightgbm import (
    LGBMPredictAdapter,
    LGBMTrainAdapter,
    ToLGBMDatasetAdapter,
)
from maxframe.learn.contrib.lightgbm import Dataset, train

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
        "var_0 = lightgbm.Dataset(data=var_1, label=var_2, free_raw_data=True, "
        "categorical_feature='auto')"
    ]


def test_spe_lgbm_dataset_keeps_explicit_none_categorical_feature():
    # None means "declare no categorical feature" and must not be dropped:
    # omitting it would make lightgbm infer categorical columns instead
    X = mt.random.rand(100, 10)
    y = mt.random.rand(100)
    ds = Dataset(X, y, categorical_feature=None)
    adapter = ToLGBMDatasetAdapter()
    context = SPECodeContext()
    results = adapter.generate_code(ds.op, context)
    assert results == [
        "var_0 = lightgbm.Dataset(data=var_1, label=var_2, free_raw_data=True, "
        "categorical_feature=None)"
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
        "num_boost_round=10, valid_sets=[], keep_training_booster=False, "
        "categorical_feature='auto')",
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
            "categorical_feature='auto', evals_result=var_3)",
            "logger.info('Train cost: %.2f s', time.time() - start_time)",
        ]


def test_spe_lgbm_train_script_gates_feature_name(monkeypatch):
    from maxframe.learn.contrib.lightgbm import core as lgbm_core

    X = mt.random.rand(100, 10)
    y = mt.random.rand(100)
    ds = Dataset(X, y)
    # fit() forwards its "auto" default, so it reaches the operator as a value
    model = train({}, ds, num_boost_round=10, feature_name="auto")
    adapter = LGBMTrainAdapter()

    monkeypatch.setattr(
        lgbm_core, "lightgbm", types.SimpleNamespace(__version__="4.5.0")
    )
    results = adapter.generate_code(model.op, SPECodeContext())
    assert results == [
        "start_time = time.time()",
        "logger.info('Trained data size: (%s, %s)', var_1.num_data(), var_1.num_feature())",
        "var_0 = LGBMTrainAdapter._run_lgbm_train(params={}, train_set=var_1, "
        "num_boost_round=10, valid_sets=[], keep_training_booster=False, "
        "feature_name='auto', categorical_feature='auto')",
        "logger.info('Train cost: %.2f s', time.time() - start_time)",
    ]

    # 4.6 dropped feature_name from train() along with categorical_feature, so
    # both must disappear from the generated script
    monkeypatch.setattr(
        lgbm_core, "lightgbm", types.SimpleNamespace(__version__="4.6.0")
    )
    results = adapter.generate_code(model.op, SPECodeContext())
    assert results[2] == (
        "var_0 = LGBMTrainAdapter._run_lgbm_train(params={}, train_set=var_1, "
        "num_boost_round=10, valid_sets=[], keep_training_booster=False)"
    )


@pytest.mark.slow
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


@pytest.mark.parametrize(
    "version, expected",
    [("4.5.0", True), ("4.6.0", False), (None, True)],
    ids=["train_accepts_arg", "train_dropped_arg", "lightgbm_absent"],
)
def test_train_accepts_categorical_feature(version, expected, monkeypatch):
    from maxframe.learn.contrib.lightgbm import core as lgbm_core

    # lazy_import resolves to None when the optional dependency is missing, and
    # codegen must then assume the historical signature
    module = None if version is None else types.SimpleNamespace(__version__=version)
    monkeypatch.setattr(lgbm_core, "lightgbm", module)
    assert LGBMTrainAdapter._train_accepts_categorical_feature() is expected
