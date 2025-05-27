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

import enum

import numpy as np
import pandas as pd

from ....core import OutputType
from ....dataframe import DataFrame as MFDataFrame
from ....dataframe import Series as MFSeries
from ....errors import TileableNotExecutedError
from ....lib.version import parse as parse_version
from ....tensor import tensor as mf_tensor
from ..models import ModelApplyChunk, ModelWithEval, ModelWithEvalData, to_remote_model


class LGBMModelType(enum.Enum):
    CLASSIFIER = 0
    REGRESSOR = 1
    RANKER = 2


_model_type_to_model = dict()


def get_model_cls_from_type(model_type: LGBMModelType):
    import lightgbm

    if not _model_type_to_model:
        _model_type_to_model.update(
            {
                LGBMModelType.CLASSIFIER: lightgbm.LGBMClassifier,
                LGBMModelType.REGRESSOR: lightgbm.LGBMRegressor,
                LGBMModelType.RANKER: lightgbm.LGBMRanker,
            }
        )
    return _model_type_to_model[model_type]


class BoosterData(ModelWithEvalData):
    @staticmethod
    def _calc_feature_importance(bst, importance_type="split", iteration=None):
        return bst.feature_importance(
            importance_type=importance_type, iteration=iteration
        )

    def feature_importance(self, importance_type="split", iteration=None):
        op = ModelApplyChunk(
            func=self._calc_feature_importance, output_types=[OutputType.tensor]
        )
        params = {
            "shape": (np.nan,),
            "dtype": np.dtype(np.float32 if importance_type == "gain" else np.int32),
        }
        return op(self, [params], importance_type=importance_type, iteration=iteration)[
            0
        ]

    def predict(
        self,
        data,
        start_iteration=0,
        num_iteration=None,
        raw_score=False,
        pred_leaf=False,
        pred_contrib=False,
        data_has_header=False,
        validate_features=False,
        **kwargs,
    ):
        from ._predict import predict

        return predict(
            self,
            data,
            start_iteration=start_iteration,
            num_iteration=num_iteration,
            raw_score=raw_score,
            pred_leaf=pred_leaf,
            pred_contrib=pred_contrib,
            data_has_header=data_has_header,
            validate_features=validate_features,
            **kwargs,
        )


class Booster(ModelWithEval):
    pass


class LGBMScikitLearnBase:
    _default_objective = None

    def __init__(self, *args, **kwargs):
        import lightgbm

        if args and isinstance(args[0], (lightgbm.LGBMModel, lightgbm.Booster)):
            if isinstance(args[0], lightgbm.LGBMModel):
                booster = args[0].booster_
                self._objective = args[0].objective
            else:
                booster = args[0]
                self._objective = (
                    args[0].params.get("objective") or self._default_objective
                )
            self._Booster = to_remote_model(booster, model_cls=Booster)
        else:
            super().__init__(*args, **kwargs)

    def __repr__(self):
        try:
            local_model = self.fetch()
            return repr(local_model)
        except TileableNotExecutedError:
            return f"<{type(self).__name__} unexecuted>"

    @classmethod
    def _fix_verbose_args(cls, kwds, params):
        if "verbose" not in kwds:
            return

        import lightgbm

        if parse_version(lightgbm.__version__).major >= 4:
            params["verbose"] = kwds.pop("verbose")

    @classmethod
    def _get_lgbm_class(cls):
        try:
            return getattr(cls, "_lgbm_class")
        except AttributeError:
            lgbm_class = next(
                base for base in cls.__bases__ if base.__module__.startswith("lightgbm")
            )
            cls._lgbm_class = lgbm_class
            return lgbm_class

    @classmethod
    def _get_param_names(cls):
        return cls._get_lgbm_class()._get_param_names()

    @staticmethod
    def _copy_extra_params(source, dest):
        params = source.get_params()
        attributes = source.__dict__
        extra_param_names = set(attributes.keys()).difference(params.keys())
        for name in extra_param_names:
            setattr(dest, name, attributes[name])

    @staticmethod
    def _convert_tileable(obj):
        if isinstance(obj, np.ndarray):
            return mf_tensor(obj)
        elif isinstance(obj, pd.DataFrame):
            return MFDataFrame(obj)
        elif isinstance(obj, pd.Series):
            return MFSeries(obj)
        return obj

    @classmethod
    def _construct_dataset(
        cls, X, y, sample_weight, init_score, group, params, categorical_feature="auto"
    ):
        from .dataset import Dataset

        return Dataset(
            cls._convert_tileable(X),
            label=cls._convert_tileable(y),
            weight=cls._convert_tileable(sample_weight),
            group=group,
            init_score=cls._convert_tileable(init_score),
            params=params,
            categorical_feature=categorical_feature,
        )

    def fit(
        self,
        X,
        y,
        *,
        sample_weight=None,
        init_score=None,
        group=None,
        eval_set=None,
        eval_names=None,
        eval_sample_weight=None,
        eval_init_score=None,
        eval_group=None,
        eval_metric=None,
        early_stopping_rounds=None,
        verbose=True,
        feature_name="auto",
        categorical_feature="auto",
        callbacks=None,
        init_model=None,
        num_class=None,
        **kwargs,
    ):
        """
        unsupported features:
        1. class weights
        2. validation of dataframe inputs
        3. eval_metric
        4. fobj
        5. feval
        """
        from ._train import train

        session = kwargs.pop("session", None)
        run_kwargs = kwargs.pop("run_kwargs", dict())

        self._n_features = X.shape[1]
        if self._objective is None:
            self._objective = self._default_objective

        self._evals_result = {}
        params = self.get_params()
        self._fix_verbose_args(kwargs, params)

        params.pop("silent", None)
        params.pop("importance_type", None)
        params.pop("n_estimators", None)
        params.pop("class_weight", None)
        params.pop("objective", None)

        if isinstance(params["random_state"], np.random.RandomState):
            params["random_state"] = params["random_state"].randint(
                np.iinfo(np.int32).max
            )
        self._n_classes = num_class or 2
        if self._n_classes is not None and self._n_classes > 2:
            params.pop("num_class", None)
            params["num_class"] = self._n_classes
        params["objective"] = self._objective

        train_set = self._construct_dataset(
            X,
            y,
            sample_weight,
            init_score,
            group,
            params,
            categorical_feature=categorical_feature,
        )

        valid_sets = []
        if eval_set is not None:

            def _get_meta_data(collection, name, i):
                if collection is None:
                    return None
                elif isinstance(collection, list):
                    return collection[i] if len(collection) > i else None
                elif isinstance(collection, dict):
                    return collection.get(i, None)
                else:
                    raise TypeError("{} should be dict or list".format(name))

            if isinstance(eval_set, tuple):
                eval_set = [eval_set]
            for i, valid_data in enumerate(eval_set):
                # reduce cost for prediction training data
                if valid_data[0] is X and valid_data[1] is y:
                    valid_set = train_set
                else:
                    valid_weight = _get_meta_data(
                        eval_sample_weight, "eval_sample_weight", i
                    )
                    valid_init_score = _get_meta_data(
                        eval_init_score, "eval_init_score", i
                    )
                    valid_group = _get_meta_data(eval_group, "eval_group", i)
                    valid_set = self._construct_dataset(
                        valid_data[0],
                        valid_data[1],
                        valid_weight,
                        valid_init_score,
                        valid_group,
                        params,
                    )
                valid_sets.append(valid_set)

        self._Booster = train(
            params,
            train_set,
            num_boost_round=self.n_estimators,
            valid_sets=valid_sets,
            valid_names=eval_names,
            early_stopping_rounds=early_stopping_rounds,
            evals_result=self._evals_result,
            verbose_eval=verbose,
            feature_name=feature_name,
            categorical_feature=categorical_feature,
            callbacks=callbacks,
            init_model=init_model,
            session=session,
            run_kwargs=run_kwargs,
            **kwargs,
        )
        self.fitted_ = True
        return self

    def predict(
        self,
        X,
        raw_score=False,
        start_iteration=0,
        num_iteration=None,
        pred_leaf=False,
        pred_contrib=False,
        **kwargs,
    ):
        return self._Booster.predict(
            X,
            raw_score=raw_score,
            start_iteration=start_iteration,
            num_iteration=num_iteration,
            pred_leaf=pred_leaf,
            pred_contrib=pred_contrib,
            **kwargs,
        )

    @property
    def feature_importances_(self):
        importances = self._Booster.feature_importance(
            importance_type=self.importance_type
        )
        importances.params = {"shape": (self._n_features,)}
        return importances

    def execute(self, session=None, run_kwargs=None):
        self._Booster.execute(session=session, run_kwargs=run_kwargs)
        return self

    def fetch(self, session=None, run_kwargs=None):
        import lightgbm

        engine_kw = {"session": session, "run_kwargs": run_kwargs or dict()}
        booster = self._Booster.fetch(**engine_kw)
        if not hasattr(self, "_le"):
            le = None
        else:
            try:
                le = self._le.fetch(**engine_kw)
            except TileableNotExecutedError:
                le = self._le.execute(**engine_kw).fetch(**engine_kw)
        remote_bst, self._Booster = self._Booster, booster
        try:
            local_cls = getattr(lightgbm, type(self).__name__)
            local_model = local_cls(**self.get_params(deep=True))
            local_model._Booster = booster
            local_model._n_features = booster.num_feature()
            if le is not None:
                # need classes and label encoder for classifiers
                local_model._n_classes = booster.params.get("num_class", 2)
                local_model._le = le
            local_model.fitted_ = True
            return local_model
        finally:
            self._Booster = remote_bst
