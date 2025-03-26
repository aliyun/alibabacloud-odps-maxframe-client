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

import json
import os
import tempfile
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import xgboost
except ImportError:
    xgboost = None

from ....core import OutputType
from ...core import Model, ModelData
from ..models import ModelApplyChunk, to_remote_model
from .dmatrix import DMatrix


class BoosterData(ModelData):
    __slots__ = ("_evals_result",)

    _evals_result: Dict

    def __init__(self, *args, evals_result=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._evals_result = evals_result if evals_result is not None else dict()

    @staticmethod
    def _get_booster_score(bst, fmap=None, importance_type="weight"):
        if not fmap:
            tmp_file_name = ""
        else:
            tmp_file = tempfile.NamedTemporaryFile(delete=False)
            tmp_file.write(fmap)
            tmp_file.close()
            tmp_file_name = tmp_file.name

        try:
            return bst.get_score(fmap=tmp_file_name, importance_type=importance_type)
        finally:
            if tmp_file_name:
                os.unlink(tmp_file_name)

    def get_score(self, fmap="", importance_type="weight"):
        op = ModelApplyChunk(
            func=self._get_booster_score, output_types=[OutputType.object]
        )
        if not fmap:
            fmap_data = None
        else:
            with open(fmap, "rb") as fmap_file:
                fmap_data = fmap_file.read()
        return op(self, [{}], fmap=fmap_data, importance_type=importance_type)[0]

    def execute(self, session=None, **kw):
        # The evals_result should be fetched when BoosterData.execute() is called.
        result = super().execute(session=session, **kw)
        if self.op.has_evals_result and self.key == self.op.outputs[0].key:
            self._evals_result.update(self.op.outputs[1].fetch(session=session))
        return result

    def predict(
        self,
        data,
        output_margin=False,
        pred_leaf=False,
        pred_contribs=False,
        approx_contribs=False,
        pred_interactions=False,
        validate_features=True,
        training=False,
        iteration_range=None,
        strict_shape=False,
    ):
        from .predict import predict

        return predict(
            self,
            data,
            output_margin=output_margin,
            pred_leaf=pred_leaf,
            pred_contribs=pred_contribs,
            approx_contribs=approx_contribs,
            pred_interactions=pred_interactions,
            validate_features=validate_features,
            training=training,
            iteration_range=iteration_range,
            strict_shape=strict_shape,
        )


class Booster(Model):
    pass


if not xgboost:
    XGBScikitLearnBase = None
else:

    class XGBScikitLearnBase(xgboost.XGBModel):
        """
        Base class for implementing scikit-learn interface
        """

        def _set_model(
            self, xgb_model: Union[xgboost.XGBModel, xgboost.Booster] = None
        ):
            booster = None
            if isinstance(xgb_model, xgboost.XGBModel):
                booster = xgb_model.get_booster()
            elif isinstance(xgb_model, xgboost.Booster):
                booster = xgb_model

            if booster is not None:
                self._Booster = to_remote_model(booster, model_cls=Booster)

        @classmethod
        def _get_param_names(cls):
            # make sure `xgb_model` not treated as a model param
            names = super()._get_param_names()
            if names:
                names = [p for p in names if p != "xgb_model"]
            return names

        def __repr__(self):
            local_model = self.fetch()
            return repr(local_model)

        def fit(
            self,
            X,
            y,
            sample_weights=None,
            eval_set=None,
            sample_weight_eval_set=None,
            **kw,
        ):
            """
            Fit the regressor. Note that fit() is an eager-execution
            API. The call will be blocked until training finished.

            Parameters
            ----------
            X : array_like
                Feature matrix
            y : array_like
                Labels
            sample_weight : array_like
                instance weights
            eval_set : list, optional
                A list of (X, y) tuple pairs to use as validation sets, for which
                metrics will be computed.
                Validation metrics will help us track the performance of the model.
            sample_weight_eval_set : list, optional
                A list of the form [L_1, L_2, ..., L_n], where each L_i is a list
                of group weights on the i-th validation set.
            """
            raise NotImplementedError

        def predict(self, data, **kw):
            """
            Predict with `data`.

            Parameters
            ----------
              data: data that can be used to perform prediction
            Returns
            -------
            prediction : maxframe.tensor.Tensor
            """
            raise NotImplementedError

        def evals_result(self, **kw) -> Dict:
            """Return the evaluation results.

            If **eval_set** is passed to the :py:meth:`fit` function, you can call
            ``evals_result()`` to get evaluation results for all passed **eval_sets**.  When
            **eval_metric** is also passed to the :py:meth:`fit` function, the
            **evals_result** will contain the **eval_metrics** passed to the :py:meth:`fit`
            function.

            The returned evaluation result is a dictionary:

            .. code-block:: python

                {'validation_0': {'logloss': ['0.604835', '0.531479']},
                 'validation_1': {'logloss': ['0.41965', '0.17686']}}

            Note that evals_result() will be blocked until the train is finished.

            Returns
            -------
            evals_result

            """
            result = super().evals_result()
            if not self._Booster.op.has_evals_result or len(result) != 0:
                return result
            session = kw.pop("session", None)
            run_kwargs = kw.pop("run_kwargs", dict())
            self._Booster.execute(session=session, **run_kwargs)
            return super().evals_result()

        def execute(self, session=None, run_kwargs=None):
            self._Booster.execute(session=session, run_kwargs=run_kwargs)
            return self

        def fetch(self, session=None, run_kwargs=None):
            from xgboost import sklearn as xgb_sklearn

            booster = self._Booster.fetch(session=session, run_kwargs=run_kwargs)
            remote_bst, self._Booster = self._Booster, booster
            try:
                local_cls = getattr(xgb_sklearn, type(self).__name__)
                local_model = local_cls(**self.get_params(deep=True))
                local_model._Booster = booster
                return local_model
            finally:
                self._Booster = remote_bst

        @staticmethod
        def _calc_feature_importance(bst, importance_type, n_features):
            config = json.loads(bst.save_config())
            bst_type = config["learner"]["gradient_booster"]["name"]
            dft = "weight" if bst_type == "gblinear" else "gain"
            importance_type = importance_type or dft
            score = bst.get_score(importance_type=importance_type)
            if bst.feature_names is None:
                feature_names = [f"f{i}" for i in range(n_features)]
            else:
                feature_names = bst.feature_names
            # gblinear returns all features so the `get` in next line is only for gbtree.
            all_features = [score.get(f, 0.0) for f in feature_names]
            all_features_arr = np.array(all_features, dtype=np.float32)
            total = all_features_arr.sum()
            if total == 0:
                return all_features_arr
            return all_features_arr / total

        @property
        def feature_importances_(self):
            op = ModelApplyChunk(
                func=self._calc_feature_importance, output_types=[OutputType.tensor]
            )
            params = {"shape": (self._n_features_in,), "dtype": np.dtype(np.float32)}
            return op(
                self.get_booster(),
                [params],
                importance_type=self.importance_type,
                n_features=self._n_features_in,
            )[0]

    def wrap_evaluation_matrices(
        missing: float,
        X: Any,
        y: Any,
        sample_weight: Optional[Any],
        base_margin: Optional[Any],
        eval_set: Optional[List[Tuple[Any, Any]]],
        sample_weight_eval_set: Optional[List[Any]],
        base_margin_eval_set: Optional[List[Any]],
        label_transform: Callable = lambda x: x,
    ) -> Tuple[Any, Optional[List[Tuple[Any, str]]]]:
        """
        Convert array_like evaluation matrices into DMatrix.
        Perform validation on the way.
        """
        train_dmatrix = DMatrix(
            data=X,
            label=label_transform(y),
            weight=sample_weight,
            base_margin=base_margin,
            missing=missing,
        )

        n_validation = 0 if eval_set is None else len(eval_set)

        def validate_or_none(meta: Optional[List], name: str) -> List:
            if meta is None:
                return [None] * n_validation
            if len(meta) != n_validation:
                raise ValueError(
                    f"{name}'s length does not equal `eval_set`'s length, "
                    + f"expecting {n_validation}, got {len(meta)}"
                )
            return meta

        if eval_set is not None:
            sample_weight_eval_set = validate_or_none(
                sample_weight_eval_set, "sample_weight_eval_set"
            )
            base_margin_eval_set = validate_or_none(
                base_margin_eval_set, "base_margin_eval_set"
            )

            evals = []
            for i, (valid_X, valid_y) in enumerate(eval_set):
                # Skip the duplicated entry.
                if all(
                    (
                        valid_X is X,
                        valid_y is y,
                        sample_weight_eval_set[i] is sample_weight,
                        base_margin_eval_set[i] is base_margin,
                    )
                ):
                    evals.append(train_dmatrix)
                else:
                    m = DMatrix(
                        data=valid_X,
                        label=label_transform(valid_y),
                        weight=sample_weight_eval_set[i],
                        base_margin=base_margin_eval_set[i],
                        missing=missing,
                    )
                    evals.append(m)
            nevals = len(evals)
            eval_names = [f"validation_{i}" for i in range(nevals)]
            evals = list(zip(evals, eval_names))
        else:
            if any(
                meta is not None
                for meta in [
                    sample_weight_eval_set,
                    base_margin_eval_set,
                ]
            ):
                raise ValueError(
                    "`eval_set` is not set but one of the other evaluation meta info is "
                    "not None."
                )
            evals = []

        return train_dmatrix, evals
