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

from ....errors import TileableNotExecutedError
from ....udf import builtin_function

try:
    import xgboost
except ImportError:  # pragma: no cover
    xgboost = None

from ....core import OutputType, enter_mode, is_kernel_mode
from ...utils.odpsio import ODPSModelMixin, ReadODPSModel
from ..models import ModelApplyChunk, ModelWithEval, ModelWithEvalData, to_remote_model
from .dmatrix import DMatrix

_xgb_type_to_np_type = {
    "float": "float32",
    "int": "int32",
    "i": "bool",
}


class BoosterData(ModelWithEvalData):
    def save_config(self) -> str:
        try:
            return self.fetch().save_config()
        except:
            if is_kernel_mode():
                return "{}"
            raise

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

    @staticmethod
    @builtin_function
    def _get_training_info(bst, evals_result, local_info):
        model_infos = {
            "iteration": bst.num_boosted_rounds(),
        }
        if evals_result:
            model_infos.update(
                dict(
                    duration_ms=evals_result.get("duration_ms"),
                )
            )
        if bst.feature_names:
            model_infos["feature_names"] = bst.feature_names
            model_infos["feature_types"] = [
                _xgb_type_to_np_type[x] for x in bst.feature_types
            ]
        model_infos.update(local_info or {})

        try:
            config = json.loads(bst.save_config())
            stack = [config]
            internal = {}
            while stack:
                obj = stack.pop()
                for k, v in obj.items():
                    if k.endswith("_param"):
                        for p_k, p_v in v.items():
                            internal[p_k] = p_v
                    elif isinstance(v, dict):
                        stack.append(v)

            for k, v in internal.items():
                for t in (int, float, str):
                    try:
                        model_infos[k] = t(v)
                        break
                    except ValueError:
                        continue
        except ValueError:
            pass

        return model_infos

    def get_training_info(self, evals_result: dict = None, local_info: dict = None):
        evals_result = getattr(self, "_evals_result", None) or evals_result
        args = (evals_result, local_info)
        op = ModelApplyChunk(
            func=self._get_training_info, output_types=[OutputType.object]
        )
        return op(self, [{}], args=args)[0]


class Booster(ModelWithEval):
    pass


if not xgboost:
    XGBScikitLearnBase = None
else:

    class XGBScikitLearnBase(xgboost.XGBModel, ODPSModelMixin):
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
            try:
                local_model = self.fetch()
                return repr(local_model)
            except TileableNotExecutedError:
                return f"<{type(self).__name__} unexecuted>"

        def fit(
            self,
            X,
            y,
            sample_weight=None,
            base_margin=None,
            eval_set=None,
            xgb_model=None,
            sample_weight_eval_set=None,
            base_margin_eval_set=None,
            **kw,
        ):
            from .train import train

            session = kw.pop("session", None)
            run_kwargs = kw.pop("run_kwargs", dict())

            self._n_features_in = X.shape[1]

            dtrain, evals = wrap_evaluation_matrices(
                None,
                X,
                y,
                sample_weight,
                base_margin,
                eval_set,
                sample_weight_eval_set,
                base_margin_eval_set,
            )
            with enter_mode(kernel=True):
                params = self.get_xgb_params()
            if not params.get("objective"):
                params["objective"] = "reg:squarederror"
            self.evals_result_ = dict()
            train_kw = {}
            train_kw.update(kw)

            if getattr(self, "n_classes_", None):
                train_kw["num_class"] = self.n_classes_

            result = train(
                params,
                dtrain,
                num_boost_round=self.get_num_boosting_rounds(),
                evals=evals,
                evals_result=self.evals_result_,
                xgb_model=xgb_model,
                callbacks=self.callbacks,
                session=session,
                run_kwargs=run_kwargs,
                **train_kw,
            )
            self._Booster = result
            self.evals_result_t_ = result.op.outputs[-1]
            return self

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

        @property
        def training_info_(self):
            local_info = {}
            attrs = [
                "n_classes_",
                "learning_rate",
            ]
            for attr in attrs:
                if getattr(self, attr, None):
                    local_info[attr] = getattr(self, attr)
            return self._Booster.get_training_info(
                evals_result=self.evals_result_t_, local_info=local_info
            )

        def _get_odps_model_info(self) -> ODPSModelMixin.ODPSModelInfo:
            model_format = (
                "BOOSTED_TREE_CLASSIFIER"
                if hasattr(self, "predict_proba")
                else "BOOSTED_TREE_REGRESSOR"
            )
            return ODPSModelMixin.ODPSModelInfo(
                model_format=model_format, model_params=self._Booster
            )

        @classmethod
        def _build_odps_source_model(cls, op: ReadODPSModel) -> Any:
            if not (
                op.format == "BOOSTED_TREE_CLASSIFIER" and hasattr(cls, "predict_proba")
            ) and not (
                op.format == "BOOSTED_TREE_REGRESSOR"
                and not hasattr(cls, "predict_proba")
            ):
                return None
            op._output_types = [OutputType.object]
            booster = op.new_tileable(None, object_class=Booster)
            estimator = cls()
            estimator._Booster = booster
            return estimator

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
