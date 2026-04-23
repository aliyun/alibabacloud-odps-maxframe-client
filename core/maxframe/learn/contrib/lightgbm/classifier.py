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

import numpy as np

from .... import tensor as mt
from ....core import ENTITY_TYPE, OutputType
from ....tensor.merge.vstack import _vstack
from ....udf import builtin_function
from ...preprocessing import LabelEncoder
from ...utils import check_classification_targets
from ...utils.checks import assert_all_finite
from ..models import to_remote_model
from ..utils import make_import_error_func
from .core import Booster, LGBMScikitLearnBase

try:
    import lightgbm
except ImportError:
    lightgbm = None


if not lightgbm:
    LGBMClassifier = make_import_error_func("lightgbm")
else:

    class LGBMClassifier(LGBMScikitLearnBase, lightgbm.LGBMClassifier):
        _default_objective = "binary"

        @staticmethod
        @builtin_function
        def _extract_clf_booster(model, array_module=np):
            if isinstance(model, lightgbm.LGBMModel):
                bst = model.booster_
                if getattr(model, "_le", None) is not None:
                    le_classes = array_module.array(model._le.classes_)
                else:
                    le_classes = array_module.arange(bst.params.get("num_class", 2))
            elif isinstance(model, lightgbm.Booster):
                bst = model
                le_classes = array_module.arange(model.params.get("num_class", 2))
            else:
                bst, le_classes = None, None
            return bst, le_classes

        def __init__(self, *args, **kwargs):
            if args and isinstance(args[0], ENTITY_TYPE):
                booster = args[0]
                extra_kw = {
                    "output_type": OutputType.tensor,
                    "shape": (np.nan,),
                    "dtype": np.dtype("O"),
                }
                self._le = LabelEncoder()
                self._Booster, self._le.classes_ = to_remote_model(
                    booster,
                    model_cls=Booster,
                    extractor=self._extract_clf_booster,
                    extra_kws=[extra_kw],
                )
                self._objective = self._default_objective
            elif args:
                bst, le_classes = self._extract_clf_booster(args[0], array_module=mt)
                if bst is not None:
                    self._le = LabelEncoder()
                    self._le.classes_ = le_classes
            super().__init__(*args, **kwargs)

        def fit(
            self,
            X,
            y,
            *,
            sample_weight=None,
            init_score=None,
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
            classes=None,
            **kwargs
        ):
            """
            unsupported features:
            1. eval_metric
            """
            if classes:
                self._le = LabelEncoder()
                self._le.classes_ = mt.array(classes)
            else:
                y = assert_all_finite(y, check_only=False)
                _, y = check_classification_targets(y, return_value=True)
                self._le = LabelEncoder().fit(y)
                _y = self._le.transform(y, execute=False)

            num_class = self._le.classes_.shape[0]
            if num_class > 2:
                # Switch to using a multiclass objective in the underlying LGBM instance
                ova_aliases = {"multiclassova", "multiclass_ova", "ova", "ovr"}
                if self._objective not in ova_aliases and not callable(self._objective):
                    self._objective = "multiclass"

            if not callable(eval_metric):
                if isinstance(eval_metric, (str, type(None))):
                    eval_metric = [eval_metric]
                if num_class > 2:
                    for index, metric in enumerate(eval_metric):
                        if metric in {"logloss", "binary_logloss"}:
                            eval_metric[index] = "multi_logloss"
                        elif metric in {"error", "binary_error"}:
                            eval_metric[index] = "multi_error"
                else:
                    for index, metric in enumerate(eval_metric):
                        if metric in {"logloss", "multi_logloss"}:
                            eval_metric[index] = "binary_logloss"
                        elif metric in {"error", "multi_error"}:
                            eval_metric[index] = "binary_error"

            # do not modify args, as it causes errors in model selection tools
            valid_sets = None
            if eval_set is not None:
                if isinstance(eval_set, tuple):
                    eval_set = [eval_set]
                valid_sets = [None] * len(eval_set)
                for i, (valid_x, valid_y) in enumerate(eval_set):
                    if valid_x is X and valid_y is y:
                        valid_sets[i] = (valid_x, _y)
                    else:
                        valid_sets[i] = (
                            valid_x,
                            self._le.transform(valid_y, execute=False),
                        )

            super().fit(
                X,
                y,
                sample_weight=sample_weight,
                init_score=init_score,
                eval_set=valid_sets,
                eval_names=eval_names,
                eval_sample_weight=eval_sample_weight,
                eval_init_score=eval_init_score,
                eval_group=eval_group,
                eval_metric=eval_metric,
                early_stopping_rounds=early_stopping_rounds,
                verbose=verbose,
                feature_name=feature_name,
                categorical_feature=categorical_feature,
                callbacks=callbacks,
                init_model=init_model,
            )
            return self

        def predict(
            self,
            X,
            raw_score=False,
            start_iteration=0,
            num_iteration=None,
            pred_leaf=False,
            pred_contrib=False,
            **kwargs
        ):
            result = self.predict_proba(
                X,
                raw_score,
                start_iteration,
                num_iteration,
                pred_leaf,
                pred_contrib,
                **kwargs
            )
            if callable(self._objective) or raw_score or pred_leaf or pred_contrib:
                return result
            else:
                class_index = mt.argmax(result, axis=1)
                return self._le.inverse_transform(class_index)

        def predict_proba(
            self,
            X,
            raw_score=False,
            start_iteration=0,
            num_iteration=None,
            pred_leaf=False,
            pred_contrib=False,
            **kwargs
        ):
            prediction = super().predict(
                X,
                raw_score,
                start_iteration,
                num_iteration,
                pred_leaf,
                pred_contrib,
                **kwargs
            )
            if len(prediction.shape) == 2 and prediction.shape[1] == self.n_classes_:
                # multi-class
                return prediction
            if (
                len(prediction.shape) == 2
                and self.n_classes_ == 2
                and prediction.shape[1] >= self.n_classes_
            ):
                # multi-label
                return prediction
            # binary logistic function
            classone_probs = prediction
            classzero_probs = 1.0 - classone_probs
            return mt.transpose(_vstack((classzero_probs, classone_probs)))
