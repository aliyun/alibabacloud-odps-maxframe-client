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

from typing import Union

import numpy as np

from .... import tensor as mt
from ....tensor.merge.vstack import _vstack
from ...utils.odpsio import register_odps_model
from ..utils import make_import_error_func
from .core import XGBScikitLearnBase, xgboost

if not xgboost:
    XGBClassifier = make_import_error_func("xgboost")
else:
    from xgboost.sklearn import XGBClassifierBase

    from .predict import predict

    @register_odps_model
    class XGBClassifier(XGBScikitLearnBase, XGBClassifierBase):
        """
        Implementation of the scikit-learn API for XGBoost classification.
        """

        def __init__(
            self,
            xgb_model: Union[xgboost.XGBClassifier, xgboost.Booster] = None,
            **kwargs,
        ):
            super().__init__(**kwargs)
            self._set_model(xgb_model)

        def get_xgb_params(self):
            params = super().get_xgb_params()
            if self.n_classes_ > 2:
                params["objective"] = "multi:softprob"
                params["num_class"] = self.n_classes_
            else:
                params["objective"] = "binary:logistic"
            return params

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
            num_class=None,
            **kw,
        ):
            session = kw.pop("session", None)
            run_kwargs = kw.pop("run_kwargs", dict())

            if num_class is not None:
                self.n_classes_ = num_class
            else:
                t_labels = mt.unique(y).execute(session=session, **run_kwargs)
                self.n_classes_ = t_labels.shape[0]

            super().fit(
                X,
                y,
                sample_weight=sample_weight,
                base_margin=base_margin,
                eval_set=eval_set,
                xgb_model=xgb_model,
                sample_weight_eval_set=sample_weight_eval_set,
                base_margin_eval_set=base_margin_eval_set,
                **kw,
            )
            return self

        def predict(self, data, **kw):
            prob = self.predict_proba(data, flag=True, **kw)
            if prob.ndim > 1:
                prediction = mt.argmax(prob, axis=1)
            else:
                prediction = (prob > 0.5).astype(np.int64)
            return prediction

        def predict_proba(self, data, ntree_limit=None, flag=False, **kw):
            if ntree_limit is not None:
                raise NotImplementedError("ntree_limit is not currently supported")
            prediction = predict(self.get_booster(), data, flag=flag, **kw)
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

        @property
        def classes_(self) -> np.ndarray:
            return np.arange(self.n_classes_)
