# Copyright 1999-2024 Alibaba Group Holding Ltd.
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

from ....tensor import argmax
from ..utils import make_import_error_func
from .core import XGBScikitLearnBase, xgboost

if not xgboost:
    XGBClassifier = make_import_error_func("xgboost")
else:
    from xgboost.sklearn import XGBClassifierBase

    from .core import wrap_evaluation_matrices
    from .predict import predict
    from .train import train

    class XGBClassifier(XGBScikitLearnBase, XGBClassifierBase):
        """
        Implementation of the scikit-learn API for XGBoost classification.
        """

        def fit(
            self,
            X,
            y,
            sample_weight=None,
            base_margin=None,
            eval_set=None,
            sample_weight_eval_set=None,
            base_margin_eval_set=None,
            num_class=None,
        ):
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
            params = self.get_xgb_params()
            self.n_classes_ = num_class or 1
            if self.n_classes_ > 2:
                params["objective"] = "multi:softprob"
                params["num_class"] = self.n_classes_
            else:
                params["objective"] = "binary:logistic"
            self.evals_result_ = dict()
            result = train(
                params,
                dtrain,
                num_boost_round=self.get_num_boosting_rounds(),
                evals=evals,
                evals_result=self.evals_result_,
                num_class=num_class,
            )
            self._Booster = result
            return self

        def predict(self, data, **kw):
            prob = self.predict_proba(data, flag=True, **kw)
            if prob.ndim > 1:
                prediction = argmax(prob, axis=1)
            else:
                prediction = (prob > 0.5).astype(np.int64)
            return prediction

        def predict_proba(self, data, ntree_limit=None, flag=False, **kw):
            if ntree_limit is not None:
                raise NotImplementedError("ntree_limit is not currently supported")
            return predict(self.get_booster(), data, flag=flag, **kw)
