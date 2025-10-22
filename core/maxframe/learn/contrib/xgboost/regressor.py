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

from ...utils.odpsio import register_odps_model
from ..utils import make_import_error_func
from .core import XGBScikitLearnBase, xgboost

if not xgboost:
    XGBRegressor = make_import_error_func("xgboost")
else:
    from xgboost.sklearn import XGBRegressorBase

    from .predict import predict

    @register_odps_model
    class XGBRegressor(XGBScikitLearnBase, XGBRegressorBase):
        """
        Implementation of the scikit-learn API for XGBoost regressor.
        """

        def __init__(
            self,
            xgb_model: Union[xgboost.XGBRegressor, xgboost.Booster] = None,
            **kwargs,
        ):
            super().__init__(**kwargs)
            self._set_model(xgb_model)

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
            if y.ndim == 2:
                kw["num_class"] = y.shape[1]
                kw["output_ndim"] = 2
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
            return predict(self.get_booster(), data, **kw)
