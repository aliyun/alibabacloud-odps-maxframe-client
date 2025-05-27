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

from typing import Any, Dict, List

import numpy as np

from .... import opcodes
from ....core import OutputType
from ....core.operator import Operator, TileableOperatorMixin
from ....serialization.serializables import BoolField, DictField, Int32Field, KeyField
from ....tensor.core import TensorOrder
from ....typing_ import EntityType
from ..models import to_remote_model
from .dataset import check_data


class LGBMPredict(Operator, TileableOperatorMixin):
    _op_type_ = opcodes.LGBM_PREDICT
    _op_module_ = "learn.contrib.lightgbm"
    output_dtype = np.dtype(np.float64)

    data = KeyField("data", default=None)
    booster = KeyField("booster", default=None)
    start_iteration = Int32Field("start_iteration", default=0)
    num_iterations = Int32Field("num_iterations", default=None)
    raw_score = BoolField("raw_score", default=False)
    pred_leaf = BoolField("pred_leaf", default=False)
    pred_contrib = BoolField("pred_contrib", default=False)
    validate_features = BoolField("validate_features", default=False)
    kwds = DictField("kwds", default=None)

    def __init__(self, output_types=None, **kw):
        super().__init__(_output_types=output_types, **kw)

    def has_custom_code(self) -> bool:
        return True

    @property
    def output_limit(self) -> int:
        return 1 + self.pred_leaf + self.pred_contrib

    @classmethod
    def _set_inputs(cls, op: "LGBMPredict", inputs: List[EntityType]):
        super()._set_inputs(op, inputs)
        it = iter(inputs)
        op.data = next(it)
        op.booster = next(it)

    def __call__(self):
        num_class = getattr(self.booster.op, "num_class", None)

        result_kw: Dict[str, Any] = {
            "dtype": self.output_dtype,
            "order": TensorOrder.C_ORDER,
        }
        kws = [result_kw]

        if num_class is not None:
            num_class = int(num_class)
        if num_class is not None and num_class > 2:
            result_kw["shape"] = (self.data.shape[0], num_class)
        else:
            result_kw["shape"] = (self.data.shape[0],)

        if self.pred_leaf:
            kws.append(
                {
                    "shape": (np.nan, np.nan),
                    "dtype": np.dtype(np.float_),
                    "order": TensorOrder.C_ORDER,
                }
            )
        if self.pred_contrib:
            kws.append(
                {
                    "shape": (np.nan, np.nan),
                    "dtype": np.dtype(np.float_),
                    "order": TensorOrder.C_ORDER,
                }
            )

        return self.new_tileables([self.data, self.booster], kws=kws)


def predict(
    booster,
    data,
    raw_score: bool = False,
    start_iteration: int = 0,
    num_iteration: int = None,
    pred_leaf: bool = False,
    pred_contrib: bool = False,
    validate_features: bool = False,
    **kwargs,
):
    import lightgbm

    from .core import Booster, BoosterData

    if not isinstance(booster, (Booster, BoosterData, lightgbm.Booster)):
        raise TypeError(
            f"model has to be a lightgbm.Booster, got {type(booster)} instead"
        )
    elif isinstance(booster, lightgbm.Booster):
        booster = to_remote_model(booster, model_cls=Booster)
    proba = kwargs.pop("proba", False)

    data = check_data(data)

    op = LGBMPredict(
        data=data,
        booster=booster,
        output_types=[OutputType.tensor],
        proba=proba,
        raw_score=raw_score,
        start_iteration=start_iteration,
        num_iteration=num_iteration,
        pred_leaf=pred_leaf,
        pred_contrib=pred_contrib,
        validate_features=validate_features,
        kwds=kwargs,
    )
    results = op()
    if len(results) == 1:
        return results[0]
    return results
