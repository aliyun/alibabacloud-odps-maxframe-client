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

from typing import List

import numpy as np
import pandas as pd

from .... import opcodes
from ....core import EntityData
from ....core.entity.output_types import OutputType
from ....core.operator.base import Operator
from ....core.operator.core import TileableOperatorMixin
from ....serialization.serializables import (
    BoolField,
    KeyField,
    ReferenceField,
    TupleField,
)
from ....tensor.core import TensorOrder
from ..models import to_remote_model
from .core import Booster, BoosterData
from .dmatrix import check_data


class XGBPredict(Operator, TileableOperatorMixin):
    _op_type_ = opcodes.XGBOOST_PREDICT
    output_dtype = np.dtype(np.float32)

    data = KeyField("data", default=None)
    model = ReferenceField("model", reference_type=BoosterData, default=None)
    pred_leaf = BoolField("pred_leaf", default=False)
    pred_contribs = BoolField("pred_contribs", default=False)
    approx_contribs = BoolField("approx_contribs", default=False)
    pred_interactions = BoolField("pred_interactions", default=False)
    validate_features = BoolField("validate_features", default=True)
    training = BoolField("training", default=False)
    iteration_range = TupleField("iteration_range", default_factory=lambda x: (0, 0))
    strict_shape = BoolField("strict_shape", default=False)

    def __init__(self, output_types=None, gpu=None, **kw):
        super().__init__(_output_types=output_types, gpu=gpu, **kw)

    def has_custom_code(self) -> bool:
        return True

    @classmethod
    def _set_inputs(cls, op: "XGBPredict", inputs: List[EntityData]):
        super()._set_inputs(op, inputs)
        op.data = op._inputs[0]
        op.model = op._inputs[1]

    def __call__(self):
        num_class = getattr(self.model.op, "num_class", None)
        output_ndim = getattr(self.model.op, "output_ndim", None)
        if num_class is not None and not pd.isna(num_class):
            num_class = int(num_class)
        if num_class is not None and (num_class > 2 or output_ndim == 2):
            shape = (self.data.shape[0], num_class)
        else:
            shape = (self.data.shape[0],)
        inputs = [self.data, self.model]
        return self.new_tileable(
            inputs,
            shape=shape,
            dtype=self.output_dtype,
            order=TensorOrder.C_ORDER,
        )


def predict(
    model,
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
    **kwargs,
):
    """
    Using MaxFrame XGBoost model to predict data.

    Parameters
    ----------
    Parameters are the same as `xgboost.train`. The predict() is lazy-execution mode.

    Returns
    -------
    results: Booster
    """
    import xgboost

    data = check_data(data)
    if not isinstance(model, (Booster, BoosterData, xgboost.Booster)):
        raise TypeError(f"model has to be a xgboost.Booster, got {type(model)} instead")
    elif isinstance(model, xgboost.Booster):
        model = to_remote_model(model, model_cls=Booster)

    output_types = [OutputType.tensor]
    iteration_range = iteration_range or (0, 0)

    return XGBPredict(
        data=data,
        model=model,
        output_margin=output_margin,
        pred_leaf=pred_leaf,
        pred_contribs=pred_contribs,
        approx_contribs=approx_contribs,
        pred_interactions=pred_interactions,
        validate_features=validate_features,
        training=training,
        iteration_range=iteration_range,
        strict_shape=strict_shape,
        gpu=data.op.gpu,
        output_types=output_types,
        **kwargs,
    )()
