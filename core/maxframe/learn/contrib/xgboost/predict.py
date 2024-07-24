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

import pickle

import numpy as np
import pandas as pd

from .... import opcodes
from ....core.entity.output_types import OutputType
from ....core.operator.base import Operator
from ....core.operator.core import TileableOperatorMixin
from ....dataframe.utils import parse_index
from ....serialization.serializables import BoolField, BytesField, KeyField, TupleField
from ....tensor.core import TENSOR_TYPE, TensorOrder
from .dmatrix import check_data


class XGBPredict(Operator, TileableOperatorMixin):
    _op_type_ = opcodes.XGBOOST_PREDICT
    output_dtype = np.dtype(np.float32)

    data = KeyField("data", default=None)
    model = BytesField(
        "model", on_serialize=pickle.dumps, on_deserialize=pickle.loads, default=None
    )
    pred_leaf = BoolField("pred_leaf", default=False)
    pred_contribs = BoolField("pred_contribs", default=False)
    approx_contribs = BoolField("approx_contribs", default=False)
    pred_interactions = BoolField("pred_interactions", default=False)
    validate_features = BoolField("validate_features", default=True)
    training = BoolField("training", default=False)
    iteration_range = TupleField("iteration_range", default_factory=lambda x: (0, 0))
    strict_shape = BoolField("strict_shape", default=False)
    flag = BoolField("flag", default=False)

    def __init__(self, output_types=None, gpu=None, **kw):
        super().__init__(_output_types=output_types, gpu=gpu, **kw)

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        self.data = self._inputs[0]
        self.model = self._inputs[1]

    def __call__(self):
        num_class = getattr(self.model.op, "num_class", None)
        if num_class is not None:
            num_class = int(num_class)
        if num_class is not None:
            shape = (self.data.shape[0], num_class)
        else:
            shape = (self.data.shape[0],)
        inputs = [self.data, self.model]
        if self.output_types[0] == OutputType.tensor:
            # tensor
            return self.new_tileable(
                inputs,
                shape=shape,
                dtype=self.output_dtype,
                order=TensorOrder.C_ORDER,
            )
        elif self.output_types[0] == OutputType.dataframe:
            # dataframe
            dtypes = pd.DataFrame(
                np.random.rand(0, num_class), dtype=self.output_dtype
            ).dtypes
            return self.new_tileable(
                inputs,
                shape=shape,
                dtypes=dtypes,
                columns_value=parse_index(dtypes.index),
                index_value=self.data.index_value,
            )
        else:
            # series
            return self.new_tileable(
                inputs,
                shape=shape,
                index_value=self.data.index_value,
                name="predictions",
                dtype=self.output_dtype,
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
    flag=False,
):
    data = check_data(data)
    # TODO: check model datatype

    num_class = getattr(model.op, "num_class", None)
    if isinstance(data, TENSOR_TYPE):
        output_types = [OutputType.tensor]
    elif num_class is not None:
        output_types = [OutputType.dataframe]
    else:
        output_types = [OutputType.series]

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
        flag=flag,
    )()
