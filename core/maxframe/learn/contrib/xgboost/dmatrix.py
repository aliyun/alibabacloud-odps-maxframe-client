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


from .... import opcodes as OperandDef
from ....core.entity.output_types import get_output_types
from ....core.operator.base import Operator
from ....core.operator.core import TileableOperatorMixin
from ....dataframe.core import DATAFRAME_TYPE
from ....serialization.serializables import Float64Field, KeyField, ListField
from ....serialization.serializables.field import AnyField, Int64Field
from ....tensor import tensor as astensor
from ....tensor.core import TENSOR_TYPE
from ....typing_ import TileableType
from ...utils import convert_to_tensor_or_dataframe


class ToDMatrix(Operator, TileableOperatorMixin):
    _op_type_ = OperandDef.TO_DMATRIX

    data = KeyField("data", default=None)
    label = KeyField("label", default=None)
    missing = Float64Field("missing", default=None)
    weight = KeyField("weight", default=None)
    base_margin = KeyField("base_margin", default=None)
    feature_names = ListField("feature_names", default=None)
    feature_types = ListField("feature_types", default=None)
    feature_weights = AnyField("feature_weights", default=None)
    nthread = Int64Field("nthread", default=None)
    group = AnyField("group", default=None)
    qid = AnyField("qid", default=None)
    label_lower_bound = AnyField("label_lower_bound", default=None)
    label_upper_bound = AnyField("label_upper_bound", default=None)

    @property
    def output_limit(self):
        return 1

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        if self.data is not None:
            self.data = self._inputs[0]
        has_label = self.label is not None
        if has_label:
            self.label = self._inputs[1]
        if self.weight is not None:
            i = 1 if not has_label else 2
            self.weight = self._inputs[i]
        if self.base_margin is not None:
            self.base_margin = self._inputs[-1]

    @staticmethod
    def _get_kw(obj):
        if isinstance(obj, TENSOR_TYPE):
            return {"shape": obj.shape, "dtype": obj.dtype, "order": obj.order}
        else:
            return {
                "shape": obj.shape,
                "dtypes": obj.dtypes,
                "index_value": obj.index_value,
                "columns_value": obj.columns_value,
            }

    def __call__(self):
        inputs = [self.data]
        kw = self._get_kw(self.data)
        if self.label is not None:
            inputs.append(self.label)
        if self.weight is not None:
            inputs.append(self.weight)
        if self.base_margin is not None:
            inputs.append(self.base_margin)

        return self.new_tileable(inputs, **kw)


def check_data(data):
    data = convert_to_tensor_or_dataframe(data)
    if data.ndim != 2:
        raise ValueError(f"Expecting 2-d data, got: {data.ndim}-d")

    return data


def check_array_like(y: TileableType, name: str) -> TileableType:
    if y is None:
        return
    y = convert_to_tensor_or_dataframe(y)
    if isinstance(y, DATAFRAME_TYPE):
        y = y.iloc[:, 0]
    y = astensor(y)
    if y.ndim != 1:
        raise ValueError(f"Expecting 1-d {name}, got: {y.ndim}-d")
    return y


def to_dmatrix(
    data,
    label=None,
    missing=None,
    weight=None,
    base_margin=None,
    feature_names=None,
    feature_types=None,
    feature_weights=None,
    nthread=None,
    group=None,
    qid=None,
    label_lower_bound=None,
    label_upper_bound=None,
):
    data = check_data(data)
    label = check_array_like(label, "label")
    weight = check_array_like(weight, "weight")
    base_margin = check_array_like(base_margin, "base_margin")

    # If not multiple outputs, try to collect the chunks on same worker into one
    # to feed the data into XGBoost for training.
    op = ToDMatrix(
        data=data,
        label=label,
        missing=missing,
        weight=weight,
        base_margin=base_margin,
        feature_names=feature_names,
        feature_types=feature_types,
        feature_weights=feature_weights,
        nthread=nthread,
        group=group,
        qid=qid,
        label_lower_bound=label_lower_bound,
        label_upper_bound=label_upper_bound,
        gpu=data.op.gpu,
        _output_types=get_output_types(data),
    )
    return op()


DMatrix = to_dmatrix
