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

from .... import opcodes
from ....core import EntityData
from ....core.entity.output_types import get_output_types
from ....core.operator.base import Operator
from ....core.operator.core import TileableOperatorMixin
from ....dataframe.core import DATAFRAME_TYPE
from ....serialization.serializables import BoolField, KeyField, ListField
from ....serialization.serializables.field import AnyField
from ....tensor import tensor as astensor
from ....tensor.core import TENSOR_TYPE
from ....typing_ import TileableType
from ...utils import convert_to_tensor_or_dataframe


class ToLGBMDataset(Operator, TileableOperatorMixin):
    _op_type_ = opcodes.TO_LGBM_DATASET

    data = KeyField("data", default=None)
    label = KeyField("label", default=None)
    reference = KeyField("reference", default=None)
    weight = KeyField("weight", default=None)
    init_score = KeyField("init_score", default=None)
    group = AnyField("group", default=None)
    feature_name = ListField("feature_name", default=None)
    categorical_feature = ListField("categorical_feature", default=None)
    params = AnyField("params", default=None)
    free_raw_data = BoolField("free_raw_data", default=None)
    position = AnyField("position", default=None)
    # if to collocate the data, label and weight
    collocate = BoolField("collocate", default=False)

    @property
    def output_limit(self):
        if self.collocate:
            return 1 + sum(bool(x) for x in [self.label, self.weight, self.init_score])
        return 1

    @classmethod
    def _set_inputs(cls, op: "ToLGBMDataset", inputs: List[EntityData]):
        super()._set_inputs(op, inputs)
        if op.data is not None:
            op.data = op._inputs[0]
        has_label = op.label is not None
        if has_label:
            op.label = op._inputs[1]
        if op.weight is not None:
            i = 1 if not has_label else 2
            op.weight = op._inputs[i]
        if op.init_score is not None:
            op.init_score = op._inputs[-1]

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
        if self.init_score is not None:
            inputs.append(self.init_score)

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
    return astensor(y)


def to_lgbm_dataset(
    data,
    label=None,
    reference=None,
    weight=None,
    group=None,
    init_score=None,
    feature_name="auto",
    categorical_feature="auto",
    params=None,
    free_raw_data=True,
    position=None,
):
    data = check_data(data)
    label = check_array_like(label, "label")
    weight = check_array_like(weight, "weight")
    init_score = check_array_like(init_score, "init_score")

    if weight is not None and weight.ndim > 1:
        raise ValueError("weight must be 1-dimensional")

    # If not multiple outputs, try to collect the chunks on same worker into one
    # to feed the data into LightGBM for training.
    op = ToLGBMDataset(
        data=data,
        label=label,
        reference=reference,
        weight=weight,
        group=group,
        init_score=init_score,
        feature_name=None if feature_name == "auto" else feature_name,
        categorical_feature=None
        if categorical_feature == "auto"
        else categorical_feature,
        params=params,
        free_raw_data=free_raw_data,
        position=position,
        gpu=data.op.gpu,
        _output_types=get_output_types(data),
    )
    return op()


Dataset = to_lgbm_dataset
