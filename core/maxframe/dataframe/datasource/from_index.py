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

from typing import List

from maxframe import opcodes
from maxframe.core import EntityData
from maxframe.dataframe.initializer import Index
from maxframe.dataframe.operators import DataFrameOperator, DataFrameOperatorMixin
from maxframe.serialization.serializables import AnyField, KeyField


class SeriesFromIndex(DataFrameOperator, DataFrameOperatorMixin):
    _op_type_ = opcodes.SERIES_FROM_INDEX

    input_ = KeyField("input_")
    index = KeyField("index")
    name = AnyField("name", default=None)

    @classmethod
    def _set_inputs(cls, op: "SeriesFromIndex", inputs: List[EntityData]):
        super()._set_inputs(op, inputs)
        op.input_ = op._inputs[0]
        if len(op._inputs) > 1:
            op.index = op._inputs[1]

    def __call__(self, index, new_index=None, name=None):
        inputs = [index]
        index_value = index.index_value
        if new_index is not None:
            inputs.append(new_index)
            index_value = new_index.index_value
        return self.new_series(
            inputs,
            shape=index.shape,
            dtype=index.dtype,
            index_value=index_value,
            name=name,
        )


def series_from_index(ind, index=None, name=None):
    name = name or ind.name
    if index is not None:
        index = Index(index)
    op = SeriesFromIndex(input_=ind, index=index, name=name)
    return op(ind, new_index=index, name=name)
