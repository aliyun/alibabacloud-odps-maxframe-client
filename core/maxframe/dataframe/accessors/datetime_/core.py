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

import pandas as pd

from .... import opcodes
from ....core import EntityData, OutputType
from ....serialization.serializables import (
    BoolField,
    DictField,
    KeyField,
    StringField,
    TupleField,
)
from ...operators import DataFrameOperator, DataFrameOperatorMixin
from ...utils import build_empty_series


class SeriesDatetimeMethod(DataFrameOperator, DataFrameOperatorMixin):
    _op_type_ = opcodes.DATETIME_METHOD

    _input = KeyField("input")
    method = StringField("method", default=None)
    method_args = TupleField("method_args", default=None)
    method_kwargs = DictField("method_kwargs", default=None)
    is_property = BoolField("is_property", default=None)

    def __init__(self, output_types=None, **kw):
        super().__init__(_output_types=output_types, **kw)
        if not self.output_types:
            self.output_types = [OutputType.series]

    @property
    def input(self):
        return self._input

    @classmethod
    def _set_inputs(cls, op: "SeriesDatetimeMethod", inputs: List[EntityData]):
        super()._set_inputs(op, inputs)
        op._input = op._inputs[0]

    def __call__(self, inp):
        return datetime_method_to_handlers[self.method].call(self, inp)


class SeriesDatetimeMethodBaseHandler:
    @classmethod
    def call(cls, op, inp):
        empty_series = build_empty_series(inp.dtype)
        if op.is_property:
            test_obj = getattr(empty_series.dt, op.method)
        else:
            test_obj = getattr(empty_series.dt, op.method)(
                *op.method_args, **op.method_kwargs
            )
        dtype = test_obj.dtype
        return op.new_series(
            [inp],
            shape=inp.shape,
            dtype=dtype,
            index_value=inp.index_value,
            name=inp.name,
        )


datetime_method_to_handlers = {}
for method in dir(pd.Series.dt):
    if not method.startswith("_"):
        datetime_method_to_handlers[method] = SeriesDatetimeMethodBaseHandler
