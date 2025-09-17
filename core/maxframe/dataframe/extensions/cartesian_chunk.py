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

from ... import opcodes
from ...core import EntityData, OutputType
from ...serialization.serializables import (
    DictField,
    FunctionField,
    KeyField,
    TupleField,
)
from ...udf import BuiltinFunction
from ...utils import quiet_stdio
from ..operators import DataFrameOperator, DataFrameOperatorMixin
from ..utils import (
    build_df,
    build_empty_df,
    build_series,
    parse_index,
    validate_output_types,
)


class DataFrameCartesianChunk(DataFrameOperator, DataFrameOperatorMixin):
    _op_type_ = opcodes.CARTESIAN_CHUNK

    left = KeyField("left")
    right = KeyField("right")
    func = FunctionField("func")
    args = TupleField("args")
    kwargs = DictField("kwargs")

    def __init__(self, output_types=None, **kw):
        super().__init__(_output_types=output_types, **kw)
        if self.memory_scale is None:
            self.memory_scale = 2.0

    @classmethod
    def _set_inputs(cls, op: "DataFrameCartesianChunk", inputs: List[EntityData]):
        super()._set_inputs(op, inputs)
        op.left, op.right = op.inputs[:2]

    @staticmethod
    def _build_test_obj(obj):
        return (
            build_df(obj, size=2)
            if obj.ndim == 2
            else build_series(obj, size=2, name=obj.name)
        )

    def has_custom_code(self) -> bool:
        return not isinstance(self.func, BuiltinFunction)

    def __call__(self, left, right, index=None, dtypes=None):
        test_left = self._build_test_obj(left)
        test_right = self._build_test_obj(right)
        output_type = self._output_types[0] if self._output_types else None

        if output_type == OutputType.df_or_series:
            return self.new_df_or_series([left, right])

        # try run to infer meta
        try:
            with np.errstate(all="ignore"), quiet_stdio():
                obj = self.func(test_left, test_right, *self.args, **self.kwargs)
        except:  # noqa: E722  # nosec  # pylint: disable=bare-except
            if output_type == OutputType.series:
                obj = pd.Series([], dtype=np.dtype(object))
            elif output_type == OutputType.dataframe and dtypes is not None:
                obj = build_empty_df(dtypes)
            else:
                raise TypeError(
                    "Cannot determine `output_type`, "
                    "you have to specify it as `dataframe` or `series`, "
                    "for dataframe, `dtypes` is required as well "
                    "if output_type='dataframe'"
                )

        if getattr(obj, "ndim", 0) == 1 or output_type == OutputType.series:
            shape = self.kwargs.pop("shape", (np.nan,))
            if index is None:
                index = obj.index
            index_value = parse_index(
                index, left, right, self.func, self.args, self.kwargs
            )
            return self.new_series(
                [left, right],
                dtype=obj.dtype,
                shape=shape,
                index_value=index_value,
                name=obj.name,
            )
        else:
            dtypes = dtypes if dtypes is not None else obj.dtypes
            # dataframe
            shape = (np.nan, len(dtypes))
            columns_value = parse_index(dtypes.index, store_data=True)
            if index is None:
                index = obj.index
            index_value = parse_index(
                index, left, right, self.func, self.args, self.kwargs
            )
            return self.new_dataframe(
                [left, right],
                shape=shape,
                dtypes=dtypes,
                index_value=index_value,
                columns_value=columns_value,
            )


def cartesian_chunk(left, right, func, skip_infer=False, args=(), **kwargs):
    output_type = kwargs.pop("output_type", None)
    output_types = kwargs.pop("output_types", None)
    object_type = kwargs.pop("object_type", None)
    output_types = validate_output_types(
        output_type=output_type, output_types=output_types, object_type=object_type
    )
    output_type = output_types[0] if output_types else None
    if output_type:
        output_types = [output_type]
    elif skip_infer:
        output_types = [OutputType.df_or_series]
    index = kwargs.pop("index", None)
    dtypes = kwargs.pop("dtypes", None)
    memory_scale = kwargs.pop("memory_scale", None)

    op = DataFrameCartesianChunk(
        left=left,
        right=right,
        func=func,
        args=args,
        kwargs=kwargs,
        output_types=output_types,
        memory_scale=memory_scale,
    )
    return op(left, right, index=index, dtypes=dtypes)
