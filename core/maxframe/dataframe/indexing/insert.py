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

from ... import opcodes
from ...core import EntityData, get_output_types
from ...serialization.serializables import AnyField, BoolField, Int64Field
from ...tensor.core import TENSOR_TYPE
from ..core import INDEX_TYPE, SERIES_TYPE
from ..operators import DataFrameOperator, DataFrameOperatorMixin
from ..utils import build_empty_df, parse_index


class DataFrameInsert(DataFrameOperator, DataFrameOperatorMixin):
    _op_type_ = opcodes.INSERT

    loc = Int64Field("loc")
    column = AnyField("column", default=None)
    value = AnyField("value", default=None)
    allow_duplicates = BoolField("allow_duplicates", default=False)

    @classmethod
    def _set_inputs(cls, op: "DataFrameInsert", inputs: List[EntityData]):
        super()._set_inputs(op, inputs)
        if len(inputs) > 1:
            op.value = op._inputs[-1]

    def __call__(self, df):
        self._output_types = get_output_types(df)
        inputs = [df]
        if isinstance(self.value, (SERIES_TYPE, TENSOR_TYPE)):
            value_dtype = self.value.dtype
            inputs.append(self.value)
        else:
            value_dtype = pd.Series(self.value).dtype

        params = df.params

        if df.ndim == 2:
            empty_obj = build_empty_df(df.dtypes)
            empty_obj.insert(
                loc=self.loc,
                column=self.column,
                allow_duplicates=self.allow_duplicates,
                value=pd.Series([], dtype=value_dtype),
            )

            params["columns_value"] = parse_index(empty_obj.columns, store_data=True)
            params["dtypes"] = empty_obj.dtypes
            params["shape"] = (df.shape[0], df.shape[1] + 1)
        else:
            assert isinstance(df, INDEX_TYPE)
            params["index_value"] = parse_index(
                df.index_value, type(self), df, self.loc, self.value
            )
            params["shape"] = (df.shape[0] + 1,)
        return self.new_tileable(inputs, **params)


def df_insert(df, loc, column, value, allow_duplicates=False):
    """
    Insert column into DataFrame at specified location.

    Raises a ValueError if `column` is already contained in the DataFrame,
    unless `allow_duplicates` is set to True.

    Parameters
    ----------
    loc : int
        Insertion index. Must verify 0 <= loc <= len(columns).
    column : str, number, or hashable object
        Label of the inserted column.
    value : int, Series, or array-like
    allow_duplicates : bool, optional
    """
    if isinstance(value, TENSOR_TYPE) and value.ndim > 1:
        raise ValueError(
            f"Wrong number of items passed {value.ndim}, placement implies 1"
        )

    op = DataFrameInsert(
        loc=loc, column=column, value=value, allow_duplicates=allow_duplicates
    )
    out_df = op(df)
    df.data = out_df.data


def index_insert(idx, loc, value):
    """
    Make new Index inserting new item at location.

    Follows Python list.append semantics for negative values.

    Parameters
    ----------
    loc : int
    item : object

    Returns
    -------
    new_index : Index
    """
    op = DataFrameInsert(loc=loc, value=value)
    return op(idx)
