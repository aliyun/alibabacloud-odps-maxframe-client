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

from numbers import Integral
from typing import List

import numpy as np
import pandas as pd

from ... import opcodes
from ...core import ENTITY_TYPE, EntityData, OutputType
from ...serialization.serializables import AnyField, BoolField
from ...tensor.core import TENSOR_TYPE
from ...tensor.datasource import tensor as astensor
from ..core import DATAFRAME_TYPE, SERIES_TYPE
from ..operators import DataFrameOperator, DataFrameOperatorMixin
from ..utils import parse_index


class SeriesIndex(DataFrameOperator, DataFrameOperatorMixin):
    _op_module_ = "series"
    _op_type_ = opcodes.INDEX

    labels = AnyField("labels", default=None)

    is_intermediate = BoolField("is_intermediate", default=None)

    def __init__(self, output_types=None, **kw):
        super().__init__(_output_types=output_types, **kw)

    def __call__(self, series, name=None):
        return self.new_tileable([series], dtype=series.dtype, name=name)

    def _new_tileables(self, inputs, kws=None, **kw):
        # Override this method to automatically decide the output type,
        # when `labels` is a list, we will set `output_types` as series,
        # otherwise it will be a scalar.
        output_types = getattr(self, "_output_types", None)
        shape = kw.pop("shape", None)
        is_scalar = not isinstance(self.labels, list)
        if not output_types:
            output_types = [OutputType.scalar] if is_scalar else [OutputType.series]
            self.output_types = output_types
        if shape is None:
            shape = () if is_scalar else ((len(self.labels)),)
            kw["shape"] = shape
        if not is_scalar:
            index_value = kw.pop("index_value", None) or parse_index(
                pd.Index(self.labels)
            )
            kw["index_value"] = index_value
        return super()._new_tileables(inputs, kws=kws, **kw)


class DataFrameIndex(DataFrameOperator, DataFrameOperatorMixin):
    _op_type_ = opcodes.INDEX

    col_names = AnyField("col_names", default=None)
    # for bool index
    mask = AnyField("mask", default=None)
    identical_index = BoolField("identical_index")

    def __init__(self, output_types=None, **kw):
        output_types = output_types or [OutputType.series]
        super().__init__(_output_types=output_types, **kw)

    @classmethod
    def _set_inputs(cls, op: "DataFrameIndex", inputs: List[EntityData]):
        super()._set_inputs(op, inputs)
        if isinstance(op.col_names, ENTITY_TYPE):
            op.col_names = op._inputs[0]
        if isinstance(op.mask, ENTITY_TYPE):
            op.mask = op._inputs[-1]

    def __call__(self, df):
        if self.col_names is not None:
            # if col_names is a list, return a DataFrame, else return a Series
            col_names = self.col_names
            if not isinstance(col_names, list):
                col_names = [col_names]
                is_list = False
            else:
                is_list = True

            dtypes_list = df._get_dtypes_by_columns(col_names)
            if is_list or len(dtypes_list) > 1:
                if len(col_names) != len(dtypes_list):
                    col_names = df._get_columns_by_columns(col_names)
                columns = parse_index(pd.Index(col_names), store_data=True)
                return self.new_dataframe(
                    [df],
                    shape=(df.shape[0], len(col_names)),
                    dtypes=pd.Series(dtypes_list, index=col_names, dtype=np.dtype("O")),
                    index_value=df.index_value,
                    columns_value=columns,
                )
            else:
                dtype = dtypes_list[0]
                return self.new_series(
                    [df],
                    shape=(df.shape[0],),
                    dtype=dtype,
                    index_value=df.index_value,
                    name=self.col_names,
                )
        else:
            if isinstance(self.mask, (SERIES_TYPE, DATAFRAME_TYPE, TENSOR_TYPE)):
                index_value = parse_index(
                    pd.Index(
                        [],
                        dtype=df.index_value.to_pandas().dtype,
                        name=df.index_value.name,
                    ),
                    df,
                    self.mask,
                )
                return self.new_dataframe(
                    [df, self.mask],
                    shape=(np.nan, df.shape[1]),
                    dtypes=df.dtypes,
                    index_value=index_value,
                    columns_value=df.columns_value,
                )
            else:
                index_value = parse_index(
                    pd.Index(
                        [],
                        dtype=df.index_value.to_pandas().dtype,
                        name=df.index_value.name,
                    ),
                    df,
                    self.mask,
                )
                return self.new_dataframe(
                    [df],
                    shape=(np.nan, df.shape[1]),
                    dtypes=df.dtypes,
                    index_value=index_value,
                    columns_value=df.columns_value,
                )


_list_like_types = (list, np.ndarray, SERIES_TYPE, pd.Series, TENSOR_TYPE)


def dataframe_getitem(df, item):
    columns_set = set(df.dtypes.keys())

    if isinstance(item, (np.ndarray, pd.Series)) and item.dtype != np.bool_:
        item = item.tolist()

    if isinstance(item, slice):
        edge = item.start if item.start is not None else item.stop
        if isinstance(edge, Integral):
            return df.iloc[item]
        else:
            return df.loc[item]
    elif isinstance(item, list):
        for col_name in item:
            if col_name not in columns_set:
                raise KeyError(f"{col_name} not in columns")
        op = DataFrameIndex(col_names=item, output_types=[OutputType.dataframe])
    elif isinstance(item, _list_like_types) or hasattr(item, "dtypes"):
        # NB: don't enforce the dtype of `item` to be `bool` since it may be unknown
        if isinstance(item, DATAFRAME_TYPE + SERIES_TYPE):
            identical_index = df.index_value.key == item.index_value.key
        else:
            identical_index = False
        op = DataFrameIndex(
            mask=item,
            identical_index=identical_index,
            output_types=[OutputType.dataframe],
        )
    else:
        if item not in columns_set:
            raise KeyError(f"{item} not in columns {columns_set}")
        op = DataFrameIndex(col_names=item)
    return op(df)


def series_getitem(series, labels):
    if isinstance(labels, list) or isinstance(labels, tuple) or np.isscalar(labels):
        op = SeriesIndex(labels=labels)
        return op(series, name=series.name)
    elif isinstance(labels, _list_like_types) and astensor(labels).dtype == np.bool_:
        return series.loc[labels]
    elif isinstance(labels, slice):
        edge = labels.start if labels.start is not None else labels.stop
        if isinstance(edge, Integral):
            return series.iloc[labels]
        else:
            return series.loc[labels]
    else:
        raise NotImplementedError(f"type {type(labels)} is not support for getitem")
