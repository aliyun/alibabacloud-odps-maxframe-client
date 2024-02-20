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

import pandas as pd

from ... import opcodes
from ...core import OutputType
from ...serialization.serializables import BoolField
from ..datasource.dataframe import from_pandas
from ..operators import (
    DATAFRAME_TYPE,
    SERIES_TYPE,
    DataFrameOperator,
    DataFrameOperatorMixin,
)
from ..utils import parse_index


class DataFrameAppend(DataFrameOperator, DataFrameOperatorMixin):
    _op_type_ = opcodes.APPEND

    ignore_index = BoolField("ignore_index", default=False)
    verify_integrity = BoolField("verify_integrity", default=False)
    sort = BoolField("sort", default=False)

    def __init__(self, output_types=None, **kw):
        super().__init__(_output_types=output_types, **kw)

    def _call_dataframe(self, df, other):
        if isinstance(other, DATAFRAME_TYPE):
            shape = (df.shape[0] + other.shape[0], df.shape[1])
            inputs = [df, other]
            if self.ignore_index:
                index_value = parse_index(pd.RangeIndex(shape[0]))
            else:
                index_value = parse_index(
                    df.index_value.to_pandas().append(other.index_value.to_pandas())
                )
        elif isinstance(other, list):
            row_length = df.shape[0]
            index = df.index_value.to_pandas()
            for item in other:
                if not isinstance(item, DATAFRAME_TYPE):  # pragma: no cover
                    raise ValueError(f"Invalid type {type(item)} to append")
                row_length += item.shape[0]
                index = index.append(item.index_value.to_pandas())
            shape = (row_length, df.shape[1])
            if self.ignore_index:  # pragma: no cover
                index_value = parse_index(pd.RangeIndex(shape[0]))
            else:
                index_value = parse_index(index)
            inputs = [df] + other
        else:  # pragma: no cover
            raise ValueError(f"Invalid type {type(other)} to append")
        return self.new_dataframe(
            inputs,
            shape=shape,
            dtypes=df.dtypes,
            index_value=index_value,
            columns_value=df.columns_value,
        )

    def _call_series(self, df, other):
        if isinstance(other, SERIES_TYPE):
            shape = (df.shape[0] + other.shape[0],)
            inputs = [df, other]
            if self.ignore_index:
                index_value = parse_index(pd.RangeIndex(shape[0]))
            else:
                index_value = parse_index(
                    df.index_value.to_pandas().append(other.index_value.to_pandas())
                )
        elif isinstance(other, list):
            row_length = df.shape[0]
            index = df.index_value.to_pandas()
            for item in other:
                if not isinstance(item, SERIES_TYPE):  # pragma: no cover
                    raise ValueError(f"Invalid type {type(item)} to append")
                row_length += item.shape[0]
                index = index.append(item.index_value.to_pandas())
            shape = (row_length,)
            if self.ignore_index:  # pragma: no cover
                index_value = parse_index(pd.RangeIndex(shape[0]))
            else:
                index_value = parse_index(index)
            inputs = [df] + other
        else:  # pragma: no cover
            raise ValueError(f"Invalid type {type(other)} to append")
        return self.new_series(
            inputs, shape=shape, dtype=df.dtype, index_value=index_value, name=df.name
        )

    def __call__(self, df, other):
        if isinstance(df, DATAFRAME_TYPE):
            self.output_types = [OutputType.dataframe]
            return self._call_dataframe(df, other)
        else:
            self.output_types = [OutputType.series]
            return self._call_series(df, other)


def append(df, other, ignore_index=False, verify_integrity=False, sort=False):
    if verify_integrity or sort:  # pragma: no cover
        raise NotImplementedError("verify_integrity and sort are not supported now")
    if isinstance(other, dict):
        other = from_pandas(pd.DataFrame(dict((k, [v]) for k, v in other.items())))
    op = DataFrameAppend(
        ignore_index=ignore_index, verify_integrity=verify_integrity, sort=sort
    )
    return op(df, other)
