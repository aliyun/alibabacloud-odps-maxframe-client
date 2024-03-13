#!/usr/bin/env python
# -*- coding: utf-8 -*-
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

import numpy as np
import pandas as pd

from ... import opcodes
from ...core import OutputType
from ...serialization.serializables import BoolField, Int32Field, ListField
from ...tensor.core import TENSOR_TYPE
from ..operators import DataFrameOperator, DataFrameOperatorMixin
from ..utils import parse_index


class DataFrameFromRecords(DataFrameOperator, DataFrameOperatorMixin):
    _op_type_ = opcodes.DATAFRAME_FROM_RECORDS

    columns = ListField("columns", default=None)
    exclude = ListField("exclude", default=None)
    coerce_float = BoolField("coerce_float", default=False)
    nrows = Int32Field("nrows", default=None)

    def __init__(self, index=None, columns=None, **kw):
        if index is not None or columns is not None:
            raise NotImplementedError("Specifying index value is not supported for now")
        super().__init__(columns=columns, _output_types=[OutputType.dataframe], **kw)

    @property
    def input(self):
        return self._inputs[0]

    def __call__(self, data):
        if self.nrows is None:
            nrows = data.shape[0]
        else:
            nrows = self.nrows
        index_value = parse_index(pd.RangeIndex(start=0, stop=nrows))
        dtypes = pd.Series(dict((k, np.dtype(v)) for k, v in data.dtype.descr))
        columns_value = parse_index(pd.Index(data.dtype.names), store_data=True)
        return self.new_dataframe(
            [data],
            (data.shape[0], len(data.dtype.names)),
            dtypes=dtypes,
            index_value=index_value,
            columns_value=columns_value,
        )


def from_records(
    data,
    index=None,
    exclude=None,
    columns=None,
    coerce_float=False,
    nrows=None,
    gpu=None,
    sparse=False,
    **kw
):
    if isinstance(data, np.ndarray):
        from .dataframe import from_pandas

        return from_pandas(
            pd.DataFrame.from_records(
                data,
                index=index,
                exclude=exclude,
                columns=columns,
                coerce_float=coerce_float,
                nrows=nrows,
            ),
            **kw
        )
    elif isinstance(data, TENSOR_TYPE):
        if data.dtype.names is None:
            raise TypeError("Not a tensor with structured dtype {0}", data.dtype)
        if data.ndim != 1:
            raise ValueError(
                "Not a tensor with non 1-D structured dtype {0}", data.shape
            )

        op = DataFrameFromRecords(
            index=None,
            exclude=exclude,
            columns=columns,
            coerce_float=coerce_float,
            nrows=nrows,
            gpu=gpu,
            sparse=sparse,
            **kw
        )
        return op(data)
    else:
        raise TypeError("Not support create DataFrame from {0}", type(data))
