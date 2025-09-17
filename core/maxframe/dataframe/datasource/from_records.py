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

from typing import MutableMapping, Union

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

    @classmethod
    def estimate_size(
        cls, ctx: MutableMapping[str, Union[int, float]], op: "DataFrameFromRecords"
    ):  # pragma: no cover
        # todo implement this to facilitate local computation
        ctx[op.outputs[0].key] = float("inf")


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
    """
    Convert structured or record ndarray to DataFrame.

    Creates a DataFrame object from a structured ndarray, sequence of
    tuples or dicts, or DataFrame.

    Parameters
    ----------
    data : structured ndarray, sequence of tuples or dicts, or DataFrame
        Structured input data.

        .. deprecated:: 2.1.0
            Passing a DataFrame is deprecated.
    index : str, list of fields, array-like
        Field of array to use as the index, alternately a specific set of
        input labels to use.
    exclude : sequence, default None
        Columns or fields to exclude.
    columns : sequence, default None
        Column names to use. If the passed data do not have names
        associated with them, this argument provides names for the
        columns. Otherwise this argument indicates the order of the columns
        in the result (any names not found in the data will become all-NA
        columns).
    coerce_float : bool, default False
        Attempt to convert values of non-string, non-numeric objects (like
        decimal.Decimal) to floating point, useful for SQL result sets.
    nrows : int, default None
        Number of rows to read if data is an iterator.

    Returns
    -------
    DataFrame

    See Also
    --------
    DataFrame.from_dict : DataFrame from dict of array-like or dicts.
    DataFrame : DataFrame object creation using constructor.

    Examples
    --------
    Data can be provided as a structured ndarray:

    >>> import maxframe.tensor as mt
    >>> import maxframe.dataframe as md
    >>> data = mt.array([(3, 'a'), (2, 'b'), (1, 'c'), (0, 'd')],
    ...                 dtype=[('col_1', 'i4'), ('col_2', 'U1')])
    >>> md.DataFrame.from_records(data).execute()
       col_1 col_2
    0      3     a
    1      2     b
    2      1     c
    3      0     d

    Data can be provided as a list of dicts:

    >>> data = [{'col_1': 3, 'col_2': 'a'},
    ...         {'col_1': 2, 'col_2': 'b'},
    ...         {'col_1': 1, 'col_2': 'c'},
    ...         {'col_1': 0, 'col_2': 'd'}]
    >>> md.DataFrame.from_records(data).execute()
       col_1 col_2
    0      3     a
    1      2     b
    2      1     c
    3      0     d

    Data can be provided as a list of tuples with corresponding columns:

    >>> data = [(3, 'a'), (2, 'b'), (1, 'c'), (0, 'd')]
    >>> md.DataFrame.from_records(data, columns=['col_1', 'col_2']).execute()
       col_1 col_2
    0      3     a
    1      2     b
    2      1     c
    3      0     d
    """
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
