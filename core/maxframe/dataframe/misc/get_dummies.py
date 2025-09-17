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

from ...core import EntityData, OutputType
from ...serialization.serializables import (
    AnyField,
    BoolField,
    KeyField,
    ListField,
    StringField,
)
from ...utils import make_dtype, pd_release_version
from ..datasource.dataframe import from_pandas as from_pandas_df
from ..datasource.series import from_pandas as from_pandas_series
from ..initializer import Series as asseries
from ..operators import DataFrameOperator, DataFrameOperatorMixin

_encoding_dtype_kind = ["O", "S", "U"]
_ret_uint8 = pd_release_version < (2, 0, 0)


class DataFrameGetDummies(DataFrameOperator, DataFrameOperatorMixin):
    prefix = AnyField("prefix", default=None)
    prefix_sep = StringField("prefix_sep", default=None)
    dummy_na = BoolField("dummy_na", default=None)
    columns = ListField("columns", default=None)
    sparse = BoolField("sparse", default=None)
    drop_first = BoolField("drop_first", default=None)
    dtype = AnyField("dtype", default=None)

    agg_results = KeyField("agg_results", default=None)

    def __init__(self, **kw):
        super().__init__(**kw)
        self.output_types = [OutputType.dataframe]

    @classmethod
    def _set_inputs(cls, op: "DataFrameGetDummies", inputs: List[EntityData]):
        super()._set_inputs(op, inputs)
        if op.agg_results is not None:  # pragma: no branch
            op.agg_results = inputs[-1]

    def __call__(self, data):
        if not self.columns:
            self.agg_results = data.agg(["unique"])
        else:
            self.agg_results = data[self.columns].agg(["unique"])

        return self.new_tileable(
            [data, self.agg_results],
            shape=(np.nan, np.nan),
            dtypes=None,
            index_value=data.index_value,
            columns_value=None,
        )


def get_dummies(
    data,
    prefix=None,
    prefix_sep="_",
    dummy_na=False,
    columns=None,
    sparse=False,
    drop_first=False,
    dtype=None,
):
    """
    Convert categorical variable into dummy/indicator variables.

    Parameters
    ----------
    data : array-like, Series, or DataFrame
        Data of which to get dummy indicators.
    prefix : str, list of str, or dict of str, default None
        String to append DataFrame column names.
        Pass a list with length equal to the number of columns
        when calling get_dummies on a DataFrame. Alternatively, `prefix`
        can be a dictionary mapping column names to prefixes.
    prefix_sep : str, default '_'
        If appending prefix, separator/delimiter to use. Or pass a
        list or dictionary as with `prefix`.
    dummy_na : bool, default False
        Add a column to indicate NaNs, if False NaNs are ignored.
    columns : list-like, default None
        Column names in the DataFrame to be encoded.
        If `columns` is None then all the columns with
        `object` or `category` dtype will be converted.
    sparse : bool, default False
        Whether the dummy-encoded columns should be backed by
        a :class:`SparseArray` (True) or a regular NumPy array (False).
    drop_first : bool, default False
        Whether to get k-1 dummies out of k categorical levels by removing the
        first level.
    dtype : dtype, default bool
        Data type for new columns. Only a single dtype is allowed.

    Returns
    -------
    DataFrame
        Dummy-coded data.

    Examples
    --------
    >>> import maxframe.dataframe as md
    >>> s = md.Series(list('abca'))

    >>> md.get_dummies(s).execute()
       a  b  c
    0  1  0  0
    1  0  1  0
    2  0  0  1
    3  1  0  0

    >>> s1 = ['a', 'b', np.nan]

    >>> md.get_dummies(s1).execute()
       a  b
    0  1  0
    1  0  1
    2  0  0

    >>> md.get_dummies(s1, dummy_na=True).execute()
       a  b  NaN
    0  1  0    0
    1  0  1    0
    2  0  0    1

    >>> df = md.DataFrame({'A': ['a', 'b', 'a'], 'B': ['b', 'a', 'c'],
    ...                    'C': [1, 2, 3]})

    >>> md.get_dummies(df, prefix=['col1', 'col2']).execute()
       C  col1_a  col1_b  col2_a  col2_b  col2_c
    0  1       1       0       0       1       0
    1  2       0       1       1       0       0
    2  3       1       0       0       0       1

    >>> md.get_dummies(pd.Series(list('abcaa'))).execute()
       a  b  c
    0  1  0  0
    1  0  1  0
    2  0  0  1
    3  1  0  0
    4  1  0  0

    >>> md.get_dummies(pd.Series(list('abcaa')), drop_first=True).execute()
       b  c
    0  0  0
    1  1  0
    2  0  1
    3  0  0
    4  0  0

    >>> md.get_dummies(pd.Series(list('abc')), dtype=float).execute()
         a    b    c
    0  1.0  0.0  0.0
    1  0.0  1.0  0.0
    2  0.0  0.0  1.0
    """
    if columns is not None and not isinstance(columns, list):
        raise TypeError("Input must be a list-like for parameter `columns`")

    if isinstance(data, (list, tuple)):
        data = asseries(data)
    elif isinstance(data, pd.Series):
        data = from_pandas_series(data)
    elif isinstance(data, pd.DataFrame):
        data = from_pandas_df(data)

    dtype = make_dtype(
        dtype if dtype is not None else np.dtype(np.uint8 if _ret_uint8 else bool)
    )

    if prefix is not None:
        if isinstance(prefix, list):
            if columns is not None:
                encoding_col_num = len(columns)
            else:
                encoding_col_num = 0
                for dt in data.dtypes.values:
                    if dt.kind in _encoding_dtype_kind:
                        encoding_col_num += 1
            prefix_num = len(prefix)
            if prefix_num != encoding_col_num:
                raise ValueError(
                    f"Length of 'prefix' ({prefix_num}) did not match "
                    + f"the length of the columns being encoded ({encoding_col_num})"
                )
        elif isinstance(prefix, dict):
            if columns is not None:
                encoding_col_num = len(columns)
                prefix_num = len(prefix)
                if prefix_num != encoding_col_num:
                    raise ValueError(
                        f"Length of 'prefix' ({prefix_num}) did not match "
                        + f"the length of the columns being encoded ({encoding_col_num})"
                    )
                prefix_cols = prefix.keys()
                for columns_columnname, prefix_columnname in zip(
                    prefix_cols, list(columns)
                ):
                    if columns_columnname != prefix_columnname:
                        raise KeyError(f"{columns_columnname}")
            else:
                columns = list(prefix.keys())
            # Convert prefix from dict to list, to simplify tile work
            prefix = list(prefix.values())

    if not columns and data.ndim == 2:
        columns = []
        for col_name, dt in data.dtypes.items():
            if dt.kind in _encoding_dtype_kind:
                columns.append(col_name)

    op = DataFrameGetDummies(
        prefix=prefix,
        prefix_sep=prefix_sep,
        dummy_na=dummy_na,
        columns=columns,
        sparse=sparse,
        drop_first=drop_first,
        dtype=dtype,
    )

    return op(data)
