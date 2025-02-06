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

import numpy as np
import pandas as pd

from ... import opcodes
from ...core import OutputType
from ...serialization.serializables import AnyField, BoolField
from ..operators import DataFrameOperator, DataFrameOperatorMixin
from ..utils import parse_index


class DataFrameExplode(DataFrameOperator, DataFrameOperatorMixin):
    _op_type_ = opcodes.EXPLODE

    column = AnyField("column", default=None)
    ignore_index = BoolField("ignore_field", default=None)

    def __init__(self, output_types=None, **kw):
        super().__init__(_output_types=output_types, **kw)

    def _rewrite_params(self, in_obj):
        params = in_obj.params.copy()
        new_shape = list(in_obj.shape)
        new_shape[0] = np.nan
        params["shape"] = tuple(new_shape)

        if self.ignore_index:
            params["index_value"] = parse_index(
                pd.RangeIndex(-1), (in_obj.key, in_obj.index_value.key)
            )
        else:
            params["index_value"] = parse_index(
                None, (in_obj.key, in_obj.index_value.key)
            )
        return params

    def __call__(self, df_or_series):
        return self.new_tileable([df_or_series], **self._rewrite_params(df_or_series))


def df_explode(df, column, ignore_index=False):
    """
    Transform each element of a list-like to a row, replicating index values.

    Parameters
    ----------
    column : str or tuple
        Column to explode.
    ignore_index : bool, default False
        If True, the resulting index will be labeled 0, 1, …, n - 1.

    Returns
    -------
    DataFrame
        Exploded lists to rows of the subset columns;
        index will be duplicated for these rows.

    Raises
    ------
    ValueError :
        if columns of the frame are not unique.

    See Also
    --------
    DataFrame.unstack : Pivot a level of the (necessarily hierarchical)
        index labels.
    DataFrame.melt : Unpivot a DataFrame from wide format to long format.
    Series.explode : Explode a DataFrame from list-like columns to long format.

    Notes
    -----
    This routine will explode list-likes including lists, tuples,
    Series, and np.ndarray. The result dtype of the subset rows will
    be object. Scalars will be returned unchanged. Empty list-likes will
    result in a np.nan for that row.

    Examples
    --------
    >>> import maxframe.tensor as mt
    >>> import maxframe.dataframe as md
    >>> df = md.DataFrame({'A': [[1, 2, 3], 'foo', [], [3, 4]], 'B': 1})
    >>> df.execute()
               A  B
    0  [1, 2, 3]  1
    1        foo  1
    2         []  1
    3     [3, 4]  1

    >>> df.explode('A').execute()
         A  B
    0    1  1
    0    2  1
    0    3  1
    1  foo  1
    2  NaN  1
    3    3  1
    3    4  1
    """
    op = DataFrameExplode(
        column=column, ignore_index=ignore_index, output_types=[OutputType.dataframe]
    )
    return op(df)


def series_explode(series, ignore_index=False):
    """
    Transform each element of a list-like to a row.

    Parameters
    ----------
    ignore_index : bool, default False
        If True, the resulting index will be labeled 0, 1, …, n - 1.

    Returns
    -------
    Series
        Exploded lists to rows; index will be duplicated for these rows.

    See Also
    --------
    Series.str.split : Split string values on specified separator.
    Series.unstack : Unstack, a.k.a. pivot, Series with MultiIndex
        to produce DataFrame.
    DataFrame.melt : Unpivot a DataFrame from wide format to long format.
    DataFrame.explode : Explode a DataFrame from list-like
        columns to long format.

    Notes
    -----
    This routine will explode list-likes including lists, tuples,
    Series, and np.ndarray. The result dtype of the subset rows will
    be object. Scalars will be returned unchanged. Empty list-likes will
    result in a np.nan for that row.

    Examples
    --------
    >>> import maxframe.tensor as mt
    >>> import maxframe.dataframe as md
    >>> s = md.Series([[1, 2, 3], 'foo', [], [3, 4]])
    >>> s.execute()
    0    [1, 2, 3]
    1          foo
    2           []
    3       [3, 4]
    dtype: object

    >>> s.explode().execute()
    0      1
    0      2
    0      3
    1    foo
    2    NaN
    3      3
    3      4
    dtype: object
    """
    op = DataFrameExplode(ignore_index=ignore_index, output_types=[OutputType.series])
    return op(series)
