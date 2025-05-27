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
from ...serialization.serializables import AnyField, BoolField, StringField
from ..operators import DataFrameOperator, DataFrameOperatorMixin, OutputType
from ..utils import build_empty_df, parse_index


class DataFrameMelt(DataFrameOperator, DataFrameOperatorMixin):
    _op_type_ = opcodes.MELT

    id_vars = AnyField("id_vars", default=None)
    value_vars = AnyField("value_vars", default=None)
    var_name = StringField("var_name", default=None)
    value_name = StringField("value_name", default=None)
    col_level = AnyField("col_level", default=None)
    ignore_index = BoolField("ignore_index", default=False)

    def __call__(self, df):
        empty_result = build_empty_df(df.dtypes).melt(
            id_vars=self.id_vars,
            value_vars=self.value_vars,
            var_name=self.var_name,
            value_name=self.value_name,
            col_level=self.col_level,
            ignore_index=self.ignore_index,
        )
        self._output_types = [OutputType.dataframe]
        return self.new_tileable(
            [df],
            shape=(np.nan, len(empty_result.columns)),
            dtypes=empty_result.dtypes,
            index_value=parse_index(pd.RangeIndex(-1), df.key, df.index_value.key),
            columns_value=parse_index(empty_result.columns, store_data=True),
        )


def melt(
    frame,
    id_vars=None,
    value_vars=None,
    var_name=None,
    value_name="value",
    col_level=None,
    ignore_index=False,
):
    """
    Unpivot a DataFrame from wide to long format, optionally leaving identifiers set.

    This function is useful to massage a DataFrame into a format where one
    or more columns are identifier variables (`id_vars`), while all other
    columns, considered measured variables (`value_vars`), are "unpivoted" to
    the row axis, leaving just two non-identifier columns, 'variable' and
    'value'.

    Parameters
    ----------
    id_vars : tuple, list, or ndarray, optional
        Column(s) to use as identifier variables.
    value_vars : tuple, list, or ndarray, optional
        Column(s) to unpivot. If not specified, uses all columns that
        are not set as `id_vars`.
    var_name : scalar
        Name to use for the 'variable' column. If None it uses
        ``frame.columns.name`` or 'variable'.
    value_name : scalar, default 'value'
        Name to use for the 'value' column.
    col_level : int or str, optional
        If columns are a MultiIndex then use this level to melt.
    ignore_index : bool, default True
        If True, original index is ignored. If False, the original index
        is retained. Index labels will be repeated as necessary.

    Returns
    -------
    DataFrame
        Unpivoted DataFrame.

    See Also
    --------
    melt
    pivot_table
    DataFrame.pivot
    Series.explode

    Examples
    --------
    >>> import maxframe.dataframe as md
    >>> df = md.DataFrame({'A': {0: 'a', 1: 'b', 2: 'c'},
    ...                    'B': {0: 1, 1: 3, 2: 5},
    ...                    'C': {0: 2, 1: 4, 2: 6}})
    >>> df.execute()
       A  B  C
    0  a  1  2
    1  b  3  4
    2  c  5  6

    >>> df.melt(id_vars=['A'], value_vars=['B']).execute()
       A variable  value
    0  a        B      1
    1  b        B      3
    2  c        B      5

    >>> df.melt(id_vars=['A'], value_vars=['B', 'C']).execute()
       A variable  value
    0  a        B      1
    1  b        B      3
    2  c        B      5
    3  a        C      2
    4  b        C      4
    5  c        C      6

    The names of 'variable' and 'value' columns can be customized:

    >>> df.melt(id_vars=['A'], value_vars=['B'],
    ...         var_name='myVarname', value_name='myValname').execute()
       A myVarname  myValname
    0  a         B          1
    1  b         B          3
    2  c         B          5

    If you have multi-index columns:

    >>> df = md.DataFrame({('A', 'D'): {0: 'a', 1: 'b', 2: 'c'},
    ...                    ('B', 'E'): {0: 1, 1: 3, 2: 5},
    ...                    ('C', 'F'): {0: 2, 1: 4, 2: 6}})
    >>> df.execute()
       A  B  C
       D  E  F
    0  a  1  2
    1  b  3  4
    2  c  5  6

    >>> df.melt(col_level=0, id_vars=['A'], value_vars=['B']).execute()
       A variable  value
    0  a        B      1
    1  b        B      3
    2  c        B      5

    >>> df.melt(id_vars=[('A', 'D')], value_vars=[('B', 'E')]).execute()
      (A, D) variable_0 variable_1  value
    0      a          B          E      1
    1      b          B          E      3
    2      c          B          E      5
    """
    op = DataFrameMelt(
        id_vars=id_vars,
        value_vars=value_vars,
        var_name=var_name,
        value_name=value_name,
        col_level=col_level,
        ignore_index=ignore_index,
    )
    return op(frame)
