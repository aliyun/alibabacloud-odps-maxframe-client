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

from ... import opcodes
from ...serialization.serializables import AnyField, Int32Field
from ..core import INDEX_TYPE
from ..operators import DataFrameOperator, DataFrameOperatorMixin
from ..utils import build_df, build_series, parse_index, validate_axis


class DataFrameDropLevel(DataFrameOperator, DataFrameOperatorMixin):
    _op_type_ = opcodes.DROPLEVEL

    level = AnyField("level")
    axis = Int32Field("axis", default=0)

    def __init__(self, output_types=None, **kw):
        super().__init__(_output_types=output_types, **kw)

    def __call__(self, df_obj):
        if isinstance(df_obj, INDEX_TYPE):
            # For Index objects
            empty_index = df_obj.index_value.to_pandas()
            result_index = empty_index.droplevel(self.level)
            return self.new_index(
                [df_obj],
                shape=(df_obj.shape[0],),
                dtype=result_index.dtype,
                index_value=parse_index(result_index, store_data=False),
                name=result_index.name,
            )
        elif df_obj.ndim == 1:
            # For Series objects
            empty_series = build_series(df_obj)
            result_index = empty_series.index.droplevel(self.level)

            return self.new_series(
                [df_obj],
                shape=df_obj.shape,
                dtype=df_obj.dtype,
                index_value=parse_index(result_index, store_data=False),
                name=df_obj.name,
            )
        else:
            # For DataFrame objects
            result_dtypes = df_obj.dtypes
            result_shape = (df_obj.shape[0], df_obj.shape[1])

            empty_df = build_df(df_obj)
            if self.axis == 0:
                # Dropping levels from index
                result_index = empty_df.index.droplevel(self.level)
                result_index_value = parse_index(result_index, store_data=False)
                result_columns_value = df_obj.columns_value
            else:
                # Dropping levels from columns
                result_columns = empty_df.columns.droplevel(self.level)
                result_columns_value = parse_index(result_columns, store_data=True)
                result_index_value = df_obj.index_value

            return self.new_dataframe(
                [df_obj],
                shape=result_shape,
                dtypes=result_dtypes,
                index_value=result_index_value,
                columns_value=result_columns_value,
            )


def _droplevel(df_obj, level, axis=0):
    axis = validate_axis(axis, df_obj)
    op = DataFrameDropLevel(level=level, axis=axis)
    return op(df_obj)


def df_series_droplevel(df_or_series, level, axis=0):
    """
    Return Series/DataFrame with requested index / column level(s) removed.

    Parameters
    ----------
    level : int, str, or list-like
        If a string is given, must be the name of a level
        If list-like, elements must be names or positional indexes
        of levels.

    axis : {0 or 'index', 1 or 'columns'}, default 0
        Axis along which the level(s) is removed:

        * 0 or 'index': remove level(s) in column.
        * 1 or 'columns': remove level(s) in row.

        For `Series` this parameter is unused and defaults to 0.

    Returns
    -------
    Series/DataFrame
        Series/DataFrame with requested index / column level(s) removed.

    Examples
    --------
    >>> import maxframe.dataframe as md
    >>> df = md.DataFrame([
    ...     [1, 2, 3, 4],
    ...     [5, 6, 7, 8],
    ...     [9, 10, 11, 12]
    ... ]).set_index([0, 1]).rename_axis(['a', 'b'])

    >>> df.columns = md.MultiIndex.from_tuples([
    ...     ('c', 'e'), ('d', 'f')
    ... ], names=['level_1', 'level_2'])

    >>> df.execute()
    level_1   c   d
    level_2   e   f
    a b
    1 2      3   4
    5 6      7   8
    9 10    11  12

    >>> df.droplevel('a').execute()
    level_1   c   d
    level_2   e   f
    b
    2        3   4
    6        7   8
    10      11  12

    >>> df.droplevel('level_2', axis=1).execute()
    level_1   c   d
    a b
    1 2      3   4
    5 6      7   8
    9 10    11  12
    """
    return _droplevel(df_or_series, level, axis)


def index_droplevel(idx, level):
    """
    Return index with requested level(s) removed.

    If resulting index has only 1 level left, the result will be
    of Index type, not MultiIndex. The original index is not modified inplace.

    Parameters
    ----------
    level : int, str, or list-like, default 0
        If a string is given, must be the name of a level
        If list-like, elements must be names or indexes of levels.

    Returns
    -------
    Index or MultiIndex

    Examples
    --------
    >>> import maxframe.dataframe as md
    >>> mi = md.MultiIndex.from_arrays(
    ... [[1, 2], [3, 4], [5, 6]], names=['x', 'y', 'z'])
    >>> mi.execute()
    MultiIndex([(1, 3, 5),
                (2, 4, 6)],
                names=['x', 'y', 'z'])

    >>> mi.droplevel().execute()
    MultiIndex([(3, 5),
                (4, 6)],
                names=['y', 'z'])

    >>> mi.droplevel(2).execute()
    MultiIndex([(1, 3),
                (2, 4)],
                names=['x', 'y'])

    >>> mi.droplevel('z').execute()
    MultiIndex([(1, 3),
                (2, 4)],
                names=['x', 'y'])

    >>> mi.droplevel(['x', 'y']).execute()
    Index([5, 6], dtype='int64', name='z')
    """
    return _droplevel(idx, level)
