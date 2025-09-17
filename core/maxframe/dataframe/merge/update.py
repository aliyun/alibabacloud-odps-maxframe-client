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
from ...core import get_output_types
from ...serialization.serializables import BoolField, StringField
from ...udf import BuiltinFunction
from ..operators import DataFrameOperator, DataFrameOperatorMixin
from ..utils import parse_index


class DataFrameUpdate(DataFrameOperator, DataFrameOperatorMixin):
    _op_type_ = opcodes.DATAFRAME_UPDATE

    join = StringField("join", default=None)
    overwrite = BoolField("overwrite", default=None)
    filter_func = BoolField("filter_func", default=None)
    errors = StringField("errors", default=None)

    def __init__(self, output_types=None, **kwargs):
        super().__init__(_output_types=output_types, **kwargs)

    def has_custom_code(self) -> bool:
        return not isinstance(self.filter_func, BuiltinFunction)

    def __call__(self, df_or_series, other):
        self._output_types = get_output_types(df_or_series)

        index_tokenize_objects = [
            df_or_series,
            other,
            self.join,
            self.overwrite,
            self.filter_func,
            self.errors,
        ]

        # Return the original object (update is in-place)
        if df_or_series.ndim == 2:
            return self.new_dataframe(
                [df_or_series, other],
                shape=df_or_series.shape,
                dtypes=df_or_series.dtypes,
                index_value=parse_index(
                    df_or_series.index_value.to_pandas(), *index_tokenize_objects
                ),
                columns_value=df_or_series.columns_value,
            )
        else:
            return self.new_series(
                [df_or_series, other],
                shape=df_or_series.shape,
                dtype=df_or_series.dtype,
                index_value=parse_index(
                    df_or_series.index_value.to_pandas(), *index_tokenize_objects
                ),
                name=df_or_series.name,
            )


def _update(
    df_or_series, other, join="left", overwrite=True, filter_func=None, errors="ignore"
):
    op = DataFrameUpdate(
        join=join,
        overwrite=overwrite,
        filter_func=filter_func,
        errors=errors,
    )
    result = op(df_or_series, other)
    df_or_series.data = result.data


def df_update(
    df, other, join="left", overwrite=True, filter_func=None, errors="ignore"
):
    """
    Modify in place using non-NA values from another DataFrame.

    Aligns on indices. There is no return value.

    Parameters
    ----------
    other : DataFrame, or object coercible into a DataFrame
        Should have at least one matching index/column label
        with the original DataFrame. If a Series is passed,
        its name attribute must be set, and that will be
        used as the column name to align with the original DataFrame.
    join : {'left'}, default 'left'
        Only left join is implemented, keeping the index and columns of the
        original object.
    overwrite : bool, default True
        How to handle non-NA values for overlapping keys:

        * True: overwrite original DataFrame's values
          with values from `other`.
        * False: only update values that are NA in
          the original DataFrame.

    filter_func : callable(1d-array) -> bool 1d-array, optional
        Can choose to replace values other than NA. Return True for values
        that should be updated.
    errors : {'raise', 'ignore'}, default 'ignore'
        If 'raise', will raise a ValueError if the DataFrame and `other`
        both contain non-NA data in the same place.

    Returns
    -------
    None
        This method directly changes calling object.

    Raises
    ------
    ValueError
        * When `errors='raise'` and there's overlapping non-NA data.
        * When `errors` is not either `'ignore'` or `'raise'`
    NotImplementedError
        * If `join != 'left'`

    See Also
    --------
    dict.update : Similar method for dictionaries.
    DataFrame.merge : For column(s)-on-column(s) operations.

    Examples
    --------
    >>> import maxframe.tensor as mt
    >>> import maxframe.dataframe as md
    >>> df = md.DataFrame({'A': [1, 2, 3],
    ...                    'B': [400, 500, 600]})
    >>> new_df = md.DataFrame({'B': [4, 5, 6],
    ...                        'C': [7, 8, 9]})
    >>> df.update(new_df)
    >>> df.execute()
       A  B
    0  1  4
    1  2  5
    2  3  6

    The DataFrame's length does not increase as a result of the update,
    only values at matching index/column labels are updated.

    >>> df = md.DataFrame({'A': ['a', 'b', 'c'],
    ...                    'B': ['x', 'y', 'z']})
    >>> new_df = md.DataFrame({'B': ['d', 'e', 'f', 'g', 'h', 'i']})
    >>> df.update(new_df)
    >>> df.execute()
       A  B
    0  a  d
    1  b  e
    2  c  f

    >>> df = md.DataFrame({'A': ['a', 'b', 'c'],
    ...                    'B': ['x', 'y', 'z']})
    >>> new_df = md.DataFrame({'B': ['d', 'f']}, index=[0, 2])
    >>> df.update(new_df)
    >>> df.execute()
       A  B
    0  a  d
    1  b  y
    2  c  f

    For Series, its name attribute must be set.

    >>> df = md.DataFrame({'A': ['a', 'b', 'c'],
    ...                    'B': ['x', 'y', 'z']})
    >>> new_column = md.Series(['d', 'e', 'f'], name='B')
    >>> df.update(new_column)
    >>> df.execute()
       A  B
    0  a  d
    1  b  e
    2  c  f

    If `other` contains NaNs the corresponding values are not updated
    in the original dataframe.

    >>> df = md.DataFrame({'A': [1, 2, 3],
    ...                    'B': [400., 500., 600.]})
    >>> new_df = md.DataFrame({'B': [4, mt.nan, 6]})
    >>> df.update(new_df)
    >>> df.execute()
       A      B
    0  1    4.0
    1  2  500.0
    2  3    6.0
    """
    return _update(df, other, join, overwrite, filter_func, errors)


def series_update(series, other):
    """
    Modify Series in place using values from passed Series.

    Uses non-NA values from passed Series to make updates. Aligns
    on index.

    Parameters
    ----------
    other : Series, or object coercible into Series

    Examples
    --------
    >>> import maxframe.tensor as mt
    >>> import maxframe.dataframe as md
    >>> s = md.Series([1, 2, 3])
    >>> s.update(md.Series([4, 5, 6]))
    >>> s.execute()
    0    4
    1    5
    2    6
    dtype: int64

    >>> s = md.Series(['a', 'b', 'c'])
    >>> s.update(md.Series(['d', 'e'], index=[0, 2]))
    >>> s.execute()
    0    d
    1    b
    2    e
    dtype: object

    >>> s = md.Series([1, 2, 3])
    >>> s.update(md.Series([4, 5, 6, 7, 8]))
    >>> s.execute()
    0    4
    1    5
    2    6
    dtype: int64

    If ``other`` contains NaNs the corresponding values are not updated
    in the original Series.

    >>> s = md.Series([1, 2, 3])
    >>> s.update(md.Series([4, mt.nan, 6]))
    >>> s.execute()
    0    4
    1    2
    2    6
    dtype: int64

    ``other`` can also be a non-Series object type
    that is coercible into a Series

    >>> s = md.Series([1, 2, 3])
    >>> s.update([4, mt.nan, 6])
    >>> s.execute()
    0    4
    1    2
    2    6
    dtype: int64

    >>> s = md.Series([1, 2, 3])
    >>> s.update({1: 9})
    >>> s.execute()
    0    1
    1    9
    2    3
    dtype: int64
    """
    return _update(series, other)
