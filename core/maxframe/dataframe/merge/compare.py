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

from typing import Tuple, Union

import numpy as np
import pandas as pd

from maxframe.utils import pd_release_version

from ... import opcodes
from ...core import OutputType
from ...serialization.serializables import AnyField, BoolField, TupleField
from ..operators import DataFrameOperator, DataFrameOperatorMixin
from ..utils import build_df, build_series, parse_index, validate_axis

_compare_has_result_names = pd_release_version >= (1, 5, 0)


def _compare_with_result_names(left, right, *args, **kwargs):
    if not _compare_has_result_names and kwargs.get("result_names"):
        result_names = kwargs.pop("result_names")
        res = left.compare(right, *args, **kwargs)
        axis = kwargs.get("align_axis", 1)
        idx_frame = res.axes[axis].to_frame(index=False)
        if len(idx_frame) > 0:
            idx_frame.iloc[-1] = idx_frame.iloc[-1].map(
                dict(zip(["self", "other"], result_names))
            )
        res.axes[axis] = pd.MultiIndex.from_frame(idx_frame, names=res.axes[axis].names)
        return res
    else:
        return left.compare(right, *args, **kwargs)


class DataFrameCompare(DataFrameOperator, DataFrameOperatorMixin):
    _op_type_ = opcodes.DATAFRAME_COMPARE

    align_axis = AnyField("align_axis", default=None)
    keep_shape = BoolField("keep_shape", default=None)
    keep_equal = BoolField("keep_equal", default=None)
    result_names = TupleField("result_names", default=None)

    def __init__(self, output_types=None, **kwargs):
        super().__init__(_output_types=output_types, **kwargs)

    def __call__(self, df_or_series, other):
        index_tokenize_objects = [
            df_or_series,
            other,
            self.align_axis,
            self.keep_shape,
            self.keep_equal,
            self.result_names,
        ]

        # Build empty objects for validation and output inference
        if df_or_series.ndim == 1:
            empty_left = build_series(df_or_series)
            empty_right = build_series(other)
        else:
            empty_left = build_df(df_or_series)
            empty_right = build_df(other)

        # Validate arguments by calling pandas compare
        compared = _compare_with_result_names(
            empty_left,
            empty_right,
            align_axis=self.align_axis,
            keep_shape=True,  # keep dims
            keep_equal=self.keep_equal,
            result_names=self.result_names,
        )

        self._output_types = [
            OutputType.dataframe if compared.ndim == 2 else OutputType.series
        ]

        index_value = columns_value = dtypes = None
        shape = [np.nan, np.nan]
        if self.keep_shape or (df_or_series.ndim == 1 and self.align_axis == 1):
            index_value = parse_index(compared.index, *index_tokenize_objects)
            columns_value = parse_index(compared.columns, store_data=True)
            dtypes = compared.dtypes
            shape[1] = len(dtypes)
        elif compared.ndim == 1:
            index_value = parse_index(compared.index, store_data=True)
            shape = (np.nan,)

        if compared.ndim == 2:
            return self.new_dataframe(
                [df_or_series, other],
                shape=tuple(shape),
                dtypes=dtypes,
                index_value=index_value,
                columns_value=columns_value,
            )
        else:
            return self.new_series(
                [df_or_series, other],
                shape=tuple(shape),
                dtype=compared.dtype,
                index_value=index_value,
                name=compared.name,
            )


def _compare(
    df_or_series,
    other,
    align_axis: Union[int, str] = 1,
    keep_shape: bool = False,
    keep_equal: bool = False,
    result_names: Tuple[str, str] = None,
):
    align_axis = validate_axis(align_axis)
    op = DataFrameCompare(
        align_axis=align_axis,
        keep_shape=keep_shape,
        keep_equal=keep_equal,
        result_names=result_names,
    )
    return op(df_or_series, other)


def df_compare(
    df,
    other,
    align_axis: Union[int, str] = 1,
    keep_shape: bool = False,
    keep_equal: bool = False,
    result_names: Tuple[str, str] = ("self", "other"),
):
    """
    Compare to another DataFrame and show the differences.

    Parameters
    ----------
    other : DataFrame
        Object to compare with.

    align_axis : {0 or 'index', 1 or 'columns'}, default 1
        Determine which axis to align the comparison on.

        * 0, or 'index' : Resulting differences are stacked vertically
            with rows drawn alternately from self and other.
        * 1, or 'columns' : Resulting differences are aligned horizontally
            with columns drawn alternately from self and other.

    keep_shape : bool, default False
        If true, all rows and columns are kept.
        Otherwise, only the ones with different values are kept.

    keep_equal : bool, default False
        If true, the result keeps values that are equal.
        Otherwise, equal values are shown as NaNs.

    result_names : tuple, default (‘self’, ‘other’)
        Set the dataframes names in the comparison.

    Returns
    -------
    DataFrame
        DataFrame that shows the differences stacked side by side.

        The resulting index will be a MultiIndex with 'self' and 'other'
        stacked alternately at the inner level.

    Raises
    ------
    ValueError
        When the two DataFrames don't have identical labels or shape.

    See Also
    --------
    Series.compare : Compare with another Series and show differences.
    DataFrame.equals : Test whether two objects contain the same elements.

    Notes
    -----
    Matching NaNs will not appear as a difference.

    Can only compare identically-labeled
    (i.e. same shape, identical row and column labels) DataFrames

    Examples
    --------
    >>> import maxframe.tensor as mt
    >>> import maxframe.dataframe as md
    >>> df = md.DataFrame(
    ...     {
    ...         "col1": ["a", "a", "b", "b", "a"],
    ...         "col2": [1.0, 2.0, 3.0, mt.nan, 5.0],
    ...         "col3": [1.0, 2.0, 3.0, 4.0, 5.0]
    ...     },
    ...     columns=["col1", "col2", "col3"],
    ... )
    >>> df.execute()
      col1  col2  col3
    0    a   1.0   1.0
    1    a   2.0   2.0
    2    b   3.0   3.0
    3    b   NaN   4.0
    4    a   5.0   5.0

    >>> df2 = df.copy()
    >>> df2.loc[0, 'col1'] = 'c'
    >>> df2.loc[2, 'col3'] = 4.0
    >>> df2.execute()
      col1  col2  col3
    0    c   1.0   1.0
    1    a   2.0   2.0
    2    b   3.0   4.0
    3    b   NaN   4.0
    4    a   5.0   5.0

    Align the differences on columns

    >>> df.compare(df2).execute()
      col1       col3
      self other self other
    0    a     c  NaN   NaN
    2  NaN   NaN  3.0   4.0

    Stack the differences on rows

    >>> df.compare(df2, align_axis=0).execute()
            col1  col3
    0 self     a   NaN
      other    c   NaN
    2 self   NaN   3.0
      other  NaN   4.0

    Keep the equal values

    >>> df.compare(df2, keep_equal=True).execute()
      col1       col3
      self other self other
    0    a     c  1.0   1.0
    2    b     b  3.0   4.0

    Keep all original rows and columns

    >>> df.compare(df2, keep_shape=True).execute()
      col1       col2       col3
      self other self other self other
    0    a     c  NaN   NaN  NaN   NaN
    1  NaN   NaN  NaN   NaN  NaN   NaN
    2  NaN   NaN  NaN   NaN  3.0   4.0
    3  NaN   NaN  NaN   NaN  NaN   NaN
    4  NaN   NaN  NaN   NaN  NaN   NaN

    Keep all original rows and columns and also all original values

    >>> df.compare(df2, keep_shape=True, keep_equal=True).execute()
      col1       col2       col3
      self other self other self other
    0    a     c  1.0   1.0  1.0   1.0
    1    a     a  2.0   2.0  2.0   2.0
    2    b     b  3.0   3.0  3.0   4.0
    3    b     b  NaN   NaN  4.0   4.0
    4    a     a  5.0   5.0  5.0   5.0
    """
    return _compare(
        df,
        other,
        align_axis=align_axis,
        keep_shape=keep_shape,
        keep_equal=keep_equal,
        result_names=result_names,
    )


def series_compare(
    series,
    other,
    align_axis: Union[int, str] = 1,
    keep_shape: bool = False,
    keep_equal: bool = False,
    result_names: Tuple[str, str] = ("self", "other"),
):
    """
    Compare to another Series and show the differences.

    Parameters
    ----------
    other : Series
        Object to compare with.

    align_axis : {0 or 'index', 1 or 'columns'}, default 1
        Determine which axis to align the comparison on.

        * 0, or 'index' : Resulting differences are stacked vertically
            with rows drawn alternately from self and other.
        * 1, or 'columns' : Resulting differences are aligned horizontally
            with columns drawn alternately from self and other.

    keep_shape : bool, default False
        If true, all rows and columns are kept.
        Otherwise, only the ones with different values are kept.

    keep_equal : bool, default False
        If true, the result keeps values that are equal.
        Otherwise, equal values are shown as NaNs.

    result_names : tuple, default (‘self’, ‘other’)
        Set the dataframes names in the comparison.

    Returns
    -------
    Series or DataFrame
        If axis is 0 or 'index' the result will be a Series.
        The resulting index will be a MultiIndex with 'self' and 'other'
        stacked alternately at the inner level.

        If axis is 1 or 'columns' the result will be a DataFrame.
        It will have two columns namely 'self' and 'other'.

    See Also
    --------
    DataFrame.compare : Compare with another DataFrame and show differences.

    Notes
    -----
    Matching NaNs will not appear as a difference.

    Examples
    --------
    >>> import maxframe.dataframe as md
    >>> s1 = md.Series(["a", "b", "c", "d", "e"])
    >>> s2 = md.Series(["a", "a", "c", "b", "e"])

    Align the differences on columns

    >>> s1.compare(s2).execute()
      self other
    1    b     a
    3    d     b

    Stack the differences on indices

    >>> s1.compare(s2, align_axis=0).execute()
    1  self     b
       other    a
    3  self     d
       other    b
    dtype: object

    Keep all original rows

    >>> s1.compare(s2, keep_shape=True).execute()
      self other
    0  NaN   NaN
    1    b     a
    2  NaN   NaN
    3    d     b
    4  NaN   NaN

    Keep all original rows and also all original values

    >>> s1.compare(s2, keep_shape=True, keep_equal=True).execute()
      self other
    0    a     a
    1    b     a
    2    c     c
    3    d     b
    4    e     e
    """
    return _compare(
        series,
        other,
        align_axis=align_axis,
        keep_shape=keep_shape,
        keep_equal=keep_equal,
        result_names=result_names,
    )
