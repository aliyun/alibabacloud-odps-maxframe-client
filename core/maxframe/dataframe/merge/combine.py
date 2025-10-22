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
from ...serialization.serializables import AnyField, BoolField, FunctionField
from ...udf import BuiltinFunction
from ..operators import DataFrameOperator, DataFrameOperatorMixin


class DataFrameCombine(DataFrameOperator, DataFrameOperatorMixin):
    _op_type_ = opcodes.DATAFRAME_COMBINE

    func = FunctionField("func")
    fill_value = AnyField("fill_value")
    overwrite = BoolField("overwrite")

    def has_custom_code(self) -> bool:
        return not isinstance(self.func, BuiltinFunction)

    def __call__(self, obj1, obj2):
        from ..indexing.align import align

        assert obj1.ndim == 1 and obj2.ndim == 1
        obj1, obj2 = align(obj1, obj2)
        # Create the output series based on the result series
        return self.new_series(
            [obj1, obj2],
            shape=obj1.shape,
            dtype=obj1.dtype,
            index_value=obj1.index_value,
            name=obj1.name,
        )


def df_combine(df, other, func, fill_value=None, overwrite=True):
    """
    Perform column-wise combine with another DataFrame.

    Combines a DataFrame with `other` DataFrame using `func`
    to element-wise combine columns. The row and column indexes of the
    resulting DataFrame will be the union of the two.

    Parameters
    ----------
    other : DataFrame
        The DataFrame to merge column-wise.
    func : function
        Function that takes two series as inputs and return a Series or a
        scalar. Used to merge the two dataframes column by columns.
    fill_value : scalar value, default None
        The value to fill NaNs with prior to passing any column to the
        merge func.
    overwrite : bool, default True
        If True, columns in `self` that do not exist in `other` will be
        overwritten with NaNs.

    Returns
    -------
    DataFrame
        Combination of the provided DataFrames.

    See Also
    --------
    DataFrame.combine_first : Combine two DataFrame objects and default to
        non-null values in frame calling the method.

    Examples
    --------
    Combine using a simple function that chooses the smaller column.

    >>> import maxframe.tensor as mt
    >>> import maxframe.dataframe as md
    >>> df1 = md.DataFrame({'A': [0, 0], 'B': [4, 4]})
    >>> df2 = md.DataFrame({'A': [1, 1], 'B': [3, 3]})
    >>> take_smaller = lambda s1, s2: s1 if s1.sum() < s2.sum() else s2
    >>> df1.combine(df2, take_smaller).execute()
       A  B
    0  0  3
    1  0  3

    Example using a true element-wise combine function.

    >>> df1 = md.DataFrame({'A': [5, 0], 'B': [2, 4]})
    >>> df2 = md.DataFrame({'A': [1, 1], 'B': [3, 3]})
    >>> df1.combine(df2, mt.minimum).execute()
       A  B
    0  1  2
    1  0  3

    Using `fill_value` fills Nones prior to passing the column to the
    merge function.

    >>> df1 = md.DataFrame({'A': [0, 0], 'B': [None, 4]})
    >>> df2 = md.DataFrame({'A': [1, 1], 'B': [3, 3]})
    >>> df1.combine(df2, take_smaller, fill_value=-5).execute()
       A    B
    0  0 -5.0
    1  0  4.0

    However, if the same element in both dataframes is None, that None
    is preserved

    >>> df1 = md.DataFrame({'A': [0, 0], 'B': [None, 4]})
    >>> df2 = md.DataFrame({'A': [1, 1], 'B': [None, 3]})
    >>> df1.combine(df2, take_smaller, fill_value=-5).execute()
        A    B
    0  0 -5.0
    1  0  3.0

    Example that demonstrates the use of `overwrite` and behavior when
    the axis differ between the dataframes.

    >>> df1 = md.DataFrame({'A': [0, 0], 'B': [4, 4]})
    >>> df2 = md.DataFrame({'B': [3, 3], 'C': [-10, 1], }, index=[1, 2])
    >>> df1.combine(df2, take_smaller).execute()
         A    B     C
    0  NaN  NaN   NaN
    1  NaN  3.0 -10.0
    2  NaN  3.0   1.0

    >>> df1.combine(df2, take_smaller, overwrite=False).execute()
         A    B     C
    0  0.0  NaN   NaN
    1  0.0  3.0 -10.0
    2  NaN  3.0   1.0

    Demonstrating the preference of the passed in dataframe.

    >>> df2 = md.DataFrame({'B': [3, 3], 'C': [1, 1], }, index=[1, 2])
    >>> df2.combine(df1, take_smaller).execute()
       A    B   C
    0  0.0  NaN NaN
    1  0.0  3.0 NaN
    2  NaN  3.0 NaN

    >>> df2.combine(df1, take_smaller, overwrite=False).execute()
         A    B   C
    0  0.0  NaN NaN
    1  0.0  3.0 1.0
    2  NaN  3.0 1.0
    """
    # todo merge series logic into whole dataframe to reduce latency
    from ..indexing.align import align
    from .concat import concat

    src_df_cols = set(df.dtypes.index)
    src_other_cols = set(other.dtypes.index)

    df, other = align(df, other)
    col_data = []
    for c in df.dtypes.index:
        if c in src_df_cols and c in src_other_cols:
            col_data.append(func(df[c], other[c]))
        elif c in src_other_cols and not overwrite:
            col_data.append(df[c])
        else:
            col_data.append(other[c])
    res = concat(col_data, axis=1)
    if fill_value is not None:
        res = res.fillna(fill_value)
    return res


def series_combine(series, other, func, fill_value=None):
    """
    Combine the Series with a Series or scalar according to `func`.

    Combine the Series and `other` using `func` to perform elementwise
    selection for combined Series.
    `fill_value` is assumed when value is missing at some index
    from one of the two objects being combined.

    Parameters
    ----------
    other : Series or scalar
        The value(s) to be combined with the `Series`.
    func : function
        Function that takes two scalars as inputs and returns an element.
    fill_value : scalar, optional
        The value to assume when an index is missing from
        one Series or the other. The default specifies to use the
        appropriate NaN value for the underlying dtype of the Series.

    Returns
    -------
    Series
        The result of combining the Series with the other object.

    See Also
    --------
    Series.combine_first : Combine Series values, choosing the calling
        Series' values first.

    Examples
    --------
    Consider 2 Datasets ``s1`` and ``s2`` containing
    highest clocked speeds of different birds.

    >>> import maxframe.dataframe as md
    >>> s1 = md.Series({'falcon': 330.0, 'eagle': 160.0})
    >>> s1.execute()
    falcon    330.0
    eagle     160.0
    dtype: float64
    >>> s2 = md.Series({'falcon': 345.0, 'eagle': 200.0, 'duck': 30.0})
    >>> s2.execute()
    falcon    345.0
    eagle     200.0
    duck       30.0
    dtype: float64

    Now, to combine the two datasets and view the highest speeds
    of the birds across the two datasets

    >>> s1.combine(s2, max).execute()
    duck        NaN
    eagle     200.0
    falcon    345.0
    dtype: float64

    In the previous example, the resulting value for duck is missing,
    because the maximum of a NaN and a float is a NaN.
    So, in the example, we set ``fill_value=0``,
    so the maximum value returned will be the value from some dataset.

    >>> s1.combine(s2, max, fill_value=0).execute()
    duck       30.0
    eagle     200.0
    falcon    345.0
    dtype: float64
    """
    op = DataFrameCombine(func=func, fill_value=fill_value, overwrite=True)
    return op(series, other)
