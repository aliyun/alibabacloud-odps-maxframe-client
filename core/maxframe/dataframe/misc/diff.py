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
from ...serialization.serializables import AnyField, Int8Field, Int64Field
from ..core import DATAFRAME_TYPE, OutputType
from ..operators import DataFrameOperator, DataFrameOperatorMixin
from ..utils import build_empty_df, build_empty_series, validate_axis


class DataFrameDiff(DataFrameOperatorMixin, DataFrameOperator):
    _op_type_ = opcodes.DIFF

    periods = Int64Field("periods", default=None)
    axis = Int8Field("axis", default=None)

    bool_columns = AnyField("bool_columns", default=None)

    def __call__(self, df_or_series):
        params = df_or_series.params.copy()

        if isinstance(df_or_series, DATAFRAME_TYPE):
            self.output_types = [OutputType.dataframe]
            mock_obj = build_empty_df(df_or_series.dtypes)
            params["dtypes"] = mock_obj.diff().dtypes
        else:
            self.output_types = [OutputType.series]
            mock_obj = build_empty_series(df_or_series.dtype, name=df_or_series.name)
            params["dtype"] = mock_obj.diff().dtype

        return self.new_tileable([df_or_series], **params)


def df_diff(df, periods=1, axis=0):
    """
    First discrete difference of element.
    Calculates the difference of a DataFrame element compared with another
    element in the DataFrame (default is the element in the same column
    of the previous row).

    Parameters
    ----------
    periods : int, default 1
        Periods to shift for calculating difference, accepts negative
        values.
    axis : {0 or 'index', 1 or 'columns'}, default 0
        Take difference over rows (0) or columns (1).

    Returns
    -------
    DataFrame

    See Also
    --------
    Series.diff : First discrete difference for a Series.
    DataFrame.pct_change : Percent change over given number of periods.
    DataFrame.shift : Shift index by desired number of periods with an
        optional time freq.

    Notes
    -----
    For boolean dtypes, this uses :meth:`operator.xor` rather than
    :meth:`operator.sub`.

    Examples
    --------
    Difference with previous row

    >>> import maxframe.dataframe as md
    >>> df = md.DataFrame({'a': [1, 2, 3, 4, 5, 6],
    ...                    'b': [1, 1, 2, 3, 5, 8],
    ...                    'c': [1, 4, 9, 16, 25, 36]})
    >>> df.execute()
       a  b   c
    0  1  1   1
    1  2  1   4
    2  3  2   9
    3  4  3  16
    4  5  5  25
    5  6  8  36

    >>> df.diff().execute()
         a    b     c
    0  NaN  NaN   NaN
    1  1.0  0.0   3.0
    2  1.0  1.0   5.0
    3  1.0  1.0   7.0
    4  1.0  2.0   9.0
    5  1.0  3.0  11.0

    Difference with previous column

    >>> df.diff(axis=1).execute()
        a    b     c
    0 NaN  0.0   0.0
    1 NaN -1.0   3.0
    2 NaN -1.0   7.0
    3 NaN -1.0  13.0
    4 NaN  0.0  20.0
    5 NaN  2.0  28.0

    Difference with 3rd previous row

    >>> df.diff(periods=3).execute()
         a    b     c
    0  NaN  NaN   NaN
    1  NaN  NaN   NaN
    2  NaN  NaN   NaN
    3  3.0  2.0  15.0
    4  3.0  4.0  21.0
    5  3.0  6.0  27.0

    Difference with following row

    >>> df.diff(periods=-1).execute()
         a    b     c
    0 -1.0  0.0  -3.0
    1 -1.0 -1.0  -5.0
    2 -1.0 -1.0  -7.0
    3 -1.0 -2.0  -9.0
    4 -1.0 -3.0 -11.0
    5  NaN  NaN   NaN
    """
    axis = validate_axis(axis, df)
    op = DataFrameDiff(periods=periods, axis=axis)
    return op(df)


def series_diff(series, periods=1):
    """
    First discrete difference of element.
    Calculates the difference of a Series element compared with another
    element in the Series (default is element in previous row).

    Parameters
    ----------
    periods : int, default 1
        Periods to shift for calculating difference, accepts negative
        values.

    Returns
    -------
    Series
        First differences of the Series.

    See Also
    --------
    Series.pct_change :
        Percent change over given number of periods.
    Series.shift :
        Shift index by desired number of periods with an optional time freq.
    DataFrame.diff :
        First discrete difference of object.

    Notes
    -----
    For boolean dtypes, this uses :meth:`operator.xor` rather than
    :meth:`operator.sub`.

    Examples
    --------

    Difference with previous row

    >>> import maxframe.dataframe as md
    >>> s = md.Series([1, 1, 2, 3, 5, 8])
    >>> s.diff().execute()
    0    NaN
    1    0.0
    2    1.0
    3    1.0
    4    2.0
    5    3.0
    dtype: float64

    Difference with 3rd previous row

    >>> s.diff(periods=3).execute()
    0    NaN
    1    NaN
    2    NaN
    3    2.0
    4    4.0
    5    6.0
    dtype: float64

    Difference with following row

    >>> s.diff(periods=-1).execute()
    0    0.0
    1   -1.0
    2   -1.0
    3   -2.0
    4   -3.0
    5    NaN
    dtype: float64
    """
    op = DataFrameDiff(periods=periods)
    return op(series)
