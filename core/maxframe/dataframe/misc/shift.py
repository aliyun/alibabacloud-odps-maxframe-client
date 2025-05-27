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

import pandas as pd

from ... import opcodes
from ...core import EntityData, OutputType
from ...serialization.serializables import AnyField, Int8Field, Int64Field, KeyField
from ...utils import no_default, pd_release_version
from ..operators import DataFrameOperator, DataFrameOperatorMixin
from ..utils import build_df, build_series, parse_index, validate_axis

_need_consolidate = pd.__version__ in ("1.1.0", "1.3.0", "1.3.1")
_enable_no_default = pd_release_version[:2] > (1, 1)
_with_column_freq_bug = (1, 2, 0) <= pd_release_version < (1, 4, 3)


class DataFrameShift(DataFrameOperator, DataFrameOperatorMixin):
    _op_type_ = opcodes.SHIFT

    _input = KeyField("input")
    periods = Int64Field("periods", default=None)
    freq = AnyField("freq", default=None)
    axis = Int8Field("axis", default=None)
    fill_value = AnyField("fill_value", default=None)

    @property
    def input(self):
        return self._input

    @classmethod
    def _set_inputs(cls, op: "DataFrameShift", inputs: List[EntityData]):
        super()._set_inputs(op, inputs)
        op._input = op._inputs[0]

    def _call_dataframe(self, df):
        test_df = build_df(df)
        result_df = test_df.shift(
            periods=self.periods,
            freq=self.freq,
            axis=self.axis,
            fill_value=self.fill_value,
        )

        if self.freq is None:
            # shift data
            index_value = df.index_value
            columns_value = df.columns_value
        else:
            # shift index
            if self.axis == 0:
                index_value = self._get_index_value(
                    df.index_value, self.periods, self.freq
                )
                columns_value = df.columns_value
            else:
                columns_value = parse_index(result_df.dtypes.index, store_data=True)
                index_value = df.index_value

        return self.new_dataframe(
            [df],
            shape=df.shape,
            dtypes=result_df.dtypes,
            index_value=index_value,
            columns_value=columns_value,
        )

    def _call_series(self, series):
        test_series = build_series(series)
        result_series = test_series.shift(
            periods=self.periods,
            freq=self.freq,
            axis=self.axis,
            fill_value=self.fill_value,
        )

        index_value = series.index_value
        if self.freq is not None:
            # shift index
            index_value = self._get_index_value(index_value, self.periods, self.freq)

        return self.new_series(
            [series],
            shape=series.shape,
            index_value=index_value,
            dtype=result_series.dtype,
            name=series.name,
        )

    def __call__(self, df_or_series):
        if df_or_series.op.output_types[0] == OutputType.dataframe:
            self.output_types = [OutputType.dataframe]
            return self._call_dataframe(df_or_series)
        else:
            assert df_or_series.op.output_types[0] == OutputType.series
            self.output_types = [OutputType.series]
            return self._call_series(df_or_series)

    @staticmethod
    def _get_index_value(input_index_value, periods, freq):
        if (
            not input_index_value.has_value()
            and input_index_value.min_val is not None
            and input_index_value.max_val is not None
            and freq is not None
            and input_index_value.is_monotonic_increasing_or_decreasing
        ):
            pd_index = pd.Index(
                [input_index_value.min_val, input_index_value.max_val]
            ).shift(periods=periods, freq=freq)
            index_value = parse_index(pd_index)
            index_value.value._min_val_close = input_index_value.min_val_close
            index_value.value._max_val_close = input_index_value.max_val_close
            return index_value
        else:
            pd_index = input_index_value.to_pandas()
            return parse_index(pd_index, periods, freq)


def shift(df_or_series, periods=1, freq=None, axis=0, fill_value=None):
    """
    Shift index by desired number of periods with an optional time `freq`.

    When `freq` is not passed, shift the index without realigning the data.
    If `freq` is passed (in this case, the index must be date or datetime,
    or it will raise a `NotImplementedError`), the index will be
    increased using the periods and the `freq`.

    Parameters
    ----------
    periods : int
        Number of periods to shift. Can be positive or negative.
    freq : DateOffset, tseries.offsets, timedelta, or str, optional
        Offset to use from the tseries module or time rule (e.g. 'EOM').
        If `freq` is specified then the index values are shifted but the
        data is not realigned. That is, use `freq` if you would like to
        extend the index when shifting and preserve the original data.
    axis : {0 or 'index', 1 or 'columns', None}, default None
        Shift direction.
    fill_value : object, optional
        The scalar value to use for newly introduced missing values.
        the default depends on the dtype of `self`.
        For numeric data, ``np.nan`` is used.
        For datetime, timedelta, or period data, etc. :attr:`NaT` is used.
        For extension dtypes, ``self.dtype.na_value`` is used.

    Returns
    -------
    DataFrame or Series
        Copy of input object, shifted.

    See Also
    --------
    Index.shift : Shift values of Index.
    DatetimeIndex.shift : Shift values of DatetimeIndex.
    PeriodIndex.shift : Shift values of PeriodIndex.
    tshift : Shift the time index, using the index's frequency if
        available.

    Examples
    --------
    >>> import maxframe.dataframe as md

    >>> df = md.DataFrame({'Col1': [10, 20, 15, 30, 45],
    ...                    'Col2': [13, 23, 18, 33, 48],
    ...                    'Col3': [17, 27, 22, 37, 52]})

    >>> df.shift(periods=3).execute()
       Col1  Col2  Col3
    0   NaN   NaN   NaN
    1   NaN   NaN   NaN
    2   NaN   NaN   NaN
    3  10.0  13.0  17.0
    4  20.0  23.0  27.0

    >>> df.shift(periods=1, axis='columns').execute()
       Col1  Col2  Col3
    0   NaN  10.0  13.0
    1   NaN  20.0  23.0
    2   NaN  15.0  18.0
    3   NaN  30.0  33.0
    4   NaN  45.0  48.0

    >>> df.shift(periods=3, fill_value=0).execute()
       Col1  Col2  Col3
    0     0     0     0
    1     0     0     0
    2     0     0     0
    3    10    13    17
    4    20    23    27
    """
    axis = validate_axis(axis, df_or_series)
    if periods == 0:
        return df_or_series.copy()
    if fill_value is no_default:  # pragma: no cover
        if not _enable_no_default or (
            _with_column_freq_bug and axis == 1 and freq is not None
        ):
            # pandas shift shows different behavior for axis=1 when freq is specified,
            # see https://github.com/pandas-dev/pandas/issues/47039 for details.
            fill_value = None
    op = DataFrameShift(periods=periods, freq=freq, axis=axis, fill_value=fill_value)
    return op(df_or_series)


def tshift(df_or_series, periods: int = 1, freq=None, axis=0):
    """
    Shift the time index, using the index's frequency if available.

    Parameters
    ----------
    periods : int
        Number of periods to move, can be positive or negative.
    freq : DateOffset, timedelta, or str, default None
        Increment to use from the tseries module
        or time rule expressed as a string (e.g. 'EOM').
    axis : {0 or ‘index’, 1 or ‘columns’, None}, default 0
        Corresponds to the axis that contains the Index.

    Returns
    -------
    shifted : Series/DataFrame

    Notes
    -----
    If freq is not specified then tries to use the freq or inferred_freq
    attributes of the index. If neither of those attributes exist, a
    ValueError is thrown
    """
    axis = validate_axis(axis, df_or_series)
    index = (
        df_or_series.index_value.to_pandas()
        if axis == 0
        else df_or_series.columns_value.to_pandas()
    )

    if freq is None:
        freq = getattr(index, "freq", None)

    if freq is None:  # pragma: no cover
        freq = getattr(index, "inferred_freq", None)

    if freq is None:
        raise ValueError("Freq was not given and was not set in the index")

    return shift(df_or_series, periods=periods, freq=freq, axis=axis)
