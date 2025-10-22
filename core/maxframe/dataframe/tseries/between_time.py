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

from ... import opcodes
from ...core import get_output_types
from ...serialization.serializables import AnyField, Int32Field, StringField
from ..operators import DataFrameOperator, DataFrameOperatorMixin
from ..utils import parse_index, validate_axis


class DataFrameBetweenTime(DataFrameOperator, DataFrameOperatorMixin):
    _op_type_ = opcodes.BETWEEN_TIME

    start_time = AnyField("start_time")
    end_time = AnyField("end_time")
    inclusive = StringField("inclusive")
    axis = Int32Field("axis")

    def __call__(self, df_or_series):
        self._output_types = get_output_types(df_or_series)
        out_params = df_or_series.params

        new_shape = list(df_or_series.shape)
        new_shape[self.axis] = np.nan
        out_params["shape"] = tuple(new_shape)

        idx_key_params = (df_or_series, self.start_time, self.end_time, self.inclusive)
        if self.axis == 0:
            out_params["index_value"] = parse_index(
                df_or_series.index_value.to_pandas()[:0], idx_key_params
            )
        else:
            out_params["columns_value"] = parse_index(
                df_or_series.columns_value.to_pandas()[:0], idx_key_params
            )

        return self.new_tileable([df_or_series], **out_params)


def between_time(df_or_series, start_time, end_time, inclusive="both", axis=0):
    """
    Select values between particular times of the day (e.g., 9:00-9:30 AM).

    By setting ``start_time`` to be later than ``end_time``,
    you can get the times that are *not* between the two times.

    Parameters
    ----------
    start_time : datetime.time or str
        Initial time as a time filter limit.
    end_time : datetime.time or str
        End time as a time filter limit.
    inclusive : {"both", "neither", "left", "right"}, default "both"
        Include boundaries; whether to set each bound as closed or open.
    axis : {0 or 'index', 1 or 'columns'}, default 0
        Determine range time on index or columns value.
        For `Series` this parameter is unused and defaults to 0.

    Returns
    -------
    Series or DataFrame
        Data from the original object filtered to the specified dates range.

    Raises
    ------
    TypeError
        If the index is not  a :class:`DatetimeIndex`

    See Also
    --------
    at_time : Select values at a particular time of the day.
    first : Select initial periods of time series based on a date offset.
    last : Select final periods of time series based on a date offset.
    DatetimeIndex.indexer_between_time : Get just the index locations for
        values between particular times of the day.

    Examples
    --------
    >>> import maxframe.dataframe as md
    >>> i = md.date_range('2018-04-09', periods=4, freq='1D20min')
    >>> ts = md.DataFrame({'A': [1, 2, 3, 4]}, index=i)
    >>> ts.execute()
                         A
    2018-04-09 00:00:00  1
    2018-04-10 00:20:00  2
    2018-04-11 00:40:00  3
    2018-04-12 01:00:00  4

    >>> ts.between_time('0:15', '0:45').execute()
                         A
    2018-04-10 00:20:00  2
    2018-04-11 00:40:00  3

    You get the times that are *not* between two times by setting
    ``start_time`` later than ``end_time``:

    >>> ts.between_time('0:45', '0:15').execute()
                         A
    2018-04-09 00:00:00  1
    2018-04-12 01:00:00  4
    """
    axis = validate_axis(axis, df_or_series)
    op = DataFrameBetweenTime(
        start_time=start_time,
        end_time=end_time,
        inclusive=inclusive,
        axis=axis,
    )
    return op(df_or_series)
