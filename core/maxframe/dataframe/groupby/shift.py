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
from ...serialization.serializables import AnyField, Int64Field
from .core import BaseGroupByWindowOp


class GroupByShift(BaseGroupByWindowOp):
    _op_type_ = opcodes.SHIFT

    periods = Int64Field("periods", default=None)
    freq = AnyField("freq", default=None)
    fill_value = AnyField("fill_value", default=None)

    def _calc_mock_result_df(self, mock_groupby):
        return mock_groupby.shift(
            self.periods, freq=self.freq, fill_value=self.fill_value
        )


def shift(
    groupby, periods=1, freq=None, fill_value=None, order_cols=None, ascending=True
):
    """
    Shift each group by periods observations.

    If freq is passed, the index will be increased using the periods and the freq.

    Parameters
    ----------
    periods : int | Sequence[int], default 1
        Number of periods to shift. If a list of values, shift each group by
        each period.
    freq : str, optional
        Frequency string.

    fill_value : optional
        The scalar value to use for newly introduced missing values.

    Returns
    -------
    Series or DataFrame
        Object shifted within each group.

    See Also
    --------
    Index.shift : Shift values of Index.

    Examples
    --------

    For SeriesGroupBy:

    >>> import maxframe.dataframe as md
    >>> lst = ['a', 'a', 'b', 'b']
    >>> ser = md.Series([1, 2, 3, 4], index=lst)
    >>> ser.execute()
    a    1
    a    2
    b    3
    b    4
    dtype: int64
    >>> ser.groupby(level=0).shift(1).execute()
    a    NaN
    a    1.0
    b    NaN
    b    3.0
    dtype: float64

    For DataFrameGroupBy:

    >>> data = [[1, 2, 3], [1, 5, 6], [2, 5, 8], [2, 6, 9]]
    >>> df = md.DataFrame(data, columns=["a", "b", "c"],
    ...                   index=["tuna", "salmon", "catfish", "goldfish"])
    >>> df.execute()
               a  b  c
        tuna   1  2  3
      salmon   1  5  6
     catfish   2  5  8
    goldfish   2  6  9
    >>> df.groupby("a").shift(1).execute()
                  b    c
        tuna    NaN  NaN
      salmon    2.0  3.0
     catfish    NaN  NaN
    goldfish    5.0  8.0
    """
    if not isinstance(ascending, list):
        ascending = [ascending]

    window_params = dict(
        order_cols=order_cols,
        ascending=ascending,
    )
    op = GroupByShift(
        periods=periods,
        freq=freq,
        fill_value=fill_value,
        groupby_params=groupby.op.groupby_params,
        window_params=window_params,
    )
    return op(groupby)
