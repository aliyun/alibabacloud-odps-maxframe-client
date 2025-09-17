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
from ...serialization.serializables import AnyField, DictField
from ..window.rolling import Rolling
from .core import BaseGroupByWindowOp

_supported_funcs = {"sum", "mean", "std", "var", "median", "min", "max", "count"}


class GroupByRollingAgg(BaseGroupByWindowOp):
    _op_type_ = opcodes.ROLLING_AGG

    func = AnyField("func", default=None)
    kwargs = DictField("kwargs", default_factory=dict)

    def _calc_mock_result_df(self, mock_groupby):
        rolling_args = self.window_params.copy()
        # exclude MF-specific args
        for key in Rolling._mf_specific_fields:
            rolling_args.pop(key, None)

        def apply_func(frame, **_):
            return getattr(frame.rolling(**rolling_args), self.func)()

        return mock_groupby.apply(apply_func, include_groups=False)


class RollingGroupby(Rolling):
    def aggregate(self, func, **kwargs):
        if func not in _supported_funcs:
            raise NotImplementedError(f"func {func} is not supported")
        op = GroupByRollingAgg(
            func=func,
            groupby_params=self.input.op.groupby_params,
            window_params=self.params,
            kwargs=kwargs,
        )
        return op(self.input)


def rolling(
    groupby,
    window,
    min_periods=None,
    *,
    center=False,
    win_type=None,
    on=None,
    axis=0,
    closed=None,
    shift=0,
    order_cols=None,
    ascending=True,
) -> RollingGroupby:
    """
    Return a rolling grouper, providing rolling functionality per group.

    Parameters
    ----------
    window : int, timedelta, str, offset, or BaseIndexer subclass
        Size of the moving window.

        If an integer, the fixed number of observations used for
        each window.

        If a timedelta, str, or offset, the time period of each window. Each
        window will be a variable sized based on the observations included in
        the time-period. This is only valid for datetimelike indexes.
        To learn more about the offsets & frequency strings, please see `this link
        <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases>`__.

        If a BaseIndexer subclass, the window boundaries
        based on the defined ``get_window_bounds`` method. Additional rolling
        keyword arguments, namely ``min_periods``, ``center``, ``closed`` and
        ``step`` will be passed to ``get_window_bounds``.

    min_periods : int, default None
        Minimum number of observations in window required to have a value;
        otherwise, result is ``np.nan``.

        For a window that is specified by an offset,
        ``min_periods`` will default to 1.

        For a window that is specified by an integer, ``min_periods`` will default
        to the size of the window.

    center : bool, default False
        If False, set the window labels as the right edge of the window index.

        If True, set the window labels as the center of the window index.

    win_type : str, default None
        If ``None``, all points are evenly weighted.

        If a string, it must be a valid `scipy.signal window function
        <https://docs.scipy.org/doc/scipy/reference/signal.windows.html#module-scipy.signal.windows>`__.

        Certain Scipy window types require additional parameters to be passed
        in the aggregation function. The additional parameters must match
        the keywords specified in the Scipy window type method signature.

    on : str, optional
        For a DataFrame, a column label or Index level on which
        to calculate the rolling window, rather than the DataFrame's index.

        Provided integer column is ignored and excluded from result since
        an integer index is not used to calculate the rolling window.

    axis : int or str, default 0
        If ``0`` or ``'index'``, roll across the rows.

        If ``1`` or ``'columns'``, roll across the columns.

        For `Series` this parameter is unused and defaults to 0.

    closed : str, default None
        If ``'right'``, the first point in the window is excluded from calculations.

        If ``'left'``, the last point in the window is excluded from calculations.

        If ``'both'``, no points in the window are excluded from calculations.

        If ``'neither'``, the first and last points in the window are excluded
        from calculations.

        Default ``None`` (``'right'``).

    shift : int, default 0
        If specified, the window will be shifted by `shift` rows (or data will be
        shifted by `-shift` rows) before computing window function.

    Returns
    -------
    maxframe.dataframe.groupby.RollingGroupby
        Return a new grouper with our rolling appended.

    See Also
    --------
    Series.rolling : Calling object with Series data.
    DataFrame.rolling : Calling object with DataFrames.
    Series.groupby : Apply a function groupby to a Series.
    DataFrame.groupby : Apply a function groupby.

    Examples
    --------
    >>> import maxframe.dataframe as md
    >>> df = md.DataFrame({'A': [1, 1, 2, 2],
    ...                    'B': [1, 2, 3, 4],
    ...                    'C': [0.362, 0.227, 1.267, -0.562]})
    >>> df.execute()
          A  B      C
    0     1  1  0.362
    1     1  2  0.227
    2     2  3  1.267
    3     2  4 -0.562

    >>> df.groupby('A').rolling(2).sum().execute()
        B      C
    A
    1 0  NaN    NaN
      1  3.0  0.589
    2 2  NaN    NaN
      3  7.0  0.705

    >>> df.groupby('A').rolling(2, min_periods=1).sum().execute()
        B      C
    A
    1 0  1.0  0.362
      1  3.0  0.589
    2 2  3.0  1.267
      3  7.0  0.705

    >>> df.groupby('A').rolling(2, on='B').sum().execute()
        B      C
    A
    1 0  1    NaN
      1  2  0.589
    2 2  3    NaN
      3  4  0.705
    """
    return RollingGroupby(
        input=groupby,
        window=window,
        min_periods=min_periods,
        center=center,
        win_type=win_type,
        on=on,
        axis=axis,
        closed=closed,
        shift=shift,
        order_cols=order_cols,
        ascending=ascending,
    )
