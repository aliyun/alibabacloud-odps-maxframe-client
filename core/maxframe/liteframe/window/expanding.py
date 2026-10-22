# Copyright 1999-2026 Alibaba Group Holding Ltd.
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

from maxframe.liteframe.operators.window import (
    EXPANDING_AGG_FUNCS,
    LiteFrameWindowTransform,
)


class LiteFrameExpanding:
    """Provide expanding window calculations.

    Returns an intermediate object whose aggregation methods (``.sum()``,
    ``.mean()``, ...) compute a growing cumulative window from the start
    of the partition to the current row.  Unlike ``cum*``, expanding fills
    NaN positions with the accumulated value rather than masking them to null.

    Parameters
    ----------
    liteframe : LiteFrame
        The LiteFrame to compute on.
    min_periods : int, default 1
        Minimum number of observations in the window required to produce a
        value (otherwise null).
    columns : str, list of str, or None, default None
        Column(s) to aggregate. If None, all columns are aggregated.
        Non-selected columns are passed through unchanged.
    groupby_params : dict or None, default None
        Set when created from a ``groupby.expanding()`` call so the tiler
        partitions data by group key before computing per-group windows.
    """

    def __init__(
        self,
        liteframe,
        min_periods=1,
        columns=None,
        groupby_params=None,
        order_by=None,
        order_by_ascending=True,
        window_shift=0,
        reverse=False,
    ):
        # bool is a subclass of int — reject it explicitly, and require an
        # integral value (same rule as rolling).
        if isinstance(min_periods, bool) or not isinstance(min_periods, int):
            raise TypeError(
                f"min_periods must be an int, " f"got {type(min_periods).__name__}"
            )
        if min_periods < 1:
            raise ValueError(f"min_periods must be >= 1, got {min_periods}")
        if isinstance(columns, str):
            columns = [columns]
        if isinstance(window_shift, bool) or not isinstance(window_shift, int):
            raise TypeError(
                f"window_shift must be an int, " f"got {type(window_shift).__name__}"
            )

        self._liteframe = liteframe
        self._min_periods = min_periods
        self._columns = columns
        self._groupby_params = groupby_params
        self._order_by = order_by
        self._order_by_ascending = order_by_ascending
        self._window_shift = window_shift
        self._reverse = reverse

    def _agg(self, func):
        """Build and apply the expanding_agg window transform operator."""
        if func not in EXPANDING_AGG_FUNCS:
            raise ValueError(
                f"Unsupported expanding aggregation function: {func!r}. "
                f"Supported: {sorted(EXPANDING_AGG_FUNCS)}"
            )
        op = LiteFrameWindowTransform.for_expanding(
            func,
            min_periods=self._min_periods,
            columns=self._columns,
            groupby_params=self._groupby_params,
            order_by=self._order_by,
            order_by_ascending=self._order_by_ascending,
            window_shift=self._window_shift,
            reverse=self._reverse,
        )
        return op(self._liteframe)

    def sum(self):
        """Expanding sum.

        Returns
        -------
        LiteFrame
            Lazy LiteFrame with the cumulative (expanding) sum from the
            start to the current row.  Call ``.execute()`` to materialize.
        """
        return self._agg("sum")

    def mean(self):
        """Expanding mean.

        Returns
        -------
        LiteFrame
            Lazy LiteFrame with the cumulative (expanding) mean from the
            start to the current row.  Call ``.execute()`` to materialize.
        """
        return self._agg("mean")

    def min(self):
        """Expanding minimum.

        Returns
        -------
        LiteFrame
            Lazy LiteFrame with the cumulative (expanding) minimum from the
            start to the current row.  Call ``.execute()`` to materialize.
        """
        return self._agg("min")

    def max(self):
        """Expanding maximum.

        Returns
        -------
        LiteFrame
            Lazy LiteFrame with the cumulative (expanding) maximum from the
            start to the current row.  Call ``.execute()`` to materialize.
        """
        return self._agg("max")

    def count(self):
        """Expanding count of non-null observations.

        Returns
        -------
        LiteFrame
            Lazy LiteFrame with the cumulative (expanding) count of
            non-null values from the start to the current row.  Call
            ``.execute()`` to materialize.
        """
        return self._agg("count")

    def std(self):
        """Expanding standard deviation (ddof=1).

        Returns
        -------
        LiteFrame
            Lazy LiteFrame with the cumulative (expanding) standard
            deviation from the start to the current row (ddof=1).  Call
            ``.execute()`` to materialize.
        """
        return self._agg("std")

    def var(self):
        """Expanding variance (ddof=1).

        Returns
        -------
        LiteFrame
            Lazy LiteFrame with the cumulative (expanding) variance from
            the start to the current row (ddof=1).  Call ``.execute()`` to
            materialize.
        """
        return self._agg("var")

    def agg(self, func):
        """Aggregate using one of the supported function names.

        Parameters
        ----------
        func : str
            One of ``sum``, ``mean``, ``min``, ``max``, ``count``,
            ``std``, ``var``.

        Returns
        -------
        LiteFrame
            Lazy LiteFrame with the requested expanding aggregation applied.
            Call ``.execute()`` to materialize.
        """
        return self._agg(func)


def expanding(
    self,
    min_periods=1,
    columns=None,
    order_by=None,
    order_by_ascending=True,
    window_shift=0,
    reverse=False,
):
    """Provide expanding window calculations.

    Returns a :class:`LiteFrameExpanding` object whose aggregation methods
    (``.sum()``, ``.mean()``, ...) compute a growing cumulative window from
    the start to the current row. Unlike ``cum*``, expanding fills NaN
    positions with the accumulated value.

    Parameters
    ----------
    min_periods : int, default 1
        Minimum number of observations in the window required to produce a
        value (otherwise null).
    columns : str, list of str, or None, default None
        Column(s) to aggregate. If None, all columns are aggregated.
        Non-selected columns are passed through unchanged.
    order_by : str, list of str, or None, default None
        Column(s) that define the order in which the window expands
        (equivalent to ``ORDER BY`` inside a MaxCompute ``OVER()`` clause).
        The output row order is unchanged; each row receives the aggregation
        computed over the rows that precede it in *order_by* order.
    order_by_ascending : bool or list of bool, default True
        Sort direction for each *order_by* column. A single bool is broadcast
        to all columns.
    window_shift : int, default 0
        Offset the leading edge of the growing window by N rows. A positive
        value extends the window forward (the window becomes ``[0, i+shift]``),
        a negative value shrinks it. Matches the MaxCompute ``shift``
        MF-specific window parameter.
    reverse : bool, default False
        Flip the expanding window so it grows from the last row backward
        (a "narrowing" window covering ``[i, n-1]``). Matches the MaxCompute
        ``reverse_range`` MF-specific window parameter. May be combined with
        ``window_shift``: the window then covers ``[i+window_shift, n-1]``.

    Returns
    -------
    LiteFrameExpanding
        Intermediate object; call an aggregation method to produce a LiteFrame.

    See Also
    --------
    rolling : Provide rolling-window calculations.
    shift : Shift rows by a given number of periods.
    """
    return LiteFrameExpanding(
        self,
        min_periods=min_periods,
        columns=columns,
        order_by=order_by,
        order_by_ascending=order_by_ascending,
        window_shift=window_shift,
        reverse=reverse,
    )
