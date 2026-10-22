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
    LiteFrameWindowTransform,
    _normalize_columns,
    _normalize_order_by,
)

# Aggregations exposed as dedicated methods on LiteFrameRolling. Each maps to
# a polars rolling_* primitive at execution time (count is synthesized).
_ROLLING_METHODS = ("sum", "mean", "min", "max", "std", "var", "median", "count")


class LiteFrameRolling:
    """Intermediate object returned by ``LiteFrame.rolling()``.

    Provides rolling-window aggregations over a fixed-size integer window.
    Execution uses the **sliced-data (concatenate-compute-slice)** cross-chunk
    strategy: for multi-chunk inputs, up to ``window - 1`` boundary rows are
    fetched from neighbor chunks, concatenated, the aggregation is computed
    over the concatenation, and the result is sliced back to the chunk's range.

    Parameters
    ----------
    liteframe : LiteFrame
        The frame to compute over.
    window : int
        Size of the moving window (number of rows). Must be a positive integer.
        Offset/time-based windows are not yet supported.
    min_periods : int or None, default None
        Minimum number of observations in a window required to produce a
        value (otherwise null). Defaults to ``window`` (matching pandas).
    center : bool, default False
        If True, the window is centered on the current row.
    columns : str, list of str, or None, default None
        Column(s) to aggregate. If None, all (non-hidden, non-group-key)
        columns are aggregated. Non-selected columns are passed through.
    groupby_params : dict or None, default None
        Set when created from a ``groupby.rolling()`` call so the tiler
        partitions data by group key before computing per-group windows.
    """

    def __init__(
        self,
        liteframe,
        window,
        min_periods=None,
        center=False,
        columns=None,
        groupby_params=None,
        order_by=None,
        order_by_ascending=True,
        window_shift=0,
    ):
        # bool is a subclass of int — reject it explicitly so True/False are
        # not silently accepted as window sizes.
        if isinstance(window, bool) or not isinstance(window, int):
            raise NotImplementedError(
                "Only integer window sizes are supported for rolling; "
                "offset/time-based windows are not yet implemented."
            )
        if window < 1:
            raise ValueError(f"window must be a positive integer, got {window}")
        if min_periods is not None:
            if isinstance(min_periods, bool) or not isinstance(min_periods, int):
                raise TypeError(
                    f"min_periods must be an int or None, "
                    f"got {type(min_periods).__name__}"
                )
            if min_periods < 0:
                raise ValueError(f"min_periods must be >= 0, got {min_periods}")
            if min_periods > window:
                raise ValueError(
                    f"min_periods ({min_periods}) must not exceed window " f"({window})"
                )
        columns = _normalize_columns(columns)
        if isinstance(window_shift, bool) or not isinstance(window_shift, int):
            raise TypeError(
                f"window_shift must be an int, " f"got {type(window_shift).__name__}"
            )

        self._liteframe = liteframe
        self._window = window
        self._min_periods = min_periods
        self._center = center
        self._columns = columns
        self._groupby_params = groupby_params
        self._order_by = order_by
        self._order_by_ascending = order_by_ascending
        self._window_shift = window_shift

    def _agg(self, func):
        """Build and apply the rolling_agg window transform operator."""
        order_by, order_by_ascending = _normalize_order_by(
            self._order_by, self._order_by_ascending
        )
        op = LiteFrameWindowTransform(
            kind="rolling_agg",
            window_size=self._window,
            min_periods=self._min_periods,
            center=self._center,
            agg_func=func,
            columns=self._columns,
            groupby_params=self._groupby_params,
            order_by=order_by,
            order_by_ascending=order_by_ascending,
            window_shift=self._window_shift,
        )
        return op(self._liteframe)

    def sum(self):
        """Rolling sum.

        Returns
        -------
        LiteFrame
            Lazy LiteFrame with the rolling sum over each window.  Call
            ``.execute()`` to materialize.
        """
        return self._agg("sum")

    def mean(self):
        """Rolling mean.

        Returns
        -------
        LiteFrame
            Lazy LiteFrame with the rolling mean over each window.  Call
            ``.execute()`` to materialize.
        """
        return self._agg("mean")

    def min(self):
        """Rolling minimum.

        Returns
        -------
        LiteFrame
            Lazy LiteFrame with the rolling minimum over each window.  Call
            ``.execute()`` to materialize.
        """
        return self._agg("min")

    def max(self):
        """Rolling maximum.

        Returns
        -------
        LiteFrame
            Lazy LiteFrame with the rolling maximum over each window.  Call
            ``.execute()`` to materialize.
        """
        return self._agg("max")

    def std(self):
        """Rolling standard deviation (ddof=1).

        Returns
        -------
        LiteFrame
            Lazy LiteFrame with the rolling standard deviation over each
            window (sample std, ddof=1).  Call ``.execute()`` to materialize.
        """
        return self._agg("std")

    def var(self):
        """Rolling variance (ddof=1).

        Returns
        -------
        LiteFrame
            Lazy LiteFrame with the rolling variance over each window
            (sample variance, ddof=1).  Call ``.execute()`` to materialize.
        """
        return self._agg("var")

    def median(self):
        """Rolling median.

        Returns
        -------
        LiteFrame
            Lazy LiteFrame with the rolling median over each window.  Call
            ``.execute()`` to materialize.
        """
        return self._agg("median")

    def count(self):
        """Rolling count of non-null observations.

        Returns
        -------
        LiteFrame
            Lazy LiteFrame with the rolling count of non-null values over
            each window.  Call ``.execute()`` to materialize.
        """
        return self._agg("count")

    def agg(self, func):
        """Aggregate using one of the supported function names.

        Parameters
        ----------
        func : str
            One of ``sum``, ``mean``, ``min``, ``max``, ``std``, ``var``,
            ``median``, ``count``.

        Returns
        -------
        LiteFrame
            Lazy LiteFrame with the requested rolling aggregation applied.
            Call ``.execute()`` to materialize.
        """
        return self._agg(func)


def rolling(
    self,
    window,
    min_periods=None,
    center=False,
    columns=None,
    order_by=None,
    order_by_ascending=True,
    window_shift=0,
):
    """Provide rolling-window calculations.

    Returns a :class:`LiteFrameRolling` object whose aggregation methods
    (``.sum()``, ``.mean()``, ...) compute a fixed-size moving window
    aggregation. Only integer window sizes are supported.

    Parameters
    ----------
    window : int
        Size of the moving window (number of rows). Must be positive.
    min_periods : int or None, default None
        Minimum number of observations in a window required to produce a
        value (otherwise null). Defaults to ``window`` (matching pandas).
        For ``count`` the value is the number of non-null observations, but
        ``min_periods`` still gates on the number of window rows (observations),
        not the non-null count — consistent with pandas.
    center : bool, default False
        Center the window on the current row.
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
        Offset the window frame by N rows. A positive value shifts the window
        forward (toward later rows), a negative value backward. Matches the
        MaxCompute ``shift`` MF-specific window parameter.

    Returns
    -------
    LiteFrameRolling
        Intermediate object; call an aggregation method to produce a LiteFrame.

    Raises
    ------
    NotImplementedError
        If ``window`` is not an integer (offset/time windows unsupported),
        or if MF-specific fields (``order_by``/``window_shift``) are used with
        grouped or multi-chunk inputs.
    ValueError
        If ``window`` is not positive or ``min_periods`` is invalid.

    See Also
    --------
    expanding : Provide expanding window calculations.
    shift : Shift rows by a given number of periods.
    """
    return LiteFrameRolling(
        self,
        window,
        min_periods=min_periods,
        center=center,
        columns=columns,
        order_by=order_by,
        order_by_ascending=order_by_ascending,
        window_shift=window_shift,
    )
