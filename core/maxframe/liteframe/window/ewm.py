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

import math

from maxframe.liteframe.operators.window import EWM_AGG_FUNCS, LiteFrameWindowTransform


def _resolve_com(com=None, span=None, halflife=None, alpha=None):
    """Resolve the center of mass (``com``) from one of four specifications.

    Exactly one of ``com``, ``span``, ``halflife``, ``alpha`` must be
    provided, mirroring the pandas API.  ``com`` is the canonical
    representation of the decay parameter: all four specs are converted
    through ``com`` so that boundary values (e.g. ``halflife=1e17``)
    are preserved accurately.

    Parameters
    ----------
    com : float, optional
        Center of mass directly.
    span : float, optional
        Span (N-day EMA): ``com = (span - 1) / 2``.
    halflife : float, optional
        Half-life: ``com = 1/decay - 1`` where ``decay = 1 - exp(ln(0.5)/halflife)``.
    alpha : float, optional
        Smoothing factor directly: ``0 < alpha <= 1``, ``com = (1-alpha)/alpha``.

    Returns
    -------
    float
        The center of mass ``com`` (``>= 0``, or ``inf``).
    """
    count = sum(x is not None for x in (com, span, halflife, alpha))
    if count == 0:
        raise ValueError("Must pass one of com, span, halflife, or alpha")
    if count > 1:
        raise ValueError("com, span, halflife, and alpha are mutually exclusive")

    # Canonicalize all four specifications through a single com path,
    # mirroring pandas' get_center_of_mass.  Performing the spec arithmetic
    # before float() preserves Decimal/Fraction inputs.  For very large
    # halflife, decay rounds to 0.0; pandas uses NumPy division (1/0 -> inf)
    # giving com=inf, so mirror that.
    if com is not None:
        if com < 0:
            raise ValueError(f"com must be >= 0, got {com}")
        com_val = com
    elif span is not None:
        if span < 1:
            raise ValueError(f"span must be >= 1, got {span}")
        com_val = (span - 1) / 2
    elif halflife is not None:
        if halflife <= 0:
            raise ValueError(f"halflife must be > 0, got {halflife}")
        decay = 1 - math.exp(math.log(0.5) / halflife)
        com_val = math.inf if decay == 0 else 1 / decay - 1
    else:
        if alpha <= 0 or alpha > 1:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        com_val = (1 - alpha) / alpha

    return float(com_val)


class LiteFrameEWM:
    """Provide exponentially weighted (EW) calculations.

    Returns an intermediate object whose aggregation methods (``.mean()``,
    ``.std()``, ``.var()``) compute exponentially weighted moving statistics.

    Parameters
    ----------
    liteframe : LiteFrame
        The LiteFrame to compute on.
    com : float
        Center of mass.  Resolved from ``com``/``span``/``halflife``/
        ``alpha`` via :func:`_resolve_com` at the public API boundary.
    min_periods : int, default 0
        Minimum number of observations in the window required to produce a
        value (otherwise null).
    adjust : bool, default True
        If True, use the adjusted (weighted average) form.  If False, use
        the recursive form (y_t = (1-α)·y_{t-1} + α·x_t).
    ignore_na : bool, default False
        If False, NaN positions are treated as absolute gaps (weights decay
        through the gap).  If True, NaN positions are ignored (weights
        apply only to observed values, relative positioning).
    bias : bool, default False
        If False, use unbiased variance (bias correction).  If True, use
        biased variance.  Only applies to ``std``/``var``.
    columns : str, list of str, or None, default None
        Column(s) to aggregate. If None, all numeric columns are aggregated.
    groupby_params : dict or None, default None
        Set when created from a ``groupby.ewm()`` call.
    order_by : str, list of str, or None, default None
        Column(s) that define the order in which the window expands
        (equivalent to ``ORDER BY`` inside a MaxCompute ``OVER()`` clause).
        The output row order is unchanged; each row receives the EWM
        computed over the rows that precede it in *order_by* order.
    order_by_ascending : bool or list of bool, default True
        Sort direction for each *order_by* column. A single bool is
        broadcast to all columns.
    window_shift : int, default 0
        Offset the EWM leading edge by N rows. A positive value shifts
        the window forward (look-ahead), a negative value backward
        (look-back). Matches the MaxCompute ``shift`` MF-specific window
        parameter.
    reverse : bool, default False
        Flip the EWM so the recurrence runs from the last row backward.
        Matches the MaxCompute ``reverse_range`` MF-specific window
        parameter. May be combined with ``window_shift``.
    placeback : bool, default True
        If True, restore output to input frame's row order (grouped only).
    """

    def __init__(
        self,
        liteframe,
        com,
        min_periods=0,
        *,
        adjust=True,
        ignore_na=False,
        bias=False,
        columns=None,
        groupby_params=None,
        order_by=None,
        order_by_ascending=True,
        window_shift=0,
        reverse=False,
        placeback=True,
    ):
        if isinstance(window_shift, bool) or not isinstance(window_shift, int):
            raise TypeError(
                f"window_shift must be an int, " f"got {type(window_shift).__name__}"
            )
        if isinstance(min_periods, bool) or not isinstance(min_periods, int):
            raise TypeError(
                f"min_periods must be an int, " f"got {type(min_periods).__name__}"
            )
        if min_periods < 0:
            raise ValueError(f"min_periods must be >= 0, got {min_periods}")
        self._liteframe = liteframe
        self._com = com
        self._min_periods = min_periods
        self._adjust = adjust
        self._ignore_na = ignore_na
        self._bias = bias
        self._columns = columns
        self._groupby_params = groupby_params
        self._order_by = order_by
        self._order_by_ascending = order_by_ascending
        self._window_shift = window_shift
        self._reverse = reverse
        self._placeback = placeback

    @property
    def com(self):
        """Center of mass."""
        return self._com

    @property
    def span(self):
        """Span (N-day EMA): ``2*com + 1``."""
        return 2 * self._com + 1

    @property
    def halflife(self):
        """Half-life: ``ln(0.5) / ln(1-alpha)``.

        Returns ``None`` when ``alpha`` is 0 or 1 (``com`` is ``inf`` or
        0) since the logarithm is undefined.
        """
        alpha = 1 / (1 + self._com)
        if alpha <= 0 or alpha >= 1:
            return None
        return math.log(0.5) / math.log(1 - alpha)

    @property
    def alpha(self):
        """Smoothing factor: ``1 / (1 + com)``."""
        return 1 / (1 + self._com)

    def _agg(self, func):
        """Build and apply the ewm_agg window transform operator."""
        if func not in EWM_AGG_FUNCS:
            raise ValueError(
                f"Unsupported EWM aggregation function: {func!r}. "
                f"Supported: {sorted(EWM_AGG_FUNCS)}"
            )
        op = LiteFrameWindowTransform.for_ewm(
            func,
            com=self._com,
            min_periods=self._min_periods,
            adjust=self._adjust,
            ignore_na=self._ignore_na,
            bias=self._bias,
            columns=self._columns,
            groupby_params=self._groupby_params,
            order_by=self._order_by,
            order_by_ascending=self._order_by_ascending,
            window_shift=self._window_shift,
            reverse=self._reverse,
            placeback=self._placeback,
        )
        return op(self._liteframe)

    def mean(self):
        """Exponentially weighted moving mean.

        Returns
        -------
        LiteFrame
            Lazy LiteFrame with the EWM mean. Call ``.execute()`` to
            materialize.
        """
        return self._agg("mean")

    def std(self):
        """Exponentially weighted moving standard deviation.

        Returns
        -------
        LiteFrame
            Lazy LiteFrame with the EWM standard deviation. Call
            ``.execute()`` to materialize.
        """
        return self._agg("std")

    def var(self):
        """Exponentially weighted moving variance.

        Returns
        -------
        LiteFrame
            Lazy LiteFrame with the EWM variance. Call ``.execute()`` to
            materialize.
        """
        return self._agg("var")

    def agg(self, func):
        """Aggregate using one of the supported function names.

        Parameters
        ----------
        func : str
            One of ``mean``, ``std``, ``var``.

        Returns
        -------
        LiteFrame
            Lazy LiteFrame with the requested EWM aggregation applied.
        """
        return self._agg(func)


def ewm(
    self,
    com=None,
    span=None,
    halflife=None,
    alpha=None,
    min_periods=0,
    *,
    adjust=True,
    ignore_na=False,
    bias=False,
    columns=None,
    order_by=None,
    order_by_ascending=True,
    window_shift=0,
    reverse=False,
    placeback=True,
):
    """Provide exponentially weighted (EW) calculations.

    Returns a :class:`LiteFrameEWM` object whose aggregation methods
    (``.mean()``, ``.std()``, ``.var()``) compute exponentially weighted
    moving statistics.

    Exactly one of ``com``, ``span``, ``halflife``, or ``alpha`` must be
    provided to specify the smoothing factor.

    Parameters
    ----------
    com : float, optional
        Specify decay in terms of center of mass: ``alpha = 1 / (1 + com)``.
    span : float, optional
        Specify decay in terms of span (N-day EMA):
        ``alpha = 2 / (span + 1)``.
    halflife : float, optional
        Specify decay in terms of half-life:
        ``alpha = 1 - exp(ln(0.5) / halflife)``.
    alpha : float, optional
        Specify smoothing factor directly: ``0 < alpha <= 1``.
    min_periods : int, default 0
        Minimum number of observations in the window required to produce a
        value (otherwise null).
    adjust : bool, default True
        If True, use the adjusted (weighted average) form.  If False, use
        the recursive form (y_t = (1-α)·y_{t-1} + α·x_t).
    ignore_na : bool, default False
        If False, NaN positions are treated as absolute gaps (weights decay
        through the gap).  If True, NaN positions are ignored (weights
        apply only to observed values, relative positioning).
    bias : bool, default False
        If False, use unbiased variance (bias correction).  If True, use
        biased variance.  Only applies to ``std``/``var``.
    columns : str, list of str, or None, default None
        Column(s) to aggregate. If None, all numeric columns are aggregated.
    order_by : str, list of str, or None, default None
        Column(s) that define the order in which the window expands
        (equivalent to ``ORDER BY`` inside a MaxCompute ``OVER()`` clause).
        The output row order is unchanged; each row receives the EWM
        computed over the rows that precede it in *order_by* order.
    order_by_ascending : bool or list of bool, default True
        Sort direction for each *order_by* column. A single bool is
        broadcast to all columns.
    window_shift : int, default 0
        Offset the EWM leading edge by N rows. A positive value shifts
        the window forward (look-ahead), a negative value backward
        (look-back). Matches the MaxCompute ``shift`` MF-specific window
        parameter.
    reverse : bool, default False
        Flip the EWM so the recurrence runs from the last row backward
        (a "narrowing" window). Matches the MaxCompute ``reverse_range``
        MF-specific window parameter. May be combined with
        ``window_shift``: the window then covers
        ``[i + window_shift, n-1]``.
    placeback : bool, default True
        If True, restore the output to the input frame's row order and chunk
        layout after the group-key shuffle (grouped EWM only).  If False,
        the output stays in reducer/hash-partition order.

    Returns
    -------
    LiteFrameEWM
        Intermediate object; call an aggregation method to produce a
        LiteFrame.

    See Also
    --------
    rolling : Provide rolling-window calculations.
    expanding : Provide expanding-window calculations.

    Notes
    -----
    With ``adjust=True`` (default), the EW mean at position *t* is a
    weighted average of all observations up to *t*, where the weight for
    observation *i* is proportional to ``(1 - alpha)^(t - i)``.  Earlier
    observations receive exponentially decaying weights.

    With ``adjust=False``, the EW mean is computed recursively:
    ``y_0 = x_0``, ``y_t = (1 - α)·y_{t-1} + α·x_t``.  This requires less
    memory but weights differ slightly from the adjusted form for the first
    few observations.

    For ``std`` and ``var``, the unbiased estimator (``bias=False``) applies
    a bias correction factor derived from the sum of weights, similar to the
    ``ddof=1`` correction in sample variance.  Set ``bias=True`` to skip
    this correction.

    See the `pandas EWM documentation
    <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.ewm.html>`_
    for the mathematical background on ``adjust``, ``ignore_na``, and
    ``bias``.
    """
    resolved_com = _resolve_com(com=com, span=span, halflife=halflife, alpha=alpha)
    return LiteFrameEWM(
        self,
        com=resolved_com,
        min_periods=min_periods,
        adjust=adjust,
        ignore_na=ignore_na,
        bias=bias,
        columns=columns,
        order_by=order_by,
        order_by_ascending=order_by_ascending,
        window_shift=window_shift,
        reverse=reverse,
        placeback=placeback,
    )
