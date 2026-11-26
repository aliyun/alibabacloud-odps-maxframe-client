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

"""LiteFrame.quantile and LiteFrame.median API entry points."""

import numpy as np

from maxframe.liteframe.reduction.compile import AggCall
from maxframe.liteframe.utils import is_numeric_dtype, normalize_col_name

_VALID_INTERPOLATIONS = ("linear", "lower", "higher", "midpoint", "nearest")


def _format_q_percent(q):
    """Format a quantile as an integer-or-decimal percent string.

    Shared core of ``_format_q_suffix`` (prefix ``q``) and
    ``describe._format_stat_label`` (suffix ``%``): 0.25 -> '25', 0.05 -> '5'.
    """
    s = repr(round(float(q) * 100, 10))
    if s.endswith(".0"):
        s = s[:-2]
    return s


def _format_q_suffix(q):
    return "q" + _format_q_percent(q)


def _normalize_q(q):
    if isinstance(q, (list, tuple, np.ndarray)):
        qs = [float(v) for v in q]
    else:
        qs = [float(q)]
    if not qs:
        raise ValueError("q must contain at least one value")
    for v in qs:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"quantile q={v} is out of range; must be in [0, 1]")
    return qs


def _resolve_target_columns(data, numeric_only):
    """Resolve target columns, validating that they are numeric.

    When ``numeric_only=True``, non-numeric columns are silently excluded.
    When ``numeric_only=False`` (default), a ``TypeError`` is raised listing
    the non-numeric columns so the caller gets a clear client-side error
    instead of a cryptic failure deep in execution.
    """
    dtypes = data.dtypes
    cols = [normalize_col_name(c) for c in dtypes.index]
    if numeric_only:
        cols = [c for c in cols if is_numeric_dtype(dtypes[c])]
    else:
        bad = [c for c in cols if not is_numeric_dtype(dtypes[c])]
        if bad:
            raise TypeError(f"quantile requires numeric columns; non-numeric: {bad}")
    if not cols:
        raise ValueError("No columns to aggregate after numeric_only")
    return cols


def _build_quantile_agg_calls(qs, interpolation, approx):
    if len(qs) == 1:
        return [
            AggCall("quantile", q=qs[0], interpolation=interpolation, approx=approx)
        ]
    return [
        AggCall(
            "quantile",
            q=v,
            interpolation=interpolation,
            approx=approx,
            output_suffix=_format_q_suffix(v),
        )
        for v in qs
    ]


def _prepare_quantile_agg_func(q, interpolation, approx):
    """Validate inputs and build the raw_func (AggCall or list) for quantile."""
    if interpolation not in _VALID_INTERPOLATIONS:
        raise ValueError(
            f"Invalid interpolation {interpolation!r}; must be one of "
            f"{list(_VALID_INTERPOLATIONS)}"
        )
    qs = _normalize_q(q)
    agg_calls = _build_quantile_agg_calls(qs, interpolation, approx)
    return agg_calls if len(agg_calls) > 1 else agg_calls[0]


def quantile(
    data, q=0.5, interpolation="linear", numeric_only=False, approx=False, method="auto"
):
    """Return values at the given quantile for each column.

    Parameters
    ----------
    q : float or list of float, default 0.5
        Quantile(s) to compute, in the range ``[0, 1]``.  A scalar keeps
        original column names; a list produces wide columns named
        ``{col}_q{pct}`` (e.g. ``a_q50`` for q=0.5).
    interpolation : {'linear', 'lower', 'higher', 'midpoint', 'nearest'}
        Interpolation method when the quantile lies between two data points.
    numeric_only : bool, default False
        If ``True``, silently exclude non-numeric columns.  If ``False``,
        raise ``TypeError`` on non-numeric columns.
    approx : bool, default False
        If ``True``, use an approximate grid-summary sketch on the
        distributed path (rank error ~1%).  NaNs are excluded in both
        modes; ±inf are kept as valid order statistics in both modes.
    method : {'auto', 'tree', 'shuffle'}, default 'auto'
        Execution strategy for grouped/multi-chunk aggregation.

    Returns
    -------
    LiteFrame
        One-row frame (whole-frame) or one-row-per-group frame with
        quantile values per column.
    """
    _resolve_target_columns(data, numeric_only)
    # Uniform LiteFrameAgg: the DPE tiler decides whether to delegate the
    # exact non-groupby case to TensorQuantile (see LiteFrameAggTiler).
    raw_func = _prepare_quantile_agg_func(q, interpolation, approx)
    return data.agg(raw_func, method=method, numeric_only=numeric_only)


def median(data, method="auto", numeric_only=None, *, approx=False, **kw):
    """Return the median (0.5 quantile, linear) of each column.

    Parameters
    ----------
    method : {'auto', 'tree', 'shuffle'}, default 'auto'
        Execution strategy for grouped/multi-chunk aggregation.
    numeric_only : bool or None, default None
        If ``True``, silently exclude non-numeric columns.  If ``False``
        or ``None``, raise ``TypeError`` on non-numeric columns.
    approx : bool, default False
        If ``True``, use an approximate grid-summary sketch on the
        distributed path.  See :func:`quantile` for semantics.
    **kw
        Additional keyword arguments forwarded to :meth:`LiteFrame.quantile`.

    Returns
    -------
    LiteFrame
        One-row frame (whole-frame) or one-row-per-group frame with
        the median value per column.
    """
    return data.quantile(
        q=0.5,
        interpolation="linear",
        numeric_only=numeric_only,
        approx=approx,
        method=method,
        **kw,
    )
