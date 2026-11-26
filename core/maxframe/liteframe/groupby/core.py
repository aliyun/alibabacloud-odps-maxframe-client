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

import numpy as np
import pandas as pd

from maxframe.liteframe.arithmetic.core import _build_fused_projection
from maxframe.liteframe.expressions import LiteFrameColumn, NamedExpr
from maxframe.liteframe.functional.flat_apply import _normalize_subset
from maxframe.liteframe.operators.reduction import LiteFrameAgg
from maxframe.liteframe.reduction.compile import AggCall
from maxframe.liteframe.reduction.cov_corr import (
    _pairwise_matrix_expand,
    _resolve_label_col,
    _resolve_numeric_cols,
)
from maxframe.liteframe.reduction.custom_agg import IdxMaxAgg, IdxMinAgg
from maxframe.liteframe.reduction.describe import (
    _build_col_key_map,
    _build_describe_agg_funcs,
    _build_out_dtypes,
    _build_stat_pairs,
    _describe_expand,
    _resolve_describe_cols,
    _validate_percentiles,
)
from maxframe.liteframe.reduction.value_counts import _check_reserved_names
from maxframe.liteframe.utils import resolve_unique_col_name
from maxframe.udf import builtin_function

# Data marker column used by ``LiteFrameGroupBy.filter(dropna=False)``: the
# predicate wrapper stamps each row's verdict so rejected rows can be nulled
# after the group keys have been attached.  Internal columns use the shared
# ``__mf_`` prefix (see ``operators/groupby.py``, ``operators/drop_duplicates.py``).
_FILTER_KEEP_COL = "__mf_groupby_filter_keep__"


def _filter_group_keep(group, predicate, args, kwargs):
    """Evaluate the user predicate on one group and normalise it to a bool."""
    result = predicate(group, *args, **kwargs)
    # pandas squeezes the predicate result, so a length-1 Series/DataFrame
    # (``g["val"] > 0`` on a one-row group) collapses to a scalar and is
    # accepted.  A 0-d ndarray stays an ndarray, so ``np.array(True)`` is
    # still rejected — as in pandas.
    try:
        result = result.squeeze()
    except AttributeError:
        pass
    if isinstance(result, (bool, np.bool_)):
        return bool(result)
    # Reject non-scalar results (Series, arrays, frames).
    # numpy scalars (np.float16, etc.) have .shape=() but are scalars.
    if hasattr(result, "__iter__") or (
        hasattr(result, "shape") and not isinstance(result, np.generic)
    ):
        raise TypeError(
            f"filter function returned a {type(result).__name__}, "
            f"but expected a scalar bool"
        )
    # A scalar missing value (pd.NA, np.nan, None, pd.NaT,
    # np.datetime64("NaT"), etc.) is falsy — pandas drops the group.
    # Any other non-bool scalar is invalid.
    try:
        if pd.isna(result):
            return False
    except (TypeError, ValueError):
        pass
    raise TypeError(
        f"filter function returned a {type(result).__name__}, "
        f"but expected a scalar bool"
    )


@builtin_function
def _filter_group_wrapper(
    group, predicate, predicate_args, predicate_kwargs, dropna, marker_col
):
    """Keep or drop one group for ``LiteFrameGroupBy.filter``.

    A ``@builtin_function`` rather than a closure, so the worker reloads the
    body by name instead of pickling it: the user predicate and its arguments
    are forwarded through the apply ``args``.  ``LiteFrameFunctional`` sees a
    callable in those args and keeps the op on the custom-code path, so the
    predicate arrives unpacked (and the user code stays isolated).
    """
    keep = _filter_group_keep(group, predicate, predicate_args, predicate_kwargs)
    if dropna:
        return group if keep else group.iloc[:0]
    # pandas dropna=False: keep every group's rows and null out the rejected
    # groups' values afterwards.  A data marker column records each row's
    # verdict; the key columns are not visible to the UDF here, so the nulling
    # happens in a later projection (see ``_nullify_rejected_groups``).
    out = group.copy()
    out[marker_col] = keep
    return out


def _nullify_rejected_groups(result, mark):
    """Set every column to null where the ``mark`` marker column is False.

    ``LiteFrameGroupBy.filter(dropna=False)`` keeps rejected groups' rows but
    blanks all their values (group keys included), matching pandas.  The keys
    are attached after the group UDF runs, so the nulling is applied here over
    the assembled result.
    """
    cols = list(result.columns)
    marker = LiteFrameColumn(name=mark, dtype=None)
    cond = _build_fused_projection(
        result,
        [NamedExpr(name=c, expr=marker, dtype=None) for c in cols],
    )
    return result.where(cond, None)


class LiteFrameGroupBy:
    """Intermediate object returned by LiteFrame.groupby().

    Supports aggregation via .agg() and shortcut methods like .sum(), .mean(), etc.
    Always uses column-based output (as_index=False equivalent), no multi-index.

    Parameters
    ----------
    liteframe : LiteFrame
        The LiteFrame to group.
    by : str or list of str
        Column name(s) to group by.
    sort : bool, default True
        Whether to sort group keys in the result.
    dropna : bool, default True
        If True, rows with null group keys are dropped before aggregation.
        If False, null-key rows form their own group.
    """

    def __init__(self, liteframe, by, sort=True, dropna=True, group_keys=True):
        if by is None:
            raise ValueError("'by' argument must be specified for groupby")
        if isinstance(by, str):
            if by not in liteframe.dtypes.index:
                raise KeyError(f"Column '{by}' not found in LiteFrame")
        elif isinstance(by, (list, tuple)):
            missing = [c for c in by if c not in liteframe.dtypes.index]
            if missing:
                raise KeyError(f"Columns {missing} not found in LiteFrame")
        else:
            raise TypeError(
                f"'by' must be a column name (str) or list of column names, "
                f"got {type(by).__name__}"
            )
        self._liteframe = liteframe
        self._by = by
        self._sort = sort
        self._dropna = dropna
        self._group_keys = group_keys
        self._selection = None  # set by __getitem__ for column selection

    @property
    def groupby_params(self):
        params = {
            "by": self._by,
            "sort": self._sort,
            "dropna": self._dropna,
            "group_keys": self._group_keys,
        }
        if self._selection is not None:
            params["selection"] = list(self._selection)
        return params

    def __getitem__(self, selection):
        """Select columns for subsequent apply/transform operations.

        Following pandas 3.0 semantics, the groupby key columns are excluded
        from the UDF input by default.  The selection further restricts
        which non-key columns the UDF receives.

        Parameters
        ----------
        selection : str or list of str
            Column name(s) to select.

        Returns
        -------
        LiteFrameGroupBy
            A new LiteFrameGroupBy with the column selection set.
        """
        if isinstance(selection, str):
            selection = [selection]
        elif isinstance(selection, (list, tuple)):
            selection = list(selection)
        else:
            raise TypeError(
                f"Selection must be a column name or list of names, "
                f"got {type(selection).__name__}"
            )
        # Validate that selected columns exist
        for c in selection:
            if c not in self._liteframe.dtypes.index:
                raise KeyError(f"Column '{c}' not found in LiteFrame")
        new_gb = LiteFrameGroupBy(
            self._liteframe, self._by, self._sort, self._dropna, self._group_keys
        )
        new_gb._selection = selection
        return new_gb

    def agg(
        self,
        func=None,
        *,
        method="auto",
        numeric_only=None,
        input_frame_type="pandas",
        **kwargs,
    ):
        """Aggregate the grouped LiteFrame using one or more functions.

        Supports string function names (e.g. ``"sum"``, ``"mean"``),
        callables, or lists thereof.  The result has one row per group with
        the group key columns prepended; no MultiIndex is produced.

        Parameters
        ----------
        func : str, callable, list, or dict, optional
            Aggregation specification.  A string names a built-in reduction
            (``"sum"``, ``"mean"``, ``"min"``, ``"max"``, ``"prod"``,
            ``"count"``, ``"size"``, ``"var"``, ``"std"``, ``"median"``,
            ``"sem"``, ``"skew"``, ``"kurt"``, ``"nunique"``, ``"all"``,
            ``"any"``).  A callable is applied per group.  A list applies
            multiple aggregations to all (non-key) columns.  A dict maps
            column names to function(s).  If None, only the group keys are
            returned.
        method : {"auto", "shuffle", "map_reduce"}, default "auto"
            Execution strategy for the aggregation.
        numeric_only : bool or None, default None
            If True, only numeric columns are aggregated.  If None, the
            default depends on the function.
        input_frame_type : {"pandas", "polars"}, default "pandas"
            Frame type passed to callable aggregations.
        **kwargs
            Additional keyword arguments forwarded to the aggregation
            function(s).

        Returns
        -------
        LiteFrame
            Lazy LiteFrame with one row per group and the aggregation
            results in the remaining columns.  Call ``.execute()`` to
            materialize.
        """
        op = LiteFrameAgg(
            raw_func=func,
            raw_func_kw=kwargs if kwargs else None,
            groupby_params=self.groupby_params,
            method=method,
            numeric_only=numeric_only,
            input_frame_type=input_frame_type,
        )
        return op(self._liteframe)

    def sum(self, min_count=0, **kw):
        """Compute the sum of each group.

        Parameters
        ----------
        min_count : int, default 0
            The required number of valid (non-null) values to perform the
            operation.  If fewer than ``min_count`` non-null values are
            present in a group, the result is null.
        **kw
            Additional keyword arguments forwarded to :meth:`agg`.

        Returns
        -------
        LiteFrame
            Lazy LiteFrame with one row per group and the per-group sum.
            Call ``.execute()`` to materialize.
        """
        return self.agg(AggCall("sum", min_count=min_count), **kw)

    def mean(self, **kw):
        """Compute the mean of each group.

        Parameters
        ----------
        **kw
            Additional keyword arguments forwarded to :meth:`agg`.

        Returns
        -------
        LiteFrame
            Lazy LiteFrame with one row per group and the per-group mean.
            Call ``.execute()`` to materialize.
        """
        return self.agg("mean", **kw)

    def min(self, min_count=-1, **kw):
        """Compute the minimum of each group.

        Parameters
        ----------
        min_count : int, default -1
            The required number of valid (non-null) values to perform the
            operation.  If fewer than ``min_count`` non-null values are
            present in a group, the result is null.  The default ``-1``
            means no guard (always compute the minimum).
        **kw
            Additional keyword arguments forwarded to :meth:`agg`.

        Returns
        -------
        LiteFrame
            Lazy LiteFrame with one row per group and the per-group minimum.
            Call ``.execute()`` to materialize.
        """
        return self.agg(AggCall("min", min_count=min_count), **kw)

    def max(self, min_count=-1, **kw):
        """Compute the maximum of each group.

        Parameters
        ----------
        min_count : int, default -1
            The required number of valid (non-null) values to perform the
            operation.  If fewer than ``min_count`` non-null values are
            present in a group, the result is null.  The default ``-1``
            means no guard (always compute the maximum).
        **kw
            Additional keyword arguments forwarded to :meth:`agg`.

        Returns
        -------
        LiteFrame
            Lazy LiteFrame with one row per group and the per-group maximum.
            Call ``.execute()`` to materialize.
        """
        return self.agg(AggCall("max", min_count=min_count), **kw)

    def prod(self, min_count=0, **kw):
        """Compute the product of each group.

        Parameters
        ----------
        min_count : int, default 0
            The required number of valid (non-null) values to perform the
            operation.  If fewer than ``min_count`` non-null values are
            present in a group, the result is null.
        **kw
            Additional keyword arguments forwarded to :meth:`agg`.

        Returns
        -------
        LiteFrame
            Lazy LiteFrame with one row per group and the per-group product.
            Call ``.execute()`` to materialize.
        """
        return self.agg(AggCall("prod", min_count=min_count), **kw)

    def count(self, **kw):
        """Compute the count of non-null values in each group.

        Parameters
        ----------
        **kw
            Additional keyword arguments forwarded to :meth:`agg`.

        Returns
        -------
        LiteFrame
            Lazy LiteFrame with one row per group and the per-group
            non-null count.  Call ``.execute()`` to materialize.
        """
        return self.agg("count", **kw)

    def size(self, **kw):
        """Compute the number of rows in each group.

        Parameters
        ----------
        **kw
            Additional keyword arguments forwarded to :meth:`agg`.

        Returns
        -------
        LiteFrame
            Lazy LiteFrame with one row per group and the per-group row
            count (including null-value rows).  Call ``.execute()`` to
            materialize.
        """
        return self.agg("size", **kw)

    def var(self, ddof=1, **kw):
        """Compute the variance of each group.

        Parameters
        ----------
        ddof : int, default 1
            Delta degrees of freedom.  The divisor is ``n - ddof``.
        **kw
            Additional keyword arguments forwarded to :meth:`agg`.

        Returns
        -------
        LiteFrame
            Lazy LiteFrame with one row per group and the per-group
            variance.  Call ``.execute()`` to materialize.
        """
        return self.agg(AggCall("var", ddof=ddof), **kw)

    def std(self, ddof=1, **kw):
        """Compute the standard deviation of each group.

        Parameters
        ----------
        ddof : int, default 1
            Delta degrees of freedom.  The divisor is ``n - ddof``.
        **kw
            Additional keyword arguments forwarded to :meth:`agg`.

        Returns
        -------
        LiteFrame
            Lazy LiteFrame with one row per group and the per-group
            standard deviation.  Call ``.execute()`` to materialize.
        """
        return self.agg(AggCall("std", ddof=ddof), **kw)

    def sem(self, ddof=1, **kw):
        """Compute the standard error of the mean of each group.

        Parameters
        ----------
        ddof : int, default 1
            Delta degrees of freedom.  The divisor is ``n - ddof``.
        **kw
            Additional keyword arguments forwarded to :meth:`agg`.

        Returns
        -------
        LiteFrame
            Lazy LiteFrame with one row per group and the per-group
            standard error of the mean.  Call ``.execute()`` to materialize.
        """
        return self.agg(AggCall("sem", ddof=ddof), **kw)

    def skew(self, **kw):
        """Compute the skewness of each group.

        Parameters
        ----------
        **kw
            Additional keyword arguments forwarded to :meth:`agg`.

        Returns
        -------
        LiteFrame
            Lazy LiteFrame with one row per group and the per-group
            skewness.  Call ``.execute()`` to materialize.
        """
        return self.agg("skew", **kw)

    def kurt(self, **kw):
        """Compute the kurtosis of each group.

        Parameters
        ----------
        **kw
            Additional keyword arguments forwarded to :meth:`agg`.

        Returns
        -------
        LiteFrame
            Lazy LiteFrame with one row per group and the per-group
            kurtosis.  Call ``.execute()`` to materialize.
        """
        return self.agg("kurt", **kw)

    def nunique(self, dropna=True, **kw):
        """Compute the number of distinct values in each group.

        Parameters
        ----------
        dropna : bool, default True
            If True, null values are excluded from the distinct count.
        **kw
            Additional keyword arguments forwarded to :meth:`agg`.

        Returns
        -------
        LiteFrame
            Lazy LiteFrame with one row per group and the per-group count
            of unique values.  Call ``.execute()`` to materialize.
        """
        return self.agg(AggCall("nunique", dropna=dropna), **kw)

    def idxmin(self, skipna=True, **kw):
        """Compute the row index of the minimum value in each group.

        The index is the 0-based integer row position in the original frame.

        Parameters
        ----------
        skipna : bool, default True
            Exclude NA/null values. If False, and any value is NA, the
            result is NA.
        **kw
            Additional keyword arguments forwarded to :meth:`agg`.

        Returns
        -------
        LiteFrame
            Lazy LiteFrame with one row per group and the per-group
            minimum-value row indices.  Call ``.execute()`` to materialize.
        """
        return self.agg(IdxMinAgg(skipna=skipna), **kw)

    def idxmax(self, skipna=True, **kw):
        """Compute the row index of the maximum value in each group.

        The index is the 0-based integer row position in the original frame.

        Parameters
        ----------
        skipna : bool, default True
            Exclude NA/null values. If False, and any value is NA, the
            result is NA.
        **kw
            Additional keyword arguments forwarded to :meth:`agg`.

        Returns
        -------
        LiteFrame
            Lazy LiteFrame with one row per group and the per-group
            maximum-value row indices.  Call ``.execute()`` to materialize.
        """
        return self.agg(IdxMaxAgg(skipna=skipna), **kw)

    def _pairwise_matrix(
        self,
        func_name,
        ddof=1,
        numeric_only=False,
        min_periods=None,
        method="auto",
        **kw,
    ):
        """Shared helper for groupby cov/corr: agg + flat_apply to N×N matrix."""
        op = LiteFrameAgg(
            raw_func=AggCall(
                func_name,
                ddof=ddof,
                min_periods=min_periods,
                numeric_only=numeric_only,
            ),
            groupby_params=self.groupby_params,
            method=method,
            numeric_only=numeric_only,
        )
        agg_result = op(self._liteframe)
        dtypes = self._liteframe.dtypes

        numeric_cols = self._numeric_cols_excluding_keys(sort=True)
        group_keys = [self._by] if isinstance(self._by, str) else list(self._by)

        label_col = _resolve_label_col(numeric_cols, group_keys)

        out_dtypes = {gk: str(dtypes.get(gk)) for gk in group_keys}
        out_dtypes[label_col] = "string"
        out_dtypes.update({c: "double" for c in numeric_cols})

        return agg_result.flat_apply(
            _pairwise_matrix_expand,
            args=(
                numeric_cols,
                func_name,
                label_col,
                group_keys,
                op.agg_output_columns,
            ),
            dtypes=out_dtypes,
        )

    def cov(self, ddof=1, numeric_only=False, min_periods=None, method="auto", **kw):
        """Compute pairwise covariance matrix per group.

        Parameters
        ----------
        ddof : int, default 1
            Delta degrees of freedom.  The divisor is ``n - ddof``.
        numeric_only : bool, default False
            If True, exclude non-numeric columns.  If False (pandas 2.0+
            default), non-numeric target columns raise TypeError.
        min_periods : int, optional
            Minimum non-null observations per pair; below this, NaN.
        method : {'auto', 'tree', 'shuffle'}, default 'auto'
            Execution strategy forwarded to :meth:`agg`.
        **kw
            Additional keyword arguments forwarded to :meth:`agg`.

        Returns
        -------
        LiteFrame
            Matrix with group key, ``columns`` row-label (probed to
            ``columns_1``, ``columns_2``, … on collision), and one column
            per numeric column.  One row per (group, matrix-row).
        """
        return self._pairwise_matrix(
            "cov",
            ddof=ddof,
            numeric_only=numeric_only,
            min_periods=min_periods,
            method=method,
            **kw,
        )

    def corr(self, ddof=1, numeric_only=False, min_periods=None, method="auto", **kw):
        """Compute pairwise correlation matrix per group.

        Parameters
        ----------
        ddof : int, default 1
            Delta degrees of freedom for the covariance/std denominator.
        numeric_only : bool, default False
            If True, exclude non-numeric columns.  If False (pandas 2.0+
            default), non-numeric target columns raise TypeError.
        min_periods : int, optional
            Minimum non-null observations per pair; below this, NaN.
        method : {'auto', 'tree', 'shuffle'}, default 'auto'
            Execution strategy forwarded to :meth:`agg`.
        **kw
            Additional keyword arguments forwarded to :meth:`agg`.

        Returns
        -------
        LiteFrame
            Matrix with group key, ``columns`` row-label (probed to
            ``columns_1``, ``columns_2``, … on collision), and one column
            per numeric column.  The diagonal is always 1.0.
        """
        return self._pairwise_matrix(
            "corr",
            ddof=ddof,
            numeric_only=numeric_only,
            min_periods=min_periods,
            method=method,
            **kw,
        )

    def median(self, **kw):
        """Compute the median of each group (0.5 quantile, linear).

        Re-implemented on top of :meth:`quantile`, gaining ``approx`` support.
        """
        return self.quantile(q=0.5, interpolation="linear", **kw)

    def quantile(
        self,
        q=0.5,
        interpolation="linear",
        numeric_only=False,
        approx=False,
        method="auto",
        **kw,
    ):
        """Compute the q-th quantile of each group.

        Parameters
        ----------
        q : float or list of float, default 0.5
            Quantile(s) to compute, in the range ``[0, 1]``.  A scalar
            keeps original column names; a list produces wide columns
            ``{col}_q{pct}``.
        interpolation : {'linear', 'lower', 'higher', 'midpoint', 'nearest'}
            Interpolation method when the quantile lies between two data
            points.
        numeric_only : bool, default False
            If ``True``, silently exclude non-numeric columns.
        approx : bool, default False
            If ``True``, use an approximate grid-summary sketch on the
            distributed path.  See :func:`LiteFrame.quantile` for semantics.
        method : {'auto', 'tree', 'shuffle'}, default 'auto'
            Execution strategy for multi-chunk aggregation.
        **kw
            Additional keyword arguments forwarded to :meth:`agg`.

        Returns
        -------
        LiteFrame
            One row per group with quantile values.  Scalar ``q`` keeps
            original column names; a list of ``q`` produces wide columns
            ``{col}_q{pct}``.
        """
        from maxframe.liteframe.reduction.quantile import _prepare_quantile_agg_func

        raw_func = _prepare_quantile_agg_func(q, interpolation, approx)
        return self.agg(
            raw_func,
            method=method,
            numeric_only=numeric_only,
            **kw,
        )

    def all(self, **kw):
        """Return True if all values in the group are truthy.

        Parameters
        ----------
        **kw
            Additional keyword arguments forwarded to :meth:`agg`.

        Returns
        -------
        LiteFrame
            Lazy LiteFrame with one row per group and the per-group
            boolean "all" result.  Call ``.execute()`` to materialize.
        """
        return self.agg("all", **kw)

    def any(self, **kw):
        """Return True if any value in the group is truthy.

        Parameters
        ----------
        **kw
            Additional keyword arguments forwarded to :meth:`agg`.

        Returns
        -------
        LiteFrame
            Lazy LiteFrame with one row per group and the per-group
            boolean "any" result.  Call ``.execute()`` to materialize.
        """
        return self.agg("any", **kw)

    def _numeric_cols_excluding_keys(self, sort=False):
        """Numeric columns minus group keys and non-selected columns."""
        group_keys = [self._by] if isinstance(self._by, str) else list(self._by)
        numeric_cols = _resolve_numeric_cols(self._liteframe)
        numeric_cols = [c for c in numeric_cols if c not in group_keys]
        selection = getattr(self, "_selection", None)
        if selection is not None:
            sel_set = set(selection)
            numeric_cols = [c for c in numeric_cols if c in sel_set]
        if sort:
            numeric_cols = sorted(numeric_cols)
        return numeric_cols

    def first(self, skipna=True, min_count=-1, **kw):
        """Return the first value per group, skipping NaN by default.

        Parameters
        ----------
        skipna : bool, default True
            If True, skip NaN/null values and return the first non-null
            value in each group.  If False, return the first value
            regardless of null status (may be null itself).
        min_count : int, default -1
            The required number of valid (non-null) values to perform the
            operation.  If fewer than ``min_count`` non-null values are
            present in a group, the result is null.  The default ``-1``
            means no guard (always return the first value).
        **kw
            Additional keyword arguments forwarded to :meth:`agg`.

        Returns
        -------
        LiteFrame
            Lazy LiteFrame with one row per group and the per-group
            first values.  Call ``.execute()`` to materialize.
        """
        return self.agg(AggCall("first", skipna=skipna, min_count=min_count), **kw)

    def last(self, skipna=True, min_count=-1, **kw):
        """Return the last value per group, skipping NaN by default.

        Parameters
        ----------
        skipna : bool, default True
            If True, skip NaN/null values and return the last non-null
            value in each group.  If False, return the last value
            regardless of null status (may be null itself).
        min_count : int, default -1
            The required number of valid (non-null) values to perform the
            operation.  If fewer than ``min_count`` non-null values are
            present in a group, the result is null.  The default ``-1``
            means no guard (always return the last value).
        **kw
            Additional keyword arguments forwarded to :meth:`agg`.

        Returns
        -------
        LiteFrame
            Lazy LiteFrame with one row per group and the per-group
            last values.  Call ``.execute()`` to materialize.
        """
        return self.agg(AggCall("last", skipna=skipna, min_count=min_count), **kw)

    def ohlc(self, **kw):
        """Compute open/high/low/close per group for numeric columns.

        Open = first non-null, high = max, low = min, close = last
        non-null.  Non-numeric columns are silently excluded (matching
        pandas).  NaN values are always skipped.

        Output columns are flattened to ``{col}_open``, ``{col}_high``,
        ``{col}_low``, ``{col}_close`` per numeric column (LiteFrame has
        no MultiIndex).  Numerical suffixes (``_1``, ``_2``, …) are
        appended when a generated name collides with a group key or
        another output column.

        Parameters
        ----------
        **kw
            Additional keyword arguments forwarded to :meth:`agg`.

        Returns
        -------
        LiteFrame
            Lazy LiteFrame with one row per group and four columns per
            numeric column (open/high/low/close).  Call ``.execute()``
            to materialize.
        """
        numeric_cols = self._numeric_cols_excluding_keys()

        funcs = [
            AggCall("first", output_suffix="open", skipna=True),
            AggCall("max", output_suffix="high"),
            AggCall("min", output_suffix="low"),
            AggCall("last", output_suffix="close", skipna=True),
        ]
        func_dict = {col: list(funcs) for col in numeric_cols}
        return self.agg(func_dict, **kw)

    def describe(self, percentiles=None, numeric_only=True, **kw):
        """Generate descriptive statistics per group.

        Parameters
        ----------
        percentiles : list of float, default [0.25, 0.5, 0.75]
            Percentiles to compute, in ``[0, 1]``.  Must be unique and
            exclude 0 and 1 (covered by min/max).  Unsorted input is
            silently sorted ascending (matching pandas).
        numeric_only : bool, default True
            If True, only numeric columns are described.  If False, both
            numeric and datetime columns are included.  For datetime
            columns, ``count``, ``mean``, and ``std`` emit ``NaT``;
            ``min``, percentiles, and ``max`` pass through as timestamps.
        **kw
            Additional keyword arguments forwarded to :meth:`agg`.

        Returns
        -------
        LiteFrame
            Long-format frame with group keys, a ``statistic`` column, and
            one column per data column.
        """
        qs = _validate_percentiles(percentiles)
        dtypes = self._liteframe.dtypes
        data_cols, datetime_cols = _resolve_describe_cols(dtypes, numeric_only)

        # Exclude group key columns from data columns
        by_set = set(self._by if isinstance(self._by, list) else [self._by])
        data_cols = [c for c in data_cols if c not in by_set]
        datetime_cols = [c for c in datetime_cols if c not in by_set]
        # Intersect with column selection when set (pandas: groupby('k')[['a']].describe()
        # describes only the selected columns)
        if self._selection is not None:
            sel_set = set(self._selection)
            data_cols = [c for c in data_cols if c in sel_set]
            datetime_cols = [c for c in datetime_cols if c in sel_set]
        stat_col = resolve_unique_col_name("statistic", list(by_set) + data_cols)

        funcs = _build_describe_agg_funcs(qs, data_cols, datetime_cols)
        if numeric_only or not datetime_cols:
            agg_result = self.agg(funcs, numeric_only=True, **kw)
        else:
            by_list = self._by if isinstance(self._by, list) else [self._by]
            selected = self._liteframe[by_list + data_cols]
            agg_result = selected.groupby(
                self._by,
                sort=self._sort,
                dropna=self._dropna,
                group_keys=self._group_keys,
            ).agg(funcs, numeric_only=False, **kw)
        stat_pairs = _build_stat_pairs(qs)
        col_key_map, stat_cols = _build_col_key_map(
            data_cols,
            stat_pairs,
            datetime_cols,
            list(agg_result.dtypes.index),
            exclude_cols=by_set,
        )
        out_dtypes = _build_out_dtypes(data_cols, datetime_cols, dtypes, stat_col)

        return agg_result.flat_apply(
            _describe_expand,
            args=(data_cols, stat_pairs, datetime_cols, stat_col, col_key_map),
            dtypes=out_dtypes,
            subset=stat_cols,
        )

    def value_counts(
        self,
        subset=None,
        normalize=False,
        sort=True,
        ascending=False,
        dropna=True,
    ):
        """Return value counts within each group.

        Groups by the outer groupby key(s) and the value column(s),
        producing one row per (group_key, value) combination with a count.

        Parameters
        ----------
        subset : str or list of str, default None
            Value column(s) to count.  If None, all non-key columns
            (intersected with the groupby's column selection when set).
        normalize : bool, default False
            If True, return proportions (normalized within each group).
        sort : bool, default True
            Sort by group keys then count/proportion.
        ascending : bool, default False
            Sort order for the count/proportion within each group.
        dropna : bool, default True
            Exclude null-value-key rows.  This composes independently with
            the outer groupby's ``dropna``: the outer controls null
            *group* keys, this controls null *value* keys.

        Returns
        -------
        LiteFrame
            Frame with group keys, value column(s), and a ``"count"`` or
            ``"proportion"`` column.
        """
        group_keys = self._by if isinstance(self._by, list) else [self._by]
        all_cols = list(self._liteframe.columns)
        if subset is None:
            value_cols = [c for c in all_cols if c not in group_keys]
            # Intersect with column selection when set (pandas:
            # groupby('k')[['a']].value_counts() counts by (k, a))
            if self._selection is not None:
                sel_set = set(self._selection)
                value_cols = [c for c in value_cols if c in sel_set]
        else:
            value_cols = _normalize_subset(subset, all_cols)
            # Validate subset against column selection (pandas raises
            # ValueError when subset references unselected columns)
            if self._selection is not None:
                sel_set = set(self._selection)
                missing = [c for c in value_cols if c not in sel_set]
                if missing:
                    raise ValueError(
                        f"Keys {missing} in subset do not exist in the "
                        f"selected columns {list(self._selection)}"
                    )
        by_cols = group_keys + value_cols

        # Validate that subset does not overlap group keys (pandas raises
        # "Keys {overlap} in subset cannot be in the groupby column keys").
        overlap = set(value_cols) & set(group_keys)
        if overlap:
            raise ValueError(
                f"Keys {sorted(overlap)} in subset cannot be in the "
                f"groupby column keys {group_keys}"
            )

        _check_reserved_names(normalize, by_cols)

        lf = self._liteframe
        # Compose dropna independently: the outer groupby's dropna controls
        # null *group* keys, the method-level dropna controls null *value*
        # keys.  A single combined groupby cannot distinguish the two, so
        # when they differ we pre-filter the appropriate columns and then
        # run the combined groupby with dropna=False.
        if self._dropna and not dropna:
            # outer True, method False: drop null group keys, keep null values
            lf = lf.dropna(subset=group_keys)
            combined_dropna = False
        elif not self._dropna and dropna:
            # outer False, method True: keep null group keys, drop null values
            lf = lf.dropna(subset=value_cols)
            combined_dropna = False
        else:
            combined_dropna = dropna

        counts = lf.groupby(
            by_cols,
            sort=self._sort,
            dropna=combined_dropna,
            group_keys=self._group_keys,
        ).size()

        if not normalize:
            result = counts.rename(columns={"size": "count"})
            if sort:
                result = result.sort(
                    group_keys + ["count"],
                    ascending=[True] * len(group_keys) + [ascending],
                )
            return result

        # Normalize within each group: divide by per-group total.
        # Sum the already-counted sizes per group (not re-group the original
        # frame) so null-value rows excluded by dropna are also excluded
        # from the denominator, matching pandas.
        group_totals = counts.groupby(
            group_keys,
            sort=self._sort,
            dropna=False,
            group_keys=self._group_keys,
        ).agg({"size": "sum"})
        total_name = resolve_unique_col_name("_total", list(all_cols) + ["size"])

        # Null group keys cannot be matched by a regular merge (null != null).
        # When self._dropna is True, null group keys cannot appear in counts
        # (pre-filtered or dropped by combined groupby), so merge directly.
        # When self._dropna is False, create per-key (string, is_null_flag)
        # sentinel pairs so null keys match only null keys — the flag column
        # prevents a real value equal to the sentinel string from colliding
        # with genuine nulls.
        if not self._dropna:
            merge_keys = []
            for k in group_keys:
                mk = resolve_unique_col_name(
                    f"_mk_{k}",
                    list(all_cols) + ["size", total_name] + merge_keys,
                )
                counts = counts.assign(**{mk: counts[k].astype(str)})
                counts = counts.fillna({mk: "__null_key__"})
                group_totals = group_totals.assign(**{mk: group_totals[k].astype(str)})
                group_totals = group_totals.fillna({mk: "__null_key__"})
                merge_keys.append(mk)
                # Is-null flag: True for genuine nulls, False for real values.
                fk = resolve_unique_col_name(
                    f"_fk_{k}",
                    list(all_cols) + ["size", total_name] + merge_keys,
                )
                counts = counts.assign(**{fk: counts[k].isna()})
                group_totals = group_totals.assign(**{fk: group_totals[k].isna()})
                merge_keys.append(fk)

            # Select before rename — column selection reverts renames.
            group_totals = group_totals[merge_keys + ["size"]].rename(
                columns={"size": total_name}
            )

            merged = counts.merge(
                group_totals,
                on=merge_keys,
                how="left",
            )
            merged = merged.assign(
                proportion=merged["size"] / merged[total_name],
            )
            result = merged.drop(columns=merge_keys)[
                group_keys + value_cols + ["proportion"]
            ]
        else:
            group_totals = group_totals[group_keys + ["size"]].rename(
                columns={"size": total_name}
            )
            merged = counts.merge(
                group_totals,
                on=group_keys,
                how="left",
            )
            merged = merged.assign(
                proportion=merged["size"] / merged[total_name],
            )
            result = merged[group_keys + value_cols + ["proportion"]]

        if sort:
            result = result.sort(
                group_keys + ["proportion"],
                ascending=[True] * len(group_keys) + [ascending],
            )
        return result

    def shift(self, periods=1, fill_value=None, columns=None, placeback=True):
        """Shift rows within each group by ``periods`` positions.

        Parameters
        ----------
        periods : int, default 1
            Number of positions to shift within each group.
        fill_value : scalar, default None
            Value to fill positions introduced by shifting.
        columns : str, list of str, or None, default None
            Column(s) to shift.  If None, all non-group-key columns are shifted.
        placeback : bool, default True
            If True, restore the output to the input frame's row order and chunk
            layout after the group-key shuffle.  If False, the output stays in
            reducer/hash-partition order.

        Returns
        -------
        LiteFrame
            Shifted LiteFrame with the same shape (row count preserved).
            Output dtypes follow the same rules as standalone ``shift``:
            they depend on the input dtype and ``fill_value`` (columns may
            be widened when a non-null ``fill_value`` is supplied).

        Notes
        -----
        .. warning::
           With ``placeback=True`` (the default), the output is aligned with
           the input frame — row order and chunk layout are restored after the
           group-key shuffle via a ``LiteFramePlaceBack`` step.  With
           ``placeback=False`` the output stays in reducer/hash-partition order
           and is **not** aligned with the input (unlike pandas, which preserves
           the index).  Row count is always preserved.

           Null-key rows are preserved with a null result (matching pandas
           ``groupby.shift()`` under the default ``dropna=True``); they are not
           grouped together.

        """
        from maxframe.liteframe.operators.window import LiteFrameWindowTransform

        return LiteFrameWindowTransform.for_positional(
            "shift",
            periods=periods,
            fill_value=fill_value,
            columns=columns,
            groupby_params=self.groupby_params,
            placeback=placeback,
        )(self._liteframe)

    def diff(self, periods=1, columns=None, placeback=True):
        """Compute diff within each group.

        Parameters
        ----------
        periods : int, default 1
            Number of positions to diff within each group.
        columns : str, list of str, or None, default None
            Column(s) to diff.  If None, all non-group-key columns are diffed.
        placeback : bool, default True
            If True, restore the output to the input frame's row order and chunk
            layout after the group-key shuffle.  If False, the output stays in
            reducer/hash-partition order.

        Returns
        -------
        LiteFrame
            Diffed LiteFrame with same shape (row count preserved).

        Notes
        -----
        .. warning::
           With ``placeback=True`` (the default), the output is aligned with
           the input frame — row order and chunk layout are restored after the
           group-key shuffle via a ``LiteFramePlaceBack`` step.  With
           ``placeback=False`` the output stays in reducer/hash-partition order
           and is **not** aligned with the input (unlike pandas, which preserves
           the index).  Row count is always preserved.

           Null-key rows are preserved with a null result (matching pandas
           ``groupby.shift()`` under the default ``dropna=True``); they are not
           grouped together.


        Raises
        ------
        TypeError
            If any transformed column is non-numeric.
        """
        from maxframe.liteframe.operators.window import LiteFrameWindowTransform

        return LiteFrameWindowTransform.for_positional(
            "diff",
            periods=periods,
            columns=columns,
            groupby_params=self.groupby_params,
            placeback=placeback,
        )(self._liteframe)

    def pct_change(self, periods=1, columns=None, placeback=True):
        """Compute pct_change within each group.

        Parameters
        ----------
        periods : int, default 1
            Number of positions for pct_change within each group.
        columns : str, list of str, or None, default None
            Column(s) for pct_change.  If None, all non-group-key columns
            are transformed.
        placeback : bool, default True
            If True, restore the output to the input frame's row order and chunk
            layout after the group-key shuffle.  If False, the output stays in
            reducer/hash-partition order.

        Returns
        -------
        LiteFrame
            Pct-change LiteFrame with same shape (row count preserved).

        Notes
        -----
        .. warning::
           With ``placeback=True`` (the default), the output is aligned with
           the input frame — row order and chunk layout are restored after the
           group-key shuffle via a ``LiteFramePlaceBack`` step.  With
           ``placeback=False`` the output stays in reducer/hash-partition order
           and is **not** aligned with the input (unlike pandas, which preserves
           the index).  Row count is always preserved.

           Null-key rows are preserved with a null result (matching pandas
           ``groupby.shift()`` under the default ``dropna=True``); they are not
           grouped together.


        Raises
        ------
        TypeError
            If any transformed column is non-numeric.
        """
        from maxframe.liteframe.operators.window import LiteFrameWindowTransform

        return LiteFrameWindowTransform.for_positional(
            "pct_change",
            periods=periods,
            columns=columns,
            groupby_params=self.groupby_params,
            placeback=placeback,
        )(self._liteframe)

    def ffill(self, limit=None, columns=None, placeback=True):
        """Forward-fill within each group.

        Propagate the last non-null value forward across consecutive nulls,
        scoped per group.  ``limit`` caps consecutive fills per run
        (``None`` = unlimited); ``limit <= 0`` raises ``ValueError``.

        Parameters
        ----------
        limit : int or None, default None
            Maximum number of consecutive nulls to fill in each run.
        columns : str, list of str, or None, default None
            Column(s) to fill.  If None, all non-group-key columns are filled.
        placeback : bool, default True
            If True, restore the output to the input frame's row order and chunk
            layout after the group-key shuffle.  If False, the output stays in
            reducer/hash-partition order.

        Returns
        -------
        LiteFrame
            Forward-filled LiteFrame with the same shape (row count
            preserved).  Output dtypes equal the input dtypes.

        Notes
        -----
        .. warning::
           With ``placeback=True`` (the default), the output is aligned with
           the input frame — row order and chunk layout are restored after the
           group-key shuffle via a ``LiteFramePlaceBack`` step.  With
           ``placeback=False`` the output stays in reducer/hash-partition order
           and is **not** aligned with the input (unlike pandas).  Row count is
           always preserved.  Intra-group row order *is* preserved (the hash
           shuffle is stable w.r.t. intra-group order), so the per-group fill
           direction follows the original input order.

           Null-key rows are preserved with a null result (matching pandas
           ``groupby.ffill()`` under the default ``dropna=True``); they are not
           grouped together.
        """
        from maxframe.liteframe.operators.window import LiteFrameWindowTransform

        return LiteFrameWindowTransform.for_fill(
            "ffill",
            limit=limit,
            columns=columns,
            groupby_params=self.groupby_params,
            placeback=placeback,
        )(self._liteframe)

    def bfill(self, limit=None, columns=None, placeback=True):
        """Backward-fill within each group.

        Propagate the next non-null value backward across consecutive nulls,
        scoped per group.  ``limit`` caps consecutive fills per run
        (``None`` = unlimited); ``limit <= 0`` raises ``ValueError``.

        Parameters
        ----------
        limit : int or None, default None
            Maximum number of consecutive nulls to fill in each run.
        columns : str, list of str, or None, default None
            Column(s) to fill.  If None, all non-group-key columns are filled.
        placeback : bool, default True
            If True, restore the output to the input frame's row order and chunk
            layout after the group-key shuffle.  If False, the output stays in
            reducer/hash-partition order.

        Returns
        -------
        LiteFrame
            Backward-filled LiteFrame with the same shape (row count
            preserved).  Output dtypes equal the input dtypes.

        Notes
        -----
        .. warning::
           Same row-order behavior as ``ffill``: with ``placeback=True``
           (the default) the output is input-aligned; with ``placeback=False``
           it stays in reducer/hash-partition order.  Intra-group order is
           always preserved.
        """
        from maxframe.liteframe.operators.window import LiteFrameWindowTransform

        return LiteFrameWindowTransform.for_fill(
            "bfill",
            limit=limit,
            columns=columns,
            groupby_params=self.groupby_params,
            placeback=placeback,
        )(self._liteframe)

    # -- Cumulative transforms (cum*) --
    #
    # All cum* methods delegate to ``for_cumulative`` which creates an
    # ``expanding_agg`` operator with ``remask_input_nan=True`` (cum* masks
    # NaN positions) and ``min_periods=1``.

    def _cumulative(self, func_name, skipna=True, columns=None, placeback=True):
        """Shared helper for cumsum/cummax/cummin/cumprod."""
        from maxframe.liteframe.operators.window import LiteFrameWindowTransform

        return LiteFrameWindowTransform.for_cumulative(
            func_name,
            skipna=skipna,
            columns=columns,
            groupby_params=self.groupby_params,
            placeback=placeback,
        )(self._liteframe)

    def cumsum(self, skipna=True, columns=None, placeback=True):
        """Cumulative sum within each group.

        Parameters
        ----------
        skipna : bool, default True
            If True, skip null values in the cumulative computation.
        columns : str, list of str, or None, default None
            Column(s) to accumulate.  If None, all non-group-key columns.
        placeback : bool, default True
            If True, restore the output to the input frame's row order and chunk
            layout after the group-key shuffle.  If False, the output stays in
            reducer/hash-partition order.

        Returns
        -------
        LiteFrame
            Lazy LiteFrame with the same row count as the input, containing
            the per-group cumulative sum.  Call ``.execute()`` to materialize.
        """
        return self._cumulative("cumsum", skipna, columns, placeback=placeback)

    def cummax(self, skipna=True, columns=None, placeback=True):
        """Cumulative maximum within each group.

        Parameters
        ----------
        skipna : bool, default True
            If True, skip null values in the cumulative computation.
        columns : str, list of str, or None, default None
            Column(s) to accumulate.  If None, all non-group-key columns.
        placeback : bool, default True
            If True, restore the output to the input frame's row order and chunk
            layout after the group-key shuffle.  If False, the output stays in
            reducer/hash-partition order.

        Returns
        -------
        LiteFrame
            Lazy LiteFrame with the same row count as the input, containing
            the per-group cumulative maximum.  Call ``.execute()`` to
            materialize.
        """
        return self._cumulative("cummax", skipna, columns, placeback=placeback)

    def cummin(self, skipna=True, columns=None, placeback=True):
        """Cumulative minimum within each group.

        Parameters
        ----------
        skipna : bool, default True
            If True, skip null values in the cumulative computation.
        columns : str, list of str, or None, default None
            Column(s) to accumulate.  If None, all non-group-key columns.
        placeback : bool, default True
            If True, restore the output to the input frame's row order and chunk
            layout after the group-key shuffle.  If False, the output stays in
            reducer/hash-partition order.

        Returns
        -------
        LiteFrame
            Lazy LiteFrame with the same row count as the input, containing
            the per-group cumulative minimum.  Call ``.execute()`` to
            materialize.
        """
        return self._cumulative("cummin", skipna, columns, placeback=placeback)

    def cumprod(self, skipna=True, columns=None, placeback=True):
        """Cumulative product within each group.

        Parameters
        ----------
        skipna : bool, default True
            If True, skip null values in the cumulative computation.
        columns : str, list of str, or None, default None
            Column(s) to accumulate.  If None, all non-group-key columns.
        placeback : bool, default True
            If True, restore the output to the input frame's row order and chunk
            layout after the group-key shuffle.  If False, the output stays in
            reducer/hash-partition order.

        Returns
        -------
        LiteFrame
            Lazy LiteFrame with the same row count as the input, containing
            the per-group cumulative product.  Call ``.execute()`` to
            materialize.
        """
        return self._cumulative("cumprod", skipna, columns, placeback=placeback)

    def cumcount(self, ascending=True, placeback=True):
        """Cumulative count (0-indexed row number) within each group.

        Unlike the other cum* methods, ``cumcount`` counts rows (not values),
        is NaN-agnostic, and has no ``columns`` or ``skipna`` parameters.
        ``ascending=False`` reverses the count (counts from the end).

        Parameters
        ----------
        ascending : bool, default True
            If True, number rows from 0 upward within each group.  If False,
            count from the last row backward.
        placeback : bool, default True
            If True, restore the output to the input frame's row order and chunk
            layout after the group-key shuffle.  If False, the output stays in
            reducer/hash-partition order.

        Returns
        -------
        LiteFrame
            Row count per group, 0-indexed.
        """
        from maxframe.liteframe.operators.window import LiteFrameWindowTransform

        op = LiteFrameWindowTransform.for_cumulative(
            "cumcount",
            columns=None,
            groupby_params=self.groupby_params,
            placeback=placeback,
        )
        op.extra_params["ascending"] = ascending
        return op(self._liteframe)

    def rank(
        self,
        method="average",
        ascending=True,
        na_option="keep",
        pct=False,
        placeback=True,
    ):
        """Compute numerical data ranks (1 through n) within each group.

        Provides the rank of values within each group, averaged over tied
        groups.  Mirrors ``pandas.groupby.DataFrameGroupBy.rank``.

        Parameters
        ----------
        method : {{'average', 'min', 'max', 'first', 'dense'}}, default 'average'
            How to rank the group of records that have the same value (ties).
        ascending : bool, default True
            Whether the elements should be ranked in ascending order.
        na_option : {{'keep', 'top', 'bottom'}}, default 'keep'
            How to rank NaN values.  Only ``'keep'`` is supported in v1.
        pct : bool, default False
            Whether to display the returned rankings in percentile form.
        placeback : bool, default True
            If True, restore the output to the input frame's row order and chunk
            layout after the group-key shuffle.  If False, the output stays in
            reducer/hash-partition order.

        Returns
        -------
        LiteFrame
            Same shape as input (row count preserved).  Ranked columns are
            Float64; group-key columns are excluded (unless explicitly
            selected via ``groupby(...)[...]``).

        Notes
        -----
        .. warning::
           With ``placeback=True`` (the default), the output is aligned with
           the input frame — row order and chunk layout are restored after the
           group-key shuffle.  With ``placeback=False`` the output stays in
           reducer/hash-partition order and is **not** aligned with the input
           (unlike pandas, which preserves the index).  Row count is always
           preserved.

           Only ``na_option='keep'`` is supported in v1; ``'top'``/``'bottom'``
           raise ``NotImplementedError``.
        """
        from maxframe.liteframe.operators.window import LiteFrameWindowTransform

        return LiteFrameWindowTransform.for_rank(
            method=method,
            ascending=ascending,
            na_option=na_option,
            pct=pct,
            groupby_params=self.groupby_params,
            placeback=placeback,
        )(self._liteframe)

    def expanding(
        self,
        min_periods=1,
        columns=None,
        order_by=None,
        order_by_ascending=True,
        window_shift=0,
        reverse=False,
        placeback=True,
    ):
        """Provide expanding window calculations within each group.

        Parameters
        ----------
        min_periods : int, default 1
            Minimum observations in the window to produce a value.
        columns : str, list of str, or None, default None
            Column(s) to aggregate. If None, all non-group-key columns.
        placeback : bool, default True
            If True, restore the output to the input frame's row order and chunk
            layout after the group-key shuffle.  If False, the output stays in
            reducer/hash-partition order.

        Returns
        -------
        LiteFrameExpanding
            Intermediate object; call an aggregation method to produce
            a LiteFrame.

        Notes
        -----
        .. warning::
           With ``placeback=True`` (the default), the output is aligned with
           the input frame — row order and chunk layout are restored after the
           group-key shuffle.  With ``placeback=False`` the output stays in
           reducer/hash-partition order and is **not** aligned with the input.
           Row count is always preserved.
        """
        from maxframe.liteframe.window.expanding import LiteFrameExpanding

        return LiteFrameExpanding(
            self._liteframe,
            min_periods=min_periods,
            columns=columns,
            groupby_params=self.groupby_params,
            order_by=order_by,
            order_by_ascending=order_by_ascending,
            window_shift=window_shift,
            reverse=reverse,
            placeback=placeback,
        )

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
        """Provide exponentially weighted (EW) calculations within each group.

        Parameters
        ----------
        com : float, optional
            Center of mass: ``alpha = 1 / (1 + com)``.
        span : float, optional
            Span: ``alpha = 2 / (span + 1)``.
        halflife : float, optional
            Half-life: ``alpha = 1 - exp(ln(0.5) / halflife)``.
        alpha : float, optional
            Smoothing factor directly: ``0 < alpha <= 1``.
        min_periods : int, default 0
            Minimum observations in the window to produce a value.
        adjust : bool, default True
            Adjusted (weighted average) vs recursive form.
        ignore_na : bool, default False
            Ignore NaN positions (relative) vs absolute positioning.
        bias : bool, default False
            Biased vs unbiased variance (std/var only).
        columns : str, list of str, or None, default None
            Column(s) to aggregate. If None, all non-group-key columns.
        order_by : str, list of str, or None, default None
            Column(s) that define the order in which the window expands
            within each group (equivalent to ``ORDER BY`` inside a
            MaxCompute ``OVER()`` clause).
        order_by_ascending : bool or list of bool, default True
            Sort direction for each *order_by* column.
        window_shift : int, default 0
            Offset the EWM leading edge by N rows.
        reverse : bool, default False
            Flip the EWM to run from the last row backward within each group.
        placeback : bool, default True
            If True, restore output to input frame's row order after shuffle.

        Returns
        -------
        LiteFrameEWM
            Intermediate object; call an aggregation method to produce
            a LiteFrame.

        Notes
        -----
        See :meth:`LiteFrame.ewm <maxframe.liteframe.LiteFrame.ewm>` for
        the mathematical details on ``adjust``, ``ignore_na``, and ``bias``.

        .. warning::
           With ``placeback=True`` (the default), the output is aligned with
           the input frame — row order and chunk layout are restored after the
           group-key shuffle.  With ``placeback=False`` the output stays in
           reducer/hash-partition order and is **not** aligned with the input.
           Row count is always preserved.
        """
        from maxframe.liteframe.window.ewm import LiteFrameEWM, _resolve_com

        resolved_com = _resolve_com(com=com, span=span, halflife=halflife, alpha=alpha)
        return LiteFrameEWM(
            self._liteframe,
            com=resolved_com,
            min_periods=min_periods,
            adjust=adjust,
            ignore_na=ignore_na,
            bias=bias,
            columns=columns,
            groupby_params=self.groupby_params,
            order_by=order_by,
            order_by_ascending=order_by_ascending,
            window_shift=window_shift,
            reverse=reverse,
            placeback=placeback,
        )

    def rolling(
        self,
        window,
        min_periods=None,
        center=False,
        columns=None,
        order_by=None,
        order_by_ascending=True,
        window_shift=0,
        placeback=True,
    ):
        """Provide rolling-window calculations within each group.

        The data is partitioned by the group key (hash shuffle) so that every
        group resides in a single partition; the rolling aggregation is then
        computed per-group. Only integer window sizes are supported.

        Parameters
        ----------
        window : int
            Size of the moving window (number of rows). Must be positive.
        min_periods : int or None, default None
            Minimum number of observations in a window required to produce a
            value (otherwise null). Defaults to ``window`` (matching pandas).
            For ``count`` the value is the number of non-null observations, but
            ``min_periods`` still gates on the number of window rows
            (observations), not the non-null count — consistent with pandas.
        center : bool, default False
            Center the window on the current row.
        columns : str, list of str, or None, default None
            Column(s) to aggregate. If None, all non-group-key columns are
            aggregated. Non-selected columns are passed through unchanged.
        placeback : bool, default True
            If True, restore the output to the input frame's row order and chunk
            layout after the group-key shuffle.  If False, the output stays in
            reducer/hash-partition order.

        Returns
        -------
        LiteFrameRolling
            Intermediate object; call an aggregation method (e.g. ``.sum()``)
            to produce a LiteFrame.

        Notes
        -----
        .. warning::
           With ``placeback=True`` (the default), the output is aligned with
           the input frame — row order and chunk layout are restored after the
           group-key shuffle via a ``LiteFramePlaceBack`` step.  With
           ``placeback=False`` the output stays in reducer/hash-partition order
           and is **not** aligned with the input (unlike pandas, which preserves
           the index).  Row count is always preserved.

           Null-key rows are preserved with a null result (matching the other
           grouped transforms and the row-count-preserving transform contract);
           they are not grouped together. (This diverges from pandas
           grouped-rolling, which drops null-key rows under ``dropna=True``.)

        Raises
        ------
        NotImplementedError
            If ``window`` is not an integer.
        ValueError
            If ``window`` is not positive or ``min_periods`` is invalid.
        """
        from maxframe.liteframe.window.rolling import LiteFrameRolling

        return LiteFrameRolling(
            self._liteframe,
            window,
            min_periods=min_periods,
            center=center,
            columns=columns,
            groupby_params=self.groupby_params,
            order_by=order_by,
            order_by_ascending=order_by_ascending,
            window_shift=window_shift,
            placeback=placeback,
        )

    def sample(
        self,
        n=None,
        frac=None,
        replace=False,
        weights=None,
        random_state=None,
        exact=False,
    ):
        """Return a random sample of rows from each group.

        After groupby shuffle, each group's data is fully collected on a
        single node, so sampling is always exact regardless of the
        ``exact`` parameter. The ``exact`` parameter is accepted solely
        for API consistency with :meth:`LiteFrame.sample`.

        Parameters
        ----------
        n : int, optional
            Number of rows to return per group. Cannot be used with
            ``frac``.
        frac : float, optional
            Fraction of rows to return per group. Cannot be used with
            ``n``.
        replace : bool, default False
            Allow sampling the same row more than once.
        weights : str or LiteFrame, optional
            Column name (str) for column-based weights, or a single-column
            LiteFrame aligned with the data.
        random_state : int, RandomState, or None
            Seed for reproducibility.
        exact : bool, default False
            Accepted for API consistency with :meth:`LiteFrame.sample`.
            Has no effect — groupby sampling is always exact.

        Returns
        -------
        LiteFrame
            Sampled frame with one or more rows per group.
        """
        from maxframe.liteframe.indexing.sample import _groupby_sample

        return _groupby_sample(self, n, frac, replace, weights, random_state, exact)

    def filter(self, func, dropna=True, *args, **kwargs):
        """Filter groups by a predicate function.

        *func* is applied to each group (as a pandas DataFrame) and must
        return a scalar ``bool``.  Groups where *func* returns ``True`` are
        kept; groups where it returns ``False`` are dropped.  Any other
        return type raises ``TypeError``, matching pandas semantics.

        Parameters
        ----------
        func : callable
            Function taking a group DataFrame and returning a scalar bool.
        dropna : bool, default True
            Drop groups whose predicate returns ``False``.  If ``False``,
            those groups are kept with all column values set to null,
            matching pandas.
        *args, **kwargs
            Additional arguments forwarded to *func*.

        Returns
        -------
        LiteFrame
            LiteFrame with rows from kept groups, preserving the original
            column order.  Row order within each group is preserved, but
            the global order of groups follows ``sort`` (consistent with
            :meth:`apply`).

        Examples
        --------
        >>> import maxframe.liteframe as mlf
        >>> lf = mlf.LiteFrame({'key': ['a', 'a', 'b', 'b'],
        ...                     'val': [1, 2, 3, 4]})
        >>> lf.groupby('key').filter(lambda g: g['val'].sum() > 3).execute()
          key  val
        2   b    3
        3   b    4
        """

        # Output dtypes match UDF input dtypes (wrapper returns group unchanged
        # or row-filtered).  Pass explicitly to skip inference on user func.
        by = self._by if isinstance(self._by, (list, tuple)) else [self._by]
        # Resolve a marker name that cannot collide with a real source or
        # grouping column, so dropna=False never overwrites user data.
        marker_col = resolve_unique_col_name(
            _FILTER_KEEP_COL, set(self._liteframe.columns) | set(by)
        )

        udf_dtypes = (
            self._liteframe.dtypes[self._selection]
            if self._selection is not None
            else self._liteframe.dtypes.drop(labels=by, errors="ignore")
        ).copy()
        if not dropna:
            udf_dtypes[marker_col] = bool

        # filter() output is independent of group_keys (pandas always returns
        # the original frame's columns).  Force group_keys=True for the
        # internal apply so key columns are present, then select the expected
        # output columns: all original columns when there is no selection,
        # or only the selected columns otherwise (pandas: selected groupby
        # filter returns just the selected columns, not the key).
        internal_gb = LiteFrameGroupBy(
            self._liteframe, self._by, self._sort, self._dropna, True
        )
        internal_gb._selection = self._selection
        result = internal_gb.apply(
            _filter_group_wrapper,
            args=(func, args, kwargs, dropna, marker_col),
            dtypes=udf_dtypes,
            input_frame_type="pandas",
        )
        if not dropna:
            result = _nullify_rejected_groups(result, marker_col)
        # Exclude the virtual range index: apply drops it (row count/order can
        # change), so selecting it by name would raise ColumnNotFoundError.
        range_cols = set(
            (self._liteframe._data.frame_metadata.range_columns or {})
            if self._liteframe._data.frame_metadata
            else {}
        )
        source_cols = (
            list(self._selection)
            if self._selection is not None
            else list(self._liteframe.columns)
        )
        expected = [c for c in source_cols if c not in range_cols]
        return result.select(expected)
