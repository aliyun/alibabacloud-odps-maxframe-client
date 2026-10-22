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
        from maxframe.liteframe.operators.reduction import LiteFrameAgg

        op = LiteFrameAgg(
            raw_func=func,
            raw_func_kw=kwargs if kwargs else None,
            groupby_params=self.groupby_params,
            method=method,
            numeric_only=numeric_only,
            input_frame_type=input_frame_type,
        )
        return op(self._liteframe)

    def sum(self, **kw):
        """Compute the sum of each group.

        Parameters
        ----------
        **kw
            Additional keyword arguments forwarded to :meth:`agg`.

        Returns
        -------
        LiteFrame
            Lazy LiteFrame with one row per group and the per-group sum.
            Call ``.execute()`` to materialize.
        """
        return self.agg("sum", **kw)

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

    def min(self, **kw):
        """Compute the minimum of each group.

        Parameters
        ----------
        **kw
            Additional keyword arguments forwarded to :meth:`agg`.

        Returns
        -------
        LiteFrame
            Lazy LiteFrame with one row per group and the per-group minimum.
            Call ``.execute()`` to materialize.
        """
        return self.agg("min", **kw)

    def max(self, **kw):
        """Compute the maximum of each group.

        Parameters
        ----------
        **kw
            Additional keyword arguments forwarded to :meth:`agg`.

        Returns
        -------
        LiteFrame
            Lazy LiteFrame with one row per group and the per-group maximum.
            Call ``.execute()`` to materialize.
        """
        return self.agg("max", **kw)

    def prod(self, **kw):
        """Compute the product of each group.

        Parameters
        ----------
        **kw
            Additional keyword arguments forwarded to :meth:`agg`.

        Returns
        -------
        LiteFrame
            Lazy LiteFrame with one row per group and the per-group product.
            Call ``.execute()`` to materialize.
        """
        return self.agg("prod", **kw)

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

    def var(self, **kw):
        """Compute the variance of each group.

        Parameters
        ----------
        **kw
            Additional keyword arguments forwarded to :meth:`agg`.

        Returns
        -------
        LiteFrame
            Lazy LiteFrame with one row per group and the per-group
            variance.  Call ``.execute()`` to materialize.
        """
        return self.agg("var", **kw)

    def std(self, **kw):
        """Compute the standard deviation of each group.

        Parameters
        ----------
        **kw
            Additional keyword arguments forwarded to :meth:`agg`.

        Returns
        -------
        LiteFrame
            Lazy LiteFrame with one row per group and the per-group
            standard deviation.  Call ``.execute()`` to materialize.
        """
        return self.agg("std", **kw)

    def sem(self, **kw):
        """Compute the standard error of the mean of each group.

        Parameters
        ----------
        **kw
            Additional keyword arguments forwarded to :meth:`agg`.

        Returns
        -------
        LiteFrame
            Lazy LiteFrame with one row per group and the per-group
            standard error of the mean.  Call ``.execute()`` to materialize.
        """
        return self.agg("sem", **kw)

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
        from maxframe.liteframe.reduction.compile import AggCall

        return self.agg(AggCall("nunique", dropna=dropna), **kw)

    def median(self, **kw):
        """Compute the median of each group.

        Parameters
        ----------
        **kw
            Additional keyword arguments forwarded to :meth:`agg`.

        Returns
        -------
        LiteFrame
            Lazy LiteFrame with one row per group and the per-group median.
            Call ``.execute()`` to materialize.
        """
        return self.agg("median", **kw)

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

    def shift(self, periods=1, fill_value=None, columns=None):
        """Shift rows within each group by ``periods`` positions.

        Parameters
        ----------
        periods : int, default 1
            Number of positions to shift within each group.
        fill_value : scalar, default None
            Value to fill positions introduced by shifting.
        columns : str, list of str, or None, default None
            Column(s) to shift.  If None, all non-group-key columns are shifted.

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
           Grouped transforms hash-shuffle rows by group key, so the output
           row order follows reducer/hash-partition order and is **not** aligned
           with the input frame (unlike pandas, which preserves the index). Row
           count is preserved. Order restoration via a place-back operator
           (``LiteFrameIlocSetItem``) is planned; see
           ``docs/designs/liteframe/operators/window.md``.

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
        )(self._liteframe)

    def diff(self, periods=1, columns=None):
        """Compute diff within each group.

        Parameters
        ----------
        periods : int, default 1
            Number of positions to diff within each group.
        columns : str, list of str, or None, default None
            Column(s) to diff.  If None, all non-group-key columns are diffed.

        Returns
        -------
        LiteFrame
            Diffed LiteFrame with same shape (row count preserved).

        Notes
        -----
        .. warning::
           Grouped transforms hash-shuffle rows by group key, so the output
           row order follows reducer/hash-partition order and is **not** aligned
           with the input frame (unlike pandas, which preserves the index). Row
           count is preserved. Order restoration via a place-back operator
           (``LiteFrameIlocSetItem``) is planned; see
           ``docs/designs/liteframe/operators/window.md``.

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
        )(self._liteframe)

    def pct_change(self, periods=1, columns=None):
        """Compute pct_change within each group.

        Parameters
        ----------
        periods : int, default 1
            Number of positions for pct_change within each group.
        columns : str, list of str, or None, default None
            Column(s) for pct_change.  If None, all non-group-key columns
            are transformed.

        Returns
        -------
        LiteFrame
            Pct-change LiteFrame with same shape (row count preserved).

        Notes
        -----
        .. warning::
           Grouped transforms hash-shuffle rows by group key, so the output
           row order follows reducer/hash-partition order and is **not** aligned
           with the input frame (unlike pandas, which preserves the index). Row
           count is preserved. Order restoration via a place-back operator
           (``LiteFrameIlocSetItem``) is planned; see
           ``docs/designs/liteframe/operators/window.md``.

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
        )(self._liteframe)

    def ffill(self, limit=None, columns=None):
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

        Returns
        -------
        LiteFrame
            Forward-filled LiteFrame with the same shape (row count
            preserved).  Output dtypes equal the input dtypes.

        Notes
        -----
        .. warning::
           Grouped transforms hash-shuffle rows by group key, so the output
           row order follows reducer/hash-partition order and is **not**
           aligned with the input frame (unlike pandas).  Row count is
           preserved.  Intra-group row order *is* preserved (the hash shuffle
           is stable w.r.t. intra-group order), so the per-group fill direction
           follows the original input order.  See
           ``docs/designs/liteframe/operators/window.md``.

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
        )(self._liteframe)

    def bfill(self, limit=None, columns=None):
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

        Returns
        -------
        LiteFrame
            Backward-filled LiteFrame with the same shape (row count
            preserved).  Output dtypes equal the input dtypes.

        Notes
        -----
        .. warning::
           Same row-order limitation as ``ffill``: global output is in
           reducer/hash-partition order, but intra-group order is preserved.
        """
        from maxframe.liteframe.operators.window import LiteFrameWindowTransform

        return LiteFrameWindowTransform.for_fill(
            "bfill",
            limit=limit,
            columns=columns,
            groupby_params=self.groupby_params,
        )(self._liteframe)

    # -- Cumulative transforms (cum*) --
    #
    # All cum* methods delegate to ``for_cumulative`` which creates an
    # ``expanding_agg`` operator with ``remask_input_nan=True`` (cum* masks
    # NaN positions) and ``min_periods=1``.

    def _cumulative(self, func_name, skipna=True, columns=None):
        """Shared helper for cumsum/cummax/cummin/cumprod."""
        from maxframe.liteframe.operators.window import LiteFrameWindowTransform

        return LiteFrameWindowTransform.for_cumulative(
            func_name,
            skipna=skipna,
            columns=columns,
            groupby_params=self.groupby_params,
        )(self._liteframe)

    def cumsum(self, skipna=True, columns=None):
        """Cumulative sum within each group.

        Parameters
        ----------
        skipna : bool, default True
            If True, skip null values in the cumulative computation.
        columns : str, list of str, or None, default None
            Column(s) to accumulate.  If None, all non-group-key columns.

        Returns
        -------
        LiteFrame
            Lazy LiteFrame with the same row count as the input, containing
            the per-group cumulative sum.  Call ``.execute()`` to materialize.
        """
        return self._cumulative("cumsum", skipna, columns)

    def cummax(self, skipna=True, columns=None):
        """Cumulative maximum within each group.

        Parameters
        ----------
        skipna : bool, default True
            If True, skip null values in the cumulative computation.
        columns : str, list of str, or None, default None
            Column(s) to accumulate.  If None, all non-group-key columns.

        Returns
        -------
        LiteFrame
            Lazy LiteFrame with the same row count as the input, containing
            the per-group cumulative maximum.  Call ``.execute()`` to
            materialize.
        """
        return self._cumulative("cummax", skipna, columns)

    def cummin(self, skipna=True, columns=None):
        """Cumulative minimum within each group.

        Parameters
        ----------
        skipna : bool, default True
            If True, skip null values in the cumulative computation.
        columns : str, list of str, or None, default None
            Column(s) to accumulate.  If None, all non-group-key columns.

        Returns
        -------
        LiteFrame
            Lazy LiteFrame with the same row count as the input, containing
            the per-group cumulative minimum.  Call ``.execute()`` to
            materialize.
        """
        return self._cumulative("cummin", skipna, columns)

    def cumprod(self, skipna=True, columns=None):
        """Cumulative product within each group.

        Parameters
        ----------
        skipna : bool, default True
            If True, skip null values in the cumulative computation.
        columns : str, list of str, or None, default None
            Column(s) to accumulate.  If None, all non-group-key columns.

        Returns
        -------
        LiteFrame
            Lazy LiteFrame with the same row count as the input, containing
            the per-group cumulative product.  Call ``.execute()`` to
            materialize.
        """
        return self._cumulative("cumprod", skipna, columns)

    def cumcount(self, ascending=True):
        """Cumulative count (0-indexed row number) within each group.

        Unlike the other cum* methods, ``cumcount`` counts rows (not values),
        is NaN-agnostic, and has no ``columns`` or ``skipna`` parameters.
        ``ascending=False`` reverses the count (counts from the end).

        Parameters
        ----------
        ascending : bool, default True
            If True, number rows from 0 upward within each group.  If False,
            count from the last row backward.

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
        )
        op.extra_params["ascending"] = ascending
        return op(self._liteframe)

    def expanding(
        self,
        min_periods=1,
        columns=None,
        order_by=None,
        order_by_ascending=True,
        window_shift=0,
        reverse=False,
    ):
        """Provide expanding window calculations within each group.

        Parameters
        ----------
        min_periods : int, default 1
            Minimum observations in the window to produce a value.
        columns : str, list of str, or None, default None
            Column(s) to aggregate. If None, all non-group-key columns.

        Returns
        -------
        LiteFrameExpanding
            Intermediate object; call an aggregation method to produce
            a LiteFrame.

        Notes
        -----
        .. warning::
           Grouped transforms hash-shuffle rows by group key, so the output
           row order follows reducer/hash-partition order and is **not** aligned
           with the input frame. Row count is preserved.
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

        Returns
        -------
        LiteFrameRolling
            Intermediate object; call an aggregation method (e.g. ``.sum()``)
            to produce a LiteFrame.

        Notes
        -----
        .. warning::
           Grouped transforms hash-shuffle rows by group key, so the output
           row order follows reducer/hash-partition order and is **not** aligned
           with the input frame (unlike pandas, which preserves the index).
           Order restoration via a place-back operator (``LiteFrameIlocSetItem``)
           is planned; see ``docs/designs/liteframe/operators/window.md``.

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
        )
