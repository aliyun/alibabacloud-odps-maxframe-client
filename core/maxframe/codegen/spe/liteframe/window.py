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

from typing import List, Optional

import numpy as np
import pandas as pd

from maxframe.codegen.spe.core import (
    SPECodeContext,
    SPEOperatorAdapter,
    register_op_adapter,
)
from maxframe.liteframe.operators.window import LiteFrameWindowTransform


def _get_target_cols(df, columns, groupby_params, hidden_columns, range_columns):
    """Columns to transform.

    Range columns are always excluded: they stay virtual for standalone
    transforms (the window operator preserves them in ``frame_metadata``),
    matching the DPE executor which operates on the physical frame only.
    """
    hidden = set(hidden_columns) if hidden_columns else set()
    range_cols = set(range_columns) if range_columns else set()
    excluded = hidden | range_cols
    if columns is not None:
        return [c for c in columns if c not in excluded]
    if groupby_params is not None:
        by = groupby_params["by"]
        by_cols = [by] if isinstance(by, str) else list(by)
        excluded = set(by_cols) | excluded
        return [c for c in df.columns if c not in excluded]
    return [c for c in df.columns if c not in excluded]


def _make_unique_col_name(base, columns):
    """Return a temp column name derived from *base* that does not collide
    with any existing label in *columns* (mirrors DPE's _make_unique_col_name).

    Used for positional-ordinal place-back in grouped transforms; a fixed
    name like ``__mf_pos__`` would collide with a legitimate user column of
    the same name and silently overwrite it."""
    if base not in columns:
        return base
    i = 1
    while f"{base}_{i}" in columns:
        i += 1
    return f"{base}_{i}"


def _project_grouped_output(result, groupby_params, hidden_columns):
    """Project a grouped window result to the planned output schema.

    Mirrors the DPE executor's ``_maybe_project_selection``: without an
    explicit selection the grouping key columns are dropped from the output;
    with a ``gb[...]`` selection the selected columns (keys included) are
    kept; hidden columns always survive.

    The grouped result is built from a copy of the input frame, so without
    this the grouping key columns leak into the output, drifting from the
    declared schema (which excludes them) and from pandas (window methods
    never emit the keys -- ``group_keys`` has no effect on them).
    """
    if not groupby_params:
        return result
    hidden = set(hidden_columns) if hidden_columns else set()
    sel = groupby_params.get("selection")
    if sel is not None:
        keep = [c for c in sel if c in result.columns]
    else:
        by = groupby_params.get("by")
        if by is None:
            return result
        by_cols = [by] if isinstance(by, str) else list(by)
        keep = [c for c in result.columns if c not in by_cols]
    keep += [c for c in result.columns if c in hidden and c not in keep]
    return result[keep]


def _grouped_positional_scatter(
    src_df,
    target,
    by_cols,
    dropna,
    sort,
    order_by,
    order_by_ascending,
    compute,
):
    """Row-count-preserving grouped transform via positional scatter-back.

    Tags each row with a collision-free positional ordinal (*pos_col*), groups
    by *by_cols*, applies *compute* (which receives the per-group sub-frame
    already sorted by *order_by* and returns the computed sub-frame), restores
    original within-group order by *pos_col*, and scatters results into a
    NaN-initialised output frame by ordinal — never by index label, which
    would corrupt duplicate-index frames.  Null-key rows (dropna=True) stay
    NaN.  Returns the output frame with *pos_col* already dropped."""
    pos_col = _make_unique_col_name("__mf_pos__", src_df.columns)
    src = src_df.copy()
    src[pos_col] = np.arange(len(src))
    out = src.copy()
    for c in target:
        out[c] = np.nan
    col_idx = {c: out.columns.get_loc(c) for c in target}
    for _key, g in src.groupby(by_cols, dropna=dropna, sort=sort):
        sub = g.copy()
        if order_by:
            asc = order_by_ascending or [True] * len(order_by)
            sub = sub.sort_values(order_by, ascending=asc, kind="stable")
        sub = compute(sub)
        # Restore original within-group order, then scatter by ordinal.
        sub = sub.sort_values(pos_col, kind="stable")
        positions = sub[pos_col].to_numpy()
        for c in target:
            out.iloc[positions, col_idx[c]] = sub[c].to_numpy()
    return out.drop(columns=[pos_col])


def _mf_sort_and_perm(df, order_by, order_by_ascending):
    """Stable sort by *order_by*; return (sorted_df, inverse_perm).

    ``inverse_perm`` is a positional index that restores the original row
    order via ``result.iloc[inverse_perm]``.  Returns ``(df, None)`` when
    ``order_by`` is falsy.
    """
    if not order_by:
        return df, None
    asc = order_by_ascending or [True] * len(order_by)
    sorted_df = df.sort_values(order_by, ascending=asc, kind="stable").reset_index(
        drop=True
    )
    # perm[j] = original index of sorted row j
    perm = (
        df.assign(_mf_i=np.arange(len(df)))
        .sort_values(order_by, ascending=asc, kind="stable")["_mf_i"]
        .to_numpy()
    )
    # inverse_perm[i] = sorted position of original row i
    inverse_perm = np.empty(len(df), dtype=np.intp)
    inverse_perm[perm] = np.arange(len(df))
    return sorted_df, inverse_perm


def _cum_col(series, agg, skipna, remask):
    """Compute a single cumulative/expanding column (pandas, skipna-aware).

    cum* (remask=True): pandas ``cum*`` with ``skipna`` — a NaN poisons the
    rest of the accumulation when ``skipna=False``.
    expanding (remask=False): fill semantics — NaN treated as identity
    (0 for sum, 1 for prod) and forward-filled for min/max, matching the
    DPE polars executor's fill-then-accumulate strategy.
    """
    if agg == "count":
        # cumcount / expanding.count: row-based, not value-based.
        return None
    if remask:
        # cum*: use pandas cum* which respects skipna
        if agg == "sum":
            return series.cumsum(skipna=skipna)
        if agg == "prod":
            return series.cumprod(skipna=skipna)
        if agg == "min":
            return series.cummin(skipna=skipna)
        if agg == "max":
            return series.cummax(skipna=skipna)
    else:
        # expanding fill semantics
        if agg == "sum":
            return series.fillna(0).cumsum()
        if agg == "prod":
            return series.fillna(1).cumprod()
        if agg == "min":
            return series.cummin(skipna=True).ffill()
        if agg == "max":
            return series.cummax(skipna=True).ffill()
        if agg == "mean":
            s = series.astype("float64")
            return s.fillna(0).cumsum() / s.notna().cumsum()
        if agg == "var":
            s = series.astype("float64")
            n = s.notna().cumsum()
            sum_ = s.fillna(0).cumsum()
            sum2 = (s * s).fillna(0).cumsum()
            return (sum2 - sum_ * sum_ / n) / (n - 1)
        if agg == "std":
            s = series.astype("float64")
            n = s.notna().cumsum()
            sum_ = s.fillna(0).cumsum()
            sum2 = (s * s).fillna(0).cumsum()
            return ((sum2 - sum_ * sum_ / n) / (n - 1)) ** 0.5
    raise ValueError(f"Unsupported expanding agg: {agg!r} remask={remask}")


def _apply_expanding_window_shift(
    raw_frame,
    target,
    window_shift,
    reverse,
    is_count,
    row_count_mode,
    agg_func,
    skipna,
    remask,
    mp,
):
    """Apply ``window_shift`` to an expanding result (mcsql frame ``[0, i+shift]``).

    Mirrors the DPE ``_expanding_agg_with_shift``: mcsql expanding with
    ``window_shift=s`` uses the frame ``[0, i+s]``.  The *raw* (min_periods=1,
    ungated) cumulative value and the *raw* cumulative observation count are
    both shifted by ``-s`` so row *i* receives the value/count at ``i+s``;
    ``min_periods`` then gates on the shifted count, never on already-gated
    output (gating must reflect the true number of observations in the shifted
    window).

    * ``s > 0`` — leading edge extends past the last row; trailing nulls are
      forward-filled with the full cumulative value (window truncates to
      ``[0, n-1]``).
    * ``s < 0`` — leading edge shrinks before row 0; leading nulls stay null.

    *reverse* negates the effective shift (the reversed frame inverts the
    expansion direction).  ``raw_frame`` is the pre-gate input block in
    already-sorted/reversed order; the returned frame carries the same index.
    """
    if window_shift == 0:
        return raw_frame.copy()
    eff_shift = -window_shift if reverse else window_shift
    n = len(raw_frame)
    out = raw_frame.copy()
    for c in target:
        if is_count:
            if row_count_mode:
                raw_val = pd.Series(np.arange(n), index=raw_frame.index)
            else:
                raw_val = raw_frame[c].notna().cumsum()
            raw_count = pd.Series(np.arange(1, n + 1), index=raw_frame.index)
        else:
            raw_val = _cum_col(raw_frame[c], agg_func, skipna, remask)
            raw_count = raw_frame[c].notna().cumsum()
        if eff_shift > 0:
            shifted_val = raw_val.shift(-eff_shift).ffill()
            shifted_count = raw_count.shift(-eff_shift).ffill()
            if shifted_val.isna().any():
                last_val = raw_val.iloc[-1] if n else None
                if last_val is not None and not (
                    isinstance(last_val, float) and pd.isna(last_val)
                ):
                    shifted_val = shifted_val.fillna(last_val)
            if shifted_count.isna().any():
                total = n if is_count else int(raw_frame[c].notna().sum())
                shifted_count = shifted_count.fillna(total)
        else:
            abs_shift = abs(eff_shift)
            shifted_val = raw_val.shift(abs_shift)
            shifted_count = raw_count.shift(abs_shift)
        if mp >= 1:
            shifted_val = shifted_val.where(shifted_count >= mp, other=np.nan)
        out[c] = shifted_val
    return out


def _run_expanding_agg(
    result,
    target,
    groupby_params,
    agg_func,
    min_periods,
    skipna,
    remask,
    order_by,
    order_by_ascending,
    window_shift,
    reverse,
    row_count_mode,
):
    """Execute ``expanding_agg`` (cum* / expanding) via pandas.

    Mirrors the DPE polars executor's semantics:
    - order_by: stable-sort, accumulate, place back to original order.
    - reverse: accumulate on reversed data, reverse back.
    - window_shift (expanding): frame ``[0, i+shift]`` — shift the raw
      cumulative by ``-shift`` (forward) and forward-fill the trailing tail,
      shifting the min_periods validity count to match (DPE single-chunk shift).
    - min_periods gating on non-null count (value aggs) or row count
      (count/cumcount).
    - Grouped + dropna=True: null-key rows survive with null target values
      (row-count-preserving scatter-back), mirroring DPE's ``.over()`` +
      ``_non_null_group_key_expr`` masking.
    """
    mp = min_periods if min_periods is not None else 1
    is_count = row_count_mode or agg_func == "count"

    # --- MF-specific preprocessing (standalone path) ---
    placeback = None
    work = result
    if groupby_params is None:
        if order_by:
            work, placeback = _mf_sort_and_perm(work, order_by, order_by_ascending)
        if reverse:
            work = work.iloc[::-1].reset_index(drop=True)

    def _compute_block(frame):
        """Compute expanding on a single (already-sorted) block."""
        out = frame.copy()
        for c in target:
            if is_count:
                if row_count_mode:
                    out[c] = np.arange(len(frame))
                else:
                    out[c] = frame[c].notna().cumsum()
            else:
                out[c] = _cum_col(frame[c], agg_func, skipna, remask)
        # min_periods gating
        if mp > 1 or (not is_count and mp >= 1):
            for c in target:
                if is_count:
                    if mp > 1:
                        row_idx = np.arange(len(frame))
                        mask = row_idx >= mp - 1
                        out.loc[~mask, c] = np.nan
                else:
                    validity = frame[c].notna().cumsum()
                    mask = validity >= mp
                    out.loc[~mask, c] = np.nan
        return out

    if groupby_params is not None:
        by = groupby_params["by"]
        by_cols = [by] if isinstance(by, str) else list(by)
        dropna = groupby_params.get("dropna", True)
        sort = groupby_params.get("sort", True)

        # Row-count-preserving grouped transform via positional scatter-back
        # (see _grouped_positional_scatter).  Null-key rows (dropna=True) are
        # excluded from grouping and keep their null target values.
        def _compute_expanding_group(sub):
            if reverse:
                sub = sub.iloc[::-1]
            if window_shift != 0:
                # _apply_expanding_window_shift recomputes the raw cumulative
                # value + count from this pre-gate block and gates on the
                # shifted count (DPE _expanding_agg_with_shift).  _compute_block
                # would pre-gate and the count would undercount.
                sub = _apply_expanding_window_shift(
                    sub,
                    target,
                    window_shift,
                    reverse,
                    is_count,
                    row_count_mode,
                    agg_func,
                    skipna,
                    remask,
                    mp,
                )
            else:
                sub = _compute_block(sub)
            if reverse:
                sub = sub.iloc[::-1]
            return sub

        return _grouped_positional_scatter(
            result,
            target,
            by_cols,
            dropna,
            sort,
            order_by,
            order_by_ascending,
            _compute_expanding_group,
        )

    # Standalone.  When window_shift != 0, _apply_expanding_window_shift
    # recomputes the raw cumulative value + count from the pre-gate work frame
    # (DPE _expanding_agg_with_shift); _compute_block's gating would
    # undercount the shifted-window observations.
    if window_shift != 0:
        computed = _apply_expanding_window_shift(
            work,
            target,
            window_shift,
            reverse,
            is_count,
            row_count_mode,
            agg_func,
            skipna,
            remask,
            mp,
        )
    else:
        computed = _compute_block(work)

    # MF post-processing
    if reverse:
        computed = computed.iloc[::-1].reset_index(drop=True)
    if placeback is not None:
        computed = computed.iloc[placeback].reset_index(drop=True)

    return computed


def _rolling_pre_shift(df, target, window_shift):
    """Pre-shift target columns by ``-window_shift`` for rolling window_shift.

    Mirrors the DPE ``_mf_rolling_pre_shift``: mcsql rolling with
    ``window_shift=s`` uses the frame ``[i-W+1+s, i+s]``.  Shifting the target
    data by ``-s`` then applying a standard trailing window reproduces this
    exactly (out-of-range positions become null and are skipped).
    """
    if window_shift == 0:
        return df
    out = df.copy()
    for c in target:
        out[c] = out[c].shift(-window_shift)
    return out


def _rolling_agg_block(frame, target, window_size, min_periods, center, agg_func):
    """Compute a rolling aggregation on one (already-sorted/shifted) block."""
    mp = min_periods if min_periods is not None else window_size
    out = frame.copy()
    for c in target:
        roller = out[c].rolling(window=window_size, min_periods=mp, center=center)
        out[c] = roller.agg(agg_func)
    return out


@register_op_adapter(LiteFrameWindowTransform)
class LiteFrameWindowTransformAdapter(SPEOperatorAdapter):
    @staticmethod
    def _run_window_transform(
        df: pd.DataFrame,
        kind: str,
        groupby_params: Optional[dict],
        periods: int = 1,
        fill_value=None,
        columns: Optional[list] = None,
        window_size: Optional[int] = None,
        min_periods: Optional[int] = None,
        agg_func: Optional[str] = None,
        center: bool = False,
        hidden_columns: Optional[list] = None,
        range_columns: dict = None,
        skipna: bool = True,
        remask_input_nan: bool = False,
        order_by: Optional[list] = None,
        order_by_ascending: Optional[list] = None,
        window_shift: int = 0,
        reverse: bool = False,
        row_count_mode: bool = False,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        target = _get_target_cols(
            df, columns, groupby_params, hidden_columns, range_columns
        )

        result = df.copy()

        if kind == "expanding_agg":
            out = _run_expanding_agg(
                result,
                target,
                groupby_params,
                agg_func,
                min_periods,
                skipna,
                remask_input_nan,
                order_by,
                order_by_ascending,
                window_shift,
                reverse,
                row_count_mode,
            )
            return _project_grouped_output(out, groupby_params, hidden_columns)

        if groupby_params is not None:
            by = groupby_params["by"]
            by_cols = [by] if isinstance(by, str) else list(by)
            dropna = groupby_params.get("dropna", True)
            sort = groupby_params.get("sort", True)
            grouped = result.groupby(by_cols, dropna=dropna, sort=sort)

            if kind == "shift":
                for c in target:
                    result[c] = grouped[c].shift(periods, fill_value=fill_value)
            elif kind == "diff":
                for c in target:
                    result[c] = grouped[c].diff(periods)
            elif kind == "pct_change":
                for c in target:
                    shifted = grouped[c].shift(periods)
                    result[c] = (result[c] - shifted) / shifted
            elif kind == "rolling_agg":
                # DPE-aligned grouped rolling.  Two paths:
                # * Fast path (no MF-specific order_by/window_shift): vectorized
                #   groupby.rolling — ~10x faster than a per-group Python loop
                #   and the path DPE's polars .over() takes.
                # * MF path (order_by and/or window_shift): per-group
                #   sort → pre-shift → roll → place-back, mirroring DPE's
                #   _grouped_mf_preprocess + _mf_rolling_pre_shift.  The source
                #   frame is NEVER cleared — clearing then grouping rolls over
                #   nulls.  Results scatter to a separate output frame by
                #   positional ordinal, not index label.
                has_mf_opts = bool(order_by) or window_shift != 0
                if not has_mf_opts:
                    # Fast path: vectorized groupby.rolling.  Place results
                    # back by *positional* ordinal, not index label — label
                    # alignment crashes on duplicate index labels
                    # ("cannot reindex on an axis with duplicate labels") and
                    # would also miswrite rows sharing a label.  Resetting to
                    # a unique RangeIndex makes groupby.rolling alignment
                    # unambiguous; reindex to the full range restores original
                    # order and leaves null-key rows (dropna=True) as NaN.
                    src = df.reset_index(drop=True)
                    grouped_pos = src.groupby(by_cols, dropna=dropna, sort=sort)
                    for c in target:
                        rolled = (
                            grouped_pos[c]
                            .rolling(
                                window=window_size,
                                min_periods=min_periods,
                                center=center,
                            )
                            .agg(agg_func)
                            .reset_index(level=by_cols, drop=True)
                            .reindex(range(len(src)))
                        )
                        result[c] = rolled.to_numpy()
                else:
                    # MF path: per-group sort → pre-shift → roll → place-back
                    # via positional scatter (see _grouped_positional_scatter).
                    def _compute_rolling_group(sub):
                        # DPE order: sort by order_by, then pre-shift target
                        # columns.  Shifting before sorting scrambles the
                        # window when both are combined.
                        if window_shift != 0:
                            sub = _rolling_pre_shift(sub, target, window_shift)
                        return _rolling_agg_block(
                            sub, target, window_size, min_periods, center, agg_func
                        )

                    out = _grouped_positional_scatter(
                        df,
                        target,
                        by_cols,
                        dropna,
                        sort,
                        order_by,
                        order_by_ascending,
                        _compute_rolling_group,
                    )
                    for c in target:
                        result[c] = out[c]
            elif kind in ("ffill", "bfill"):
                fill_fn = "ffill" if kind == "ffill" else "bfill"
                for c in target:
                    result[c] = getattr(grouped[c], fill_fn)(limit=limit)
            else:
                raise ValueError(f"Unsupported window kind: {kind!r}")
        else:
            if kind == "shift":
                for c in target:
                    if fill_value is not None and result[c].dtype == bool:
                        result[c] = result[c].shift(
                            periods, fill_value=bool(fill_value)
                        )
                    else:
                        result[c] = result[c].shift(periods, fill_value=fill_value)
            elif kind == "diff":
                for c in target:
                    result[c] = result[c].diff(periods)
            elif kind == "pct_change":
                for c in target:
                    shifted = result[c].shift(periods)
                    result[c] = (result[c] - shifted) / shifted
            elif kind == "rolling_agg":
                # DPE order (lines 2007-2012): sort by order_by FIRST, then
                # pre-shift target columns, then roll.  Shifting before
                # sorting (the old code) scrambles the window when both are
                # combined — the shift applies to unsorted positions.
                work = result
                placeback = None
                if order_by:
                    work, placeback = _mf_sort_and_perm(
                        work, order_by, order_by_ascending
                    )
                if window_shift != 0:
                    work = _rolling_pre_shift(work, target, window_shift)
                work = _rolling_agg_block(
                    work, target, window_size, min_periods, center, agg_func
                )
                if placeback is not None:
                    # Reset the index so positional assignment back into result
                    # (which keeps the original index) lands by position, not
                    # by label -- iloc[placeback] reorders rows but preserves
                    # the original index labels, which would misalign.
                    work = work.iloc[placeback].reset_index(drop=True)
                    work.index = result.index
                for c in target:
                    result[c] = work[c]
            elif kind == "ffill":
                for c in target:
                    result[c] = result[c].ffill(limit=limit)
            elif kind == "bfill":
                for c in target:
                    result[c] = result[c].bfill(limit=limit)
            else:
                raise ValueError(f"Unsupported window kind: {kind!r}")

        return _project_grouped_output(result, groupby_params, hidden_columns)

    def generate_code(self, op, context: SPECodeContext) -> List[str]:
        cls_name = type(self).__name__
        context.register_import(__name__, from_item=cls_name)
        input_var = context.get_input_tileable_variable(op.inputs[0])
        res_var = context.get_output_tileable_variable(op.outputs[0])

        range_columns = None
        in_tileable = op.inputs[0]
        if hasattr(in_tileable, "frame_metadata") and in_tileable.frame_metadata:
            if in_tileable.frame_metadata.range_columns:
                range_columns = {}
                for name, info in in_tileable.frame_metadata.range_columns.items():
                    range_columns[name] = (info.start, info.stop, info.step)

        hidden_columns = getattr(op, "hidden_columns", None)
        row_count_mode = op.extra_params.get("row_count_mode", False)

        args = self._translate_call_args(
            context,
            op.kind,
            op.groupby_params,
            periods=op.periods,
            fill_value=op.fill_value,
            columns=op.columns,
            window_size=op.window_size,
            min_periods=op.min_periods,
            agg_func=op.agg_func,
            center=op.center,
            hidden_columns=hidden_columns,
            range_columns=range_columns,
            skipna=op.skipna,
            remask_input_nan=op.remask_input_nan,
            order_by=op.order_by,
            order_by_ascending=op.order_by_ascending,
            window_shift=op.window_shift,
            reverse=op.reverse,
            row_count_mode=row_count_mode,
            limit=op.limit,
        )
        return [
            f"{res_var} = {cls_name}._run_window_transform("
            f"{input_var}, {', '.join(args)})"
        ]
