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

"""LiteFrame.describe() API entry point.

Descriptive statistics via ``agg()`` + ``flat_apply()`` with a
``@builtin_function`` UDF that unpivots the wide agg result into
statistic-indexed rows.
"""

import pandas as pd
import pyarrow.types as patypes

from maxframe.liteframe.reduction.compile import AggCall, _is_numeric_dtype
from maxframe.liteframe.reduction.quantile import (
    _format_q_percent,
    _format_q_suffix,
    _normalize_q,
)
from maxframe.liteframe.utils import is_datetime_dtype, resolve_unique_col_name
from maxframe.udf import builtin_function


def _format_stat_label(q):
    """Format a quantile as a display label: 0.25 -> '25%', 0.05 -> '5%'."""
    return _format_q_percent(q) + "%"


def _datetime_output_dtype(pandas_dtype):
    """Derive the flat_apply output dtype for a datetime column.

    Maps the source column's arrow type to a pandas datetime64 string:
      timestamp[us] -> 'datetime64[us]'
      timestamp[ns] -> 'datetime64[ns]'
      date32[day]   -> 'datetime64[us]'   (date has no time; us is safe)
    Falls back to 'datetime64[us]' for any unrecognised datetime dtype.
    """
    if hasattr(pandas_dtype, "pyarrow_dtype"):
        arrow_type = pandas_dtype.pyarrow_dtype
        if patypes.is_timestamp(arrow_type):
            return f"datetime64[{arrow_type.unit}]"
        if patypes.is_date(arrow_type):
            return "datetime64[us]"
    if pd.api.types.is_datetime64_any_dtype(pandas_dtype):
        return str(pandas_dtype)
    return "datetime64[us]"


def _resolve_describe_cols(dtypes, numeric_only):
    """Return ``(data_cols, datetime_cols)`` for describe.

    *data_cols* is the list of eligible columns (numeric, or numeric +
    datetime when *numeric_only* is False).  *datetime_cols* is the subset
    that are datetime (empty when *numeric_only* is True).
    """
    if numeric_only:
        data_cols = [c for c in dtypes.index if _is_numeric_dtype(dtypes[c])]
        return data_cols, []
    data_cols = [
        c
        for c in dtypes.index
        if _is_numeric_dtype(dtypes[c]) or is_datetime_dtype(dtypes[c])
    ]
    datetime_cols = [c for c in data_cols if is_datetime_dtype(dtypes[c])]
    return data_cols, datetime_cols


def _build_describe_agg_funcs(qs, data_cols=None, datetime_cols=None):
    """Build agg funcs for describe: count, mean, std, min, q*, max.

    When *datetime_cols* is non-empty, returns a dict mapping each column
    in *data_cols* to its own func list so datetime columns skip
    ``mean``/``std`` (which polars cannot evaluate on a Datetime column).
    Numeric columns get the full func list.  When *datetime_cols* is empty
    or None, returns a flat list (applies to all columns uniformly).
    """
    numeric_funcs = ["count", "mean", "std", "min"]
    for q in qs:
        numeric_funcs.append(
            AggCall(
                "quantile",
                q=q,
                interpolation="linear",
                output_suffix=_format_q_suffix(q),
            )
        )
    numeric_funcs.append("max")

    if not datetime_cols:
        return numeric_funcs

    # Datetime columns: skip mean/std (emit NaT in the UDF instead)
    dt_funcs = ["count", "min"]
    for q in qs:
        dt_funcs.append(
            AggCall(
                "quantile",
                q=q,
                interpolation="linear",
                output_suffix=_format_q_suffix(q),
            )
        )
    dt_funcs.append("max")

    dt_set = set(datetime_cols)
    return {
        col: (list(dt_funcs) if col in dt_set else list(numeric_funcs))
        for col in data_cols
    }


def _build_stat_pairs(qs):
    """Build ``(display_label, agg_suffix)`` tuples for the UDF."""
    pairs = [("count", "count"), ("mean", "mean"), ("std", "std"), ("min", "min")]
    for q in qs:
        pairs.append((_format_stat_label(q), _format_q_suffix(q)))
    pairs.append(("max", "max"))
    return pairs


def _build_col_key_map(
    data_cols, stat_pairs, datetime_cols, agg_cols, exclude_cols=None
):
    """Build ``(col, suffix) -> actual agg column name`` map and the list of
    stat column names for ``flat_apply``'s ``subset``.

    The actual agg result column names are used (rather than reconstructing
    ``f"{col}_{suffix}"``) because the compile layer may rename an output
    column to avoid collision with a group key (e.g. ``a_count`` ->
    ``a_count_1`` when a group key is also named ``a_count``).

    *exclude_cols* (group key names for groupby describe) are never matched
    as stat columns -- a candidate that equals a group key must be resolved
    to its suffixed variant.

    Returns ``(col_key_map, stat_cols)`` where *col_key_map* maps
    ``(col, suffix)`` to the actual column name in the agg result, and
    *stat_cols* is the ordered list of those column names (excluding
    datetime cols' mean/std which are not computed).
    """
    dt_set = set(datetime_cols)
    agg_set = set(agg_cols)
    exclude = set(exclude_cols) if exclude_cols else set()
    used = set()
    col_key_map = {}
    stat_cols = []

    def _resolve(candidate):
        """Find the actual agg column for *candidate*, skipping names in
        *exclude* or already *used*; falls back to ``{candidate}_1`` etc."""
        if candidate not in exclude and candidate not in used and candidate in agg_set:
            return candidate
        for i in range(1, 100):
            suffixed = f"{candidate}_{i}"
            if suffixed in agg_set and suffixed not in used and suffixed not in exclude:
                return suffixed
        return candidate

    for col in data_cols:
        for _, suffix in stat_pairs:
            if col in dt_set and suffix in ("mean", "std"):
                continue
            actual = _resolve(f"{col}_{suffix}")
            col_key_map[(col, suffix)] = actual
            stat_cols.append(actual)
            used.add(actual)
    return col_key_map, stat_cols


def _build_out_dtypes(data_cols, datetime_cols, dtypes, stat_col):
    """Build the ``dtypes`` dict for ``flat_apply``."""
    out = {stat_col: "string"}
    for c in data_cols:
        out[c] = _datetime_output_dtype(dtypes[c]) if c in datetime_cols else "double"
    return out


def _validate_percentiles(percentiles):
    """Validate and normalize the *percentiles* argument."""
    if percentiles is None:
        percentiles = [0.25, 0.5, 0.75]
    qs = _normalize_q(percentiles)
    seen = set()
    for q in qs:
        if q in seen:
            raise ValueError(f"percentiles cannot contain duplicates; got {qs}")
        seen.add(q)
        if q in (0.0, 1.0):
            raise ValueError(
                "percentiles must not contain 0.0 or 1.0; "
                "they are covered by 'min' and 'max'"
            )
    return sorted(qs)


@builtin_function
def _describe_expand(row, data_cols, stat_pairs, datetime_cols, stat_col, col_key_map):
    """Expand one agg row into N statistic rows.

    Parameters (passed via flat_apply ``args``):
        row          -- pandas.Series indexed by {col}_{stat} columns
        data_cols    -- list of original data column names
        stat_pairs   -- list of (display_name, agg_suffix) tuples
        datetime_cols -- list of datetime column names
        stat_col     -- name for the statistic-index column
        col_key_map   -- dict mapping (col, suffix) -> actual agg result
                        column name (may differ from f"{col}_{suffix}" when
                        the compile layer resolved a name collision with a
                        group key, e.g. "a_count" -> "a_count_1")
    """
    dt_set = set(datetime_cols)
    result = []
    for stat_name, suffix in stat_pairs:
        d = {stat_col: stat_name}
        for col in data_cols:
            col_key = col_key_map.get((col, suffix), f"{col}_{suffix}")
            if col_key in row.index:
                val = row[col_key]
                if col in dt_set:
                    if stat_name in ("count", "mean", "std"):
                        d[col] = pd.NaT
                    else:
                        d[col] = val
                else:
                    d[col] = float("nan") if pd.isna(val) else float(val)
            elif col in dt_set:
                # mean/std not computed for datetime columns
                d[col] = pd.NaT
            else:
                d[col] = float("nan")
        result.append(d)
    return result


def describe(self, percentiles=None, numeric_only=True):
    """Generate descriptive statistics.

    Computes count, mean, std, min, percentiles, and max for each
    eligible column.  The result is a long-format LiteFrame with a
    ``statistic`` column identifying the statistic and one column per
    data column.  If a data column is already named ``"statistic"``, the
    index column is suffixed (``"statistic_1"``, ``"statistic_2"``, …).

    Parameters
    ----------
    percentiles : list of float, default [0.25, 0.5, 0.75]
        Percentiles to compute, in ``[0, 1]``.  Must be unique and exclude
        0 and 1 (covered by min/max).  Unsorted input is silently sorted
        ascending (matching pandas).
    numeric_only : bool, default True
        If True, only numeric columns are described.  If False, both
        numeric and datetime columns are included.  For datetime columns,
        ``count``, ``mean``, and ``std`` emit ``NaT`` (polars cannot
        evaluate mean/std on Datetime, and an integer count cannot fit a
        datetime type); ``min``, percentiles, and ``max`` pass through as
        timestamps.  This diverges from pandas, which computes a real
        Timestamp mean and an integer count.

    Returns
    -------
    LiteFrame
        Long-format frame with columns ``["statistic", {col_1}, ...]``.

    Examples
    --------
    >>> import maxframe.liteframe as mlf
    >>> lf = mlf.LiteFrame({'a': [1, 2, 3], 'b': [4.0, 5.0, 6.0]})
    >>> result = lf.describe()
    >>> result.execute()
    statistic    a    b
    0     count  3.0  3.0
    1      mean  2.0  5.0
    2       std  1.0  1.0
    3       min  1.0  4.0
    4       25%  1.5  4.5
    5       50%  2.0  5.0
    6       75%  2.5  5.5
    7       max  3.0  6.0
    """
    qs = _validate_percentiles(percentiles)
    dtypes = self.dtypes
    data_cols, datetime_cols = _resolve_describe_cols(dtypes, numeric_only)
    stat_col = resolve_unique_col_name("statistic", data_cols)
    if not numeric_only and datetime_cols:
        agg_input = self[data_cols]
        agg_numeric_only = False
    else:
        agg_input = self
        agg_numeric_only = True

    funcs = _build_describe_agg_funcs(qs, data_cols, datetime_cols)
    agg_result = agg_input.agg(funcs, numeric_only=agg_numeric_only)

    stat_pairs = _build_stat_pairs(qs)
    col_key_map, stat_cols = _build_col_key_map(
        data_cols, stat_pairs, datetime_cols, list(agg_result.dtypes.index)
    )
    out_dtypes = _build_out_dtypes(data_cols, datetime_cols, dtypes, stat_col)

    return agg_result.flat_apply(
        _describe_expand,
        args=(data_cols, stat_pairs, datetime_cols, stat_col, col_key_map),
        dtypes=out_dtypes,
        subset=stat_cols,
    )
