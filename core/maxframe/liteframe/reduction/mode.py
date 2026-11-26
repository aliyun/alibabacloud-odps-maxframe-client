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

"""LiteFrame.mode() API entry point.

The mode(s) of each column via a ``CustomAgg`` tree reduction (per-column
frequency counting packed into single-row list columns) followed by a single
``flat_apply`` call (max + filter + sort + coalesce).  Zero post-reduction
shuffles: the CustomAgg tree always produces a single output chunk for the
whole-frame case.

Architecture
------------
1. **ModeFreqAgg** (CustomAgg): For each input column, group-by value and
   count occurrences, then pack the (val, cnt) pairs into two list columns
   per original column: ``__mf_mode_<colname>_val`` and
   ``__mf_mode_<colname>_cnt``.  The output is always a single row,
   satisfying the ``(1, 2*N)`` shape contract of whole-frame CustomAgg.
   Each column is processed independently so its dtype is preserved (no
   cross-column type promotion).

2. **Combine stage**: detects the list columns, explodes, re-aggregates
   counts per (col, val), and re-packs into a single row.

3. **_mode_extract** (flat_apply UDF): receives one row containing list
   columns for each original column, finds the max count per column,
   filters to max-count values, sorts (with nulls last, matching pandas),
   and yields one output row per mode value.
"""

import warnings

import pandas as pd

from maxframe.lib.version import parse as _parse_version
from maxframe.liteframe.reduction.compile import _is_numeric_dtype
from maxframe.liteframe.reduction.custom_agg import CustomAgg
from maxframe.liteframe.utils import resolve_unique_col_name
from maxframe.udf import builtin_function


def _polars_group_by(df, *by, **kwargs):
    """Compatibility shim: call ``group_by`` (polars >=1.0) or
    ``groupby`` (polars <1.0) depending on the installed version.
    """
    import polars as pl

    if _parse_version(pl.__version__) >= _parse_version("1.0"):
        return df.group_by(*by, **kwargs)
    return df.groupby(*by, **kwargs)


def _polars_len():
    """Return ``pl.len()`` (polars >=0.20) or ``pl.count()`` (older)."""
    import polars as pl

    if _parse_version(pl.__version__) >= _parse_version("0.20"):
        return pl.len()
    return pl.count()


def _val_col(col):
    """Return the list-column name holding mode values for *col*."""
    return f"__mf_mode_{col}_val"


def _cnt_col(col):
    """Return the list-column name holding mode counts for *col*."""
    return f"__mf_mode_{col}_cnt"


def _is_mode_numeric_dtype(dt):
    """Mode-specific numeric predicate.

    Pandas includes boolean columns in ``DataFrame.mode(numeric_only=True)``,
    but the shared ``_is_numeric_dtype`` excludes them.  Using a mode-specific
    predicate avoids changing global aggregation semantics.
    """
    import pyarrow as pa

    try:
        pa_type = dt.pyarrow_dtype if hasattr(dt, "pyarrow_dtype") else None
        if pa_type is not None:
            return pa.types.is_boolean(pa_type) or _is_numeric_dtype(dt)
    except AttributeError:
        pass
    return _is_numeric_dtype(dt) or pd.api.types.is_bool_dtype(dt)


@builtin_function
class ModeFreqAgg(CustomAgg):
    """Tree-reducible frequency table for ``mode()``.

    ``agg()`` handles two schemas, detected by column-name inspection:

    - **Map stage** — input has the original column names (A, B, C …).
      For each column, group-by value + count, then pack into two list
      columns per original column.  Output is a single row.
    - **Combine/agg stage** — input already carries the frequency-table
      list columns (``__mf_mode_<col>_val``, ``__mf_mode_<col>_cnt``).
      Explode the lists, re-aggregate counts, and re-pack into a single row.

    The output schema is identical at every stage, so type inference
    (which calls ``agg`` once on mock data with original columns) sees
    the map-stage path and infers the correct output dtypes (list types
    preserving each column's original dtype).
    """

    def __init__(self, dropna=True, columns=None):
        super().__init__(name="mode_freq")
        self.dropna = dropna
        self.columns = columns

    def agg(self, df):
        import polars as pl

        cols = self.columns or [c for c in df.columns]

        # Detect combine stage: all expected freq columns are present.
        freq_cols = set()
        for col in cols:
            freq_cols.add(_val_col(col))
            freq_cols.add(_cnt_col(col))

        if freq_cols.issubset(set(df.columns)):
            # --- Combine / agg stage: merge frequency tables ---
            packed = {}
            for col in cols:
                vc, cc = _val_col(col), _cnt_col(col)
                # Explode the two list columns in lock-step, re-aggregate.
                exploded = df.select(vc, cc).explode(vc, cc)
                # Filter out synthetic null/0 rows produced by exploding
                # empty typed list columns (all-null chunks with dropna=True).
                exploded = exploded.filter(pl.col(cc).is_not_null())
                merged = _polars_group_by(exploded, vc).agg(pl.col(cc).sum()).sort(vc)
                packed[vc] = [merged[vc].to_list()]
                packed[cc] = [merged[cc].to_list()]
            return pl.DataFrame(packed)
        else:
            # --- Map stage: per-column group-by + count → list pack ---
            packed = {}
            for col in cols:
                vc, cc = _val_col(col), _cnt_col(col)
                # The count alias must not collide with the group key, or
                # polars raises ``DuplicateError`` (e.g. a column literally
                # named ``__cnt``).  Only names inside the grouped frame
                # matter, so the group key alone is the collision set.
                cnt_alias = resolve_unique_col_name("__cnt", [col])
                grouped = _polars_group_by(df, col).agg(_polars_len().alias(cnt_alias))
                if self.dropna:
                    grouped = grouped.filter(pl.col(col).is_not_null())
                packed[vc] = [grouped[col].to_list()]
                packed[cc] = [grouped[cnt_alias].to_list()]
            return pl.DataFrame(packed)


@builtin_function
def _mode_extract(row, all_cols=None):
    """Extract modes from packed frequency lists and yield wide-format rows.

    Called once per row of the single-chunk frequency table produced by the
    CustomAgg tree reduction.  Each row carries, for every original column,
    two list fields: ``__mf_mode_<col>_val`` and ``__mf_mode_<col>_cnt``.

    For each column, finds the max count, filters to max-count values,
    sorts (non-null values first, then NaN last — matching pandas' default
    null-last ordering), and yields one dict per mode value.  Columns with
    fewer modes than the column with the most modes are NaN-padded.
    """
    # Build per-column (val, cnt) lists.
    freq = {}
    for col in all_cols:
        vals = row[_val_col(col)]
        cnts = row[_cnt_col(col)]
        freq[col] = list(zip(vals, cnts)) if vals is not None else []

    # Find max per column, filter to max, sort.
    modes_per_col = {}
    max_rows = 0
    for col in all_cols:
        pairs = freq.get(col, [])
        if not pairs:
            modes_per_col[col] = []
            continue
        max_cnt = max(c for _, c in pairs)
        mode_vals = [v for v, c in pairs if c == max_cnt]
        # Pandas-compatible sort: non-null values sorted naturally,
        # NaN placed last.
        non_null = [v for v in mode_vals if v is not None and not _is_na(v)]
        null_vals = [v for v in mode_vals if v is None or _is_na(v)]
        try:
            non_null.sort()
        except TypeError:
            # Struct/dict/list values cannot be compared for ordering;
            # pandas emits "Unable to sort modes" and returns unsorted.
            warnings.warn(
                "Unable to sort modes; returning modes in arbitrary order",
                UserWarning,
                stacklevel=2,
            )
        modes_per_col[col] = non_null + null_vals
        if len(modes_per_col[col]) > max_rows:
            max_rows = len(modes_per_col[col])

    if max_rows == 0:
        return None

    rows = []
    for i in range(max_rows):
        row_dict = {}
        for col in all_cols:
            modes = modes_per_col[col]
            row_dict[col] = modes[i] if i < len(modes) else None
        rows.append(row_dict)
    return rows


def _is_na(val):
    """Check if *val* is NaN/NA without raising on non-numeric types.

    Returns False for list/array values (treats them as non-null scalars),
    since ``pd.isna`` on a list returns an array that cannot be coerced to
    a boolean.
    """
    import numpy as np

    if isinstance(val, (list, tuple, np.ndarray)):
        return False
    try:
        return bool(pd.isna(val))
    except (TypeError, ValueError):
        return False


def mode(self, dropna=True, numeric_only=False):
    """Get the mode(s) of each column.

    The mode is the value that appears most often.  Multiple modes are
    returned when multiple values tie for the highest frequency.  The
    result is a LiteFrame with the same columns as the input, where each
    column contains its mode value(s), NaN-padded for alignment.

    Parameters
    ----------
    dropna : bool, default True
        Don't consider counts of NaN/NaT.
    numeric_only : bool, default False
        If True, only apply to numeric columns (including boolean, matching
        pandas ``DataFrame.mode(numeric_only=True)``).

    Returns
    -------
    LiteFrame
        The modes of each column.

    Examples
    --------
    >>> import maxframe.liteframe as mlf
    >>> lf = mlf.LiteFrame({'a': [1, 2, 2, 3], 'b': [4, 4, 5, 5]})
    >>> lf.mode().execute()
         a    b
    0  2.0    4
    1  NaN    5
    """
    original_columns = list(self.columns)

    # --- Column selection ---
    if numeric_only:
        data_cols = [
            c for c in original_columns if _is_mode_numeric_dtype(self.dtypes[c])
        ]
        lf = self.select(data_cols)
    else:
        data_cols = original_columns
        lf = self

    # --- Step 1: CustomAgg → packed frequency lists (single row, tree-reduced) ---
    freq = lf.agg(
        ModeFreqAgg(dropna=dropna, columns=data_cols),
        input_frame_type="polars",
    )

    # --- Step 2: flat_apply → max + filter + sort + coalesce (0 shuffles) ---
    out_dtypes = {col: self.dtypes[col] for col in data_cols}
    result = freq.flat_apply(
        _mode_extract,
        dtypes=out_dtypes,
        input_frame_type="pandas",
        args=(data_cols,),
    )

    # --- Step 3: post-processing ---
    # Reorder to match original column order (flat_apply output follows
    # the all_cols list, which is already in original order — no-op for
    # numeric_only=False; for numeric_only=True, the subset is already
    # in original order).
    result = result.select(data_cols)

    return result
