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

"""LiteFrame.cov() / corr() API entry points.

Pairwise covariance/correlation via ``agg()`` (with ``sum_xy`` primitive)
+ ``flat_apply`` (to expand the single-row upper-triangle result into an
N×N matrix with a ``columns`` row-label column).
"""

from maxframe.liteframe.operators.reduction import LiteFrameAgg
from maxframe.liteframe.reduction.compile import AggCall
from maxframe.liteframe.utils import is_numeric_dtype
from maxframe.udf import builtin_function

# Row-label column name used in the N×N matrix output.  Matches pandas'
# index convention (the column names become the row labels).  When the
# default name collides with a numeric output column or group key,
# ``_resolve_label_col`` probes ``columns``, ``columns_1``, … until a
# free name is found.
_LABEL_COL = "columns"


def _resolve_numeric_cols(liteframe):
    """Return the list of numeric column names from the frame."""
    dtypes = liteframe.dtypes
    return [c for c in dtypes.index if is_numeric_dtype(dtypes[c])]


def _resolve_label_col(numeric_cols, group_key_cols=None):
    """Return the row-label column name, avoiding collisions.

    The label must not collide with any numeric output column or group
    key name, since both appear alongside it in the expanded matrix.
    Probes ``columns``, then ``columns_1``, ``columns_2``, … until a
    free name is found.
    """
    avoid = set(numeric_cols)
    if group_key_cols:
        avoid |= set(group_key_cols)
    if _LABEL_COL not in avoid:
        return _LABEL_COL
    i = 1
    while f"{_LABEL_COL}_{i}" in avoid:
        i += 1
    return f"{_LABEL_COL}_{i}"


@builtin_function
def _pairwise_matrix_expand(
    row,
    numeric_cols,
    func_name,
    label_col,
    group_key_cols=None,
    agg_output_columns=None,
):
    """Expand one agg row into N matrix rows.

    Each row dict carries the group key values (if *group_key_cols* is
    given), a row-label column named *label_col*, and one float value
    per numeric column drawn from the upper-triangle agg output.

    *agg_output_columns* is the ordered list of agg output column names
    (from ``CompileAggResult.agg_output_columns``), indexed in the same
    triangular order as the pairs generated from *numeric_cols*.
    """
    result = []
    # Sort to match the compiler's sorted pair generation.
    numeric_cols = sorted(numeric_cols)
    pairs = [
        (numeric_cols[i], numeric_cols[j])
        for i in range(len(numeric_cols))
        for j in range(i, len(numeric_cols))
    ]
    col_lookup = dict(zip(pairs, agg_output_columns)) if agg_output_columns else {}
    for ci in numeric_cols:
        d = {gk: row[gk] for gk in group_key_cols} if group_key_cols else {}
        d[label_col] = ci
        for cj in numeric_cols:
            lo, hi = (ci, cj) if ci <= cj else (cj, ci)
            col = col_lookup.get((lo, hi), f"{lo}__{hi}__{func_name}")
            d[cj] = float(row[col])
        result.append(d)
    return result


def _pairwise_call(
    self, func_name, method="auto", numeric_only=False, min_periods=None, ddof=1, **kw
):
    """Run the pairwise agg, then flat_apply to expand into an N×N matrix.

    ``numeric_only`` and ``min_periods`` are enforced by
    ``_compile_pairwise_agg`` (the shared pairwise boundary for both
    global and grouped callers) via ``AggCall.kwds``.
    """
    numeric_cols = _resolve_numeric_cols(self)
    label_col = _resolve_label_col(numeric_cols)

    op = LiteFrameAgg(
        raw_func=AggCall(
            func_name,
            ddof=ddof,
            min_periods=min_periods,
            numeric_only=numeric_only,
        ),
        groupby_params=None,
        method=method,
        numeric_only=numeric_only,
        **kw,
    )
    agg_result = op(self)
    out_dtypes = {label_col: "string", **{c: "double" for c in numeric_cols}}
    return agg_result.flat_apply(
        _pairwise_matrix_expand,
        args=(numeric_cols, func_name, label_col, None, op.agg_output_columns),
        dtypes=out_dtypes,
    )


_PAIRWISE_PARAMS = """
Parameters
----------
method : {'auto', 'tree', 'shuffle'}, default 'auto'
    Execution strategy.  ``'auto'`` selects based on data size.
numeric_only : bool, default False
    If True, only numeric columns are included.  If False (the pandas
    2.0+ default), non-numeric columns cause a TypeError.
min_periods : int, optional
    Minimum non-null observations per pair; below this, NaN.
ddof : int, default 1
    Delta degrees of freedom.  Divides by ``n - ddof``.
**kw
    Additional keyword arguments forwarded to the aggregation.
"""


def cov(self, method="auto", numeric_only=False, min_periods=None, ddof=1, **kw):
    """Compute pairwise covariance of columns.

    Returns an N×N matrix via flat_apply, with a ``columns`` string
    column as the row label (matching pandas' index).  When a data
    column or group key is named ``columns``, the label probes to
    ``columns_1``, ``columns_2``, … to avoid collision.  The diagonal
    equals the variance of each column.

    Covariance is computed pairwise-complete: each pair (i, j) uses
    only rows where both columns are non-null, matching pandas.
    """
    return _pairwise_call(
        self,
        "cov",
        method=method,
        numeric_only=numeric_only,
        min_periods=min_periods,
        ddof=ddof,
        **kw,
    )


cov.__doc__ += (
    _PAIRWISE_PARAMS
    + """
Returns
-------
LiteFrame
    N×N matrix with a ``columns`` string column and one column per
    numeric column.
"""
)


def corr(self, method="auto", numeric_only=False, min_periods=None, ddof=1, **kw):
    """Compute pairwise Pearson correlation of columns.

    Returns an N×N matrix via flat_apply, with a ``columns`` string
    column as the row label (matching pandas' index).  When a data
    column or group key is named ``columns``, the label probes to
    ``columns_1``, ``columns_2``, … to avoid collision.  The diagonal
    is always 1.0.

    Correlation is computed pairwise-complete: each pair (i, j) uses
    only rows where both columns are non-null, matching pandas.
    """
    return _pairwise_call(
        self,
        "corr",
        method=method,
        numeric_only=numeric_only,
        min_periods=min_periods,
        ddof=ddof,
        **kw,
    )


corr.__doc__ += (
    _PAIRWISE_PARAMS
    + """
Returns
-------
LiteFrame
    N×N matrix with a ``columns`` string column and one column per
    numeric column.
"""
)
