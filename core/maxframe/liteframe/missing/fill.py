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

import pyarrow as pa

from maxframe.liteframe.arithmetic.core import (
    _build_binary_projection,
    _build_fused_projection,
    _resolve_column_expr,
    _resolve_named_expr,
)
from maxframe.liteframe.expressions import (
    LiteFrameEq,
    LiteFrameFillNA,
    LiteFrameLiteral,
    LiteFrameWhere,
    NamedExpr,
)
from maxframe.liteframe.utils import normalize_col_name
from maxframe.utils import wrap_arrow_dtype


def fillna(lf, value=None):
    """Fill missing values.

    Parameters
    ----------
    value : scalar or dict
        Scalar value to fill all NAs, or ``{column: value}`` for
        per-column fill values.  Columns not present in the dict
        are left unchanged.

    Returns
    -------
    LiteFrame
        LiteFrame with missing values filled.

    See Also
    --------
    isna : Detect missing values.
    dropna : Drop rows with missing values.
    replace : Replace values in the LiteFrame.

    Examples
    --------
    >>> import maxframe.liteframe as mlf
    >>> lf = mlf.LiteFrame({'a': [1.0, None, 3.0], 'b': [4.0, 5.0, None]})
    >>> result = lf.fillna(0)
    >>> result.execute()
         a    b
    0  1.0  4.0
    1  0.0  5.0
    2  3.0  0.0
    """
    if not isinstance(value, dict):
        return _build_binary_projection(lf, LiteFrameFillNA, value)

    # Normalize fill dict keys once before the loop
    fill_map = {normalize_col_name(col): val for col, val in value.items()}

    # Range columns are virtual indices and never contain NAs, so skip
    # them even if present in the fill dict — no FillNA needed.
    range_col_names = (
        set(lf.frame_metadata.range_columns.keys())
        if lf.frame_metadata and lf.frame_metadata.range_columns
        else set()
    )

    # Per-column fill: only apply FillNA to columns in the dict
    projections = []
    for col_name in lf.columns:
        col_dtype = lf.dtypes[col_name]
        col_ref = _resolve_named_expr(lf, col_name, col_dtype)
        if col_name in fill_map and col_name not in range_col_names:
            # _resolve_named_expr returns a bare LiteFrameColumn only when it
            # already carries the logical name; extract its inner expr so the
            # FillNA operand is the root-source column, then wrap with the
            # logical output name.
            inner = col_ref.expr if isinstance(col_ref, NamedExpr) else col_ref
            fill_literal = LiteFrameLiteral(value=fill_map[col_name], dtype=col_dtype)
            expr = LiteFrameFillNA(left=inner, right=fill_literal, dtype=col_dtype)
            projections.append(NamedExpr(name=col_name, expr=expr))
        else:
            projections.append(col_ref)

    return _build_fused_projection(lf, projections)


def replace(lf, to_replace=None, value=None):
    """Replace values in the LiteFrame.

    When ``to_replace`` is ``None``, behaves like ``fillna`` (replaces
    only NA/null values).  When ``to_replace`` is a scalar, replaces all
    occurrences of that value with ``value`` across every column.

    Parameters
    ----------
    to_replace : scalar, optional
        Value to replace.  When ``None`` (the default), only NA/null values
        are replaced, behaving like :func:`fillna`.  Only scalar
        ``to_replace`` is supported.
    value : scalar, optional
        Replacement value used in place of ``to_replace`` (or in place of
        NA when ``to_replace`` is ``None``).

    Returns
    -------
    LiteFrame
        LiteFrame with matching values replaced.

    See Also
    --------
    fillna : Fill missing values.
    where : Replace values where a condition is False.

    Examples
    --------
    >>> import maxframe.liteframe as mlf
    >>> lf = mlf.LiteFrame({'a': [1, 2, 1], 'b': [1, 1, 2]})
    >>> result = lf.replace(1, 99)
    >>> result.execute()
         a    b
    0  99  99
    1   2  99
    2  99   2
    """
    if to_replace is None:
        return fillna(lf, value=value)

    projections = []
    for col_name in lf.columns:
        col_dtype = lf.dtypes[col_name]
        col_ref = _resolve_column_expr(lf, col_name, col_dtype)
        literal_to_replace = LiteFrameLiteral(value=to_replace, dtype=col_dtype)
        literal_value = LiteFrameLiteral(value=value, dtype=col_dtype)

        condition = LiteFrameEq(
            left=col_ref,
            right=literal_to_replace,
            dtype=wrap_arrow_dtype(pa.bool_()),
        )
        expr = LiteFrameWhere(
            condition=condition,
            true_value=literal_value,
            false_value=col_ref,
            dtype=col_dtype,
        )
        projections.append(NamedExpr(name=col_name, expr=expr))

    return _build_fused_projection(lf, projections)
