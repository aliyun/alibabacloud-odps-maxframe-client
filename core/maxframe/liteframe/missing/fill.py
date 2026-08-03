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
)
from maxframe.liteframe.expressions import (
    LiteFrameColumn,
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
        col_ref = _resolve_column_expr(lf, col_name, col_dtype)
        if col_name in fill_map and col_name not in range_col_names:
            fill_literal = LiteFrameLiteral(value=fill_map[col_name], dtype=col_dtype)
            expr = LiteFrameFillNA(left=col_ref, right=fill_literal, dtype=col_dtype)
            projections.append(NamedExpr(name=col_name, expr=expr))
        else:
            if isinstance(col_ref, (NamedExpr, LiteFrameColumn)):
                projections.append(col_ref)
            else:
                projections.append(NamedExpr(name=col_name, expr=col_ref))

    return _build_fused_projection(lf, projections)


def replace(lf, to_replace=None, value=None):
    """Replace values in the LiteFrame.

    When ``to_replace`` is ``None``, behaves like ``fillna`` (replaces
    only NA/null values).  When ``to_replace`` is a scalar, replaces all
    occurrences of that value with ``value`` across every column.
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

    # replace() transforms every column including virtual range columns,
    # so materialize them when present.
    range_col_names = (
        set(lf.frame_metadata.range_columns.keys())
        if lf.frame_metadata and lf.frame_metadata.range_columns
        else set()
    )
    materialize_range_cols = bool(range_col_names & set(lf.columns))

    return _build_fused_projection(
        lf, projections, materialize_range_cols=materialize_range_cols
    )
