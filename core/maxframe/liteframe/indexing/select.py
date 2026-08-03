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

import pandas as pd

from maxframe.liteframe.arithmetic.core import (
    _build_fused_projection,
    _extract_projection_exprs,
    _resolve_named_expr,
    _rewrite_column_refs,
    _root_source_lf,
)
from maxframe.liteframe.core import LITEFRAME_TYPE, LiteFrame
from maxframe.liteframe.expressions import (
    LiteFrameColumn,
    LiteFrameExpr,
    LiteFrameLiteral,
    NamedExpr,
)
from maxframe.liteframe.operators.project import LiteFrameProjection
from maxframe.liteframe.utils import normalize_col_name, validate_errors


def select(lf, columns):
    """Explicit column selection."""
    columns = [normalize_col_name(c) for c in columns]
    projections = [_resolve_named_expr(lf, c, lf.dtypes[c]) for c in columns]
    return _build_fused_projection(lf, projections)


def select_expr(lf, *exprs):
    """Select columns using LiteFrameExpr objects, binding dtypes from lf."""
    existing_exprs = _extract_projection_exprs(lf)
    bound = []
    for i, expr in enumerate(exprs):
        bound_expr = expr.bind(lf.dtypes)
        # Rewrite column references through lf's projection so the
        # expression is self-contained and safe for later fusion.
        bound_expr = _rewrite_column_refs(bound_expr, existing_exprs)
        # Wrap in NamedExpr if not already named
        if isinstance(bound_expr, NamedExpr):
            bound.append(bound_expr)
        elif isinstance(bound_expr, LiteFrameColumn):
            bound.append(bound_expr)
        else:
            # Anonymous expression - wrap in NamedExpr with a generated name
            name = getattr(bound_expr, "name", None) or f"_expr_{i}"
            bound.append(NamedExpr(name=name, expr=bound_expr))
    return _build_fused_projection(lf, bound)


def select_dispatch(self, *args):
    """Dispatch to select or select_expr based on argument types."""
    if args and isinstance(args[0], LiteFrameExpr):
        return select_expr(self, *args)
    # Normalize: single string → list, multiple args → list
    if len(args) == 1:
        columns = [args[0]] if isinstance(args[0], str) else args[0]
    else:
        columns = list(args)
    return select(self, columns)


def drop(lf, columns, errors="raise"):
    """Drop columns from LiteFrame."""
    validate_errors(errors, allow_coerce=False)
    if isinstance(columns, str):
        columns = [columns]
    columns = [normalize_col_name(c) for c in columns]
    columns_set = set(columns)

    # Separate into visible drops and hidden drops
    visible_drops = columns_set & set(lf.dtypes.index)
    hidden_drops = columns_set & lf._hidden_columns
    unknown = columns_set - visible_drops - hidden_drops

    if errors == "raise" and unknown:
        raise KeyError(f"Column(s) {unknown} not found")

    # Build projection over remaining visible columns
    remaining_visible = [c for c in lf.columns if c not in visible_drops]
    projections = [_resolve_named_expr(lf, c, lf.dtypes[c]) for c in remaining_visible]

    # Append surviving hidden columns (hidden_drops are simply omitted)
    surviving_hidden = lf._hidden_columns - hidden_drops
    for hc in sorted(surviving_hidden):
        projections.append(LiteFrameColumn(name=hc, dtype=lf._physical_dtypes[hc]))

    # Build projection directly — hidden columns already handled above,
    # so we bypass _build_fused_projection which would re-add all hidden columns.
    op = LiteFrameProjection(projections=projections)
    root_source = _root_source_lf(lf)
    return op(LiteFrame(root_source))


def rename(lf, columns):
    """Rename columns in LiteFrame."""
    # Normalize rename dict keys (old names) and values (new names)
    normalized_columns = {
        normalize_col_name(k): normalize_col_name(v) for k, v in columns.items()
    }
    existing_exprs = _extract_projection_exprs(lf)
    projections = []
    for c in lf.columns:
        col_expr = existing_exprs.get(c, LiteFrameColumn(name=c, dtype=lf.dtypes[c]))
        if c in normalized_columns:
            projections.append(NamedExpr(name=normalized_columns[c], expr=col_expr))
        else:
            if isinstance(col_expr, (NamedExpr, LiteFrameColumn)):
                projections.append(col_expr)
            else:
                projections.append(NamedExpr(name=c, expr=col_expr))

    return _build_fused_projection(lf, projections)


def _resolve_assign_value(name, value, lf):
    """Resolve an assign value to a (projection, dtype) pair.

    Returns (LiteFrameExpr, dtype) where the projection's output name is `name`.
    """
    if callable(value):
        value = value(lf)

    # Unwrap a single-column LiteFrame projection into its LiteFrameExpr,
    # so that assign(col=df[col].astype(...)) preserves the operation graph.
    if (
        isinstance(value, LITEFRAME_TYPE)
        and isinstance(value.op, LiteFrameProjection)
        and len(value.op.projections) == 1
    ):
        value = value.op.projections[0]

    if isinstance(value, LiteFrameExpr):
        bound = value.bind(lf.dtypes)
        # Rewrite column references through lf's projection so the
        # expression is self-contained and safe for fusion.
        existing = _extract_projection_exprs(lf)
        bound = _rewrite_column_refs(bound, existing)
        if isinstance(bound, NamedExpr):
            return bound, bound.dtype
        return bound.rename(name), bound.dtype

    if isinstance(value, LITEFRAME_TYPE):
        if len(value.columns) != 1:
            raise ValueError(
                f"Cannot assign a LiteFrame with {len(value.columns)} columns "
                f"to a single column '{name}'"
            )
        col_name = value.columns[0]
        expr = LiteFrameColumn(name=col_name, dtype=value.dtypes[col_name])
        return expr.rename(name), value.dtypes.iloc[0]

    dtype = pd.Series([value]).dtype
    expr = LiteFrameLiteral(value=value, dtype=dtype)
    return expr.rename(name), dtype


def assign(lf, **kwargs):
    """Add new columns to LiteFrame."""
    normalized_kwargs = {normalize_col_name(k): v for k, v in kwargs.items()}
    existing_names = set(lf.columns)
    existing_exprs = _extract_projection_exprs(lf)

    # Build projections in original column order, replacing overwritten
    # columns in-place so they keep their position; new columns are appended.
    projections = []

    for c in lf.columns:
        if c in normalized_kwargs:
            proj, dtype = _resolve_assign_value(c, normalized_kwargs[c], lf)
        else:
            col_expr = existing_exprs.get(
                c, LiteFrameColumn(name=c, dtype=lf.dtypes[c])
            )
            if isinstance(col_expr, (NamedExpr, LiteFrameColumn)):
                proj = col_expr
            else:
                proj = NamedExpr(name=c, expr=col_expr)
        projections.append(proj)

    for name, value in normalized_kwargs.items():
        if name not in existing_names:
            proj, dtype = _resolve_assign_value(name, value, lf)
            projections.append(proj)

    return _build_fused_projection(lf, projections)
