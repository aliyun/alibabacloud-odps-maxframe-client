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

from numbers import Integral

import pandas as pd

from maxframe.liteframe.arithmetic.core import (
    _build_fused_projection,
    _extract_projection_exprs,
    _rewrite_column_refs,
)
from maxframe.liteframe.core import LITEFRAME_TYPE
from maxframe.liteframe.expressions import (
    LiteFrameColumn,
    LiteFrameExpr,
    LiteFrameLiteral,
    NamedExpr,
)
from maxframe.liteframe.operators.project import LiteFrameProjection
from maxframe.liteframe.utils import normalize_col_name


def liteframe_setitem(lf, key, value):
    """Add or replace a column in LiteFrame, or set rows by position."""
    if isinstance(key, slice):
        edge = key.start if key.start is not None else key.stop
        if isinstance(edge, Integral):
            lf.iloc[key] = value
            return
        raise KeyError(f"Unsupported slice key: {key}")

    key = normalize_col_name(key)

    # Unwrap a single-column LiteFrame projection into its LiteFrameExpr,
    # so that lf["c"] = lf["a"] + 1 preserves the operation graph.
    if (
        isinstance(value, LITEFRAME_TYPE)
        and isinstance(value.op, LiteFrameProjection)
        and len(value.op.projections) == 1
    ):
        value = value.op.projections[0]

    if isinstance(value, LiteFrameExpr):
        # Bind to resolve dtypes; if already a NamedExpr, use its inner expr
        # so the final NamedExpr(name=key, ...) wrapping below is correct.
        inner = value.expr if isinstance(value, NamedExpr) else value
        expr = inner.bind(lf.dtypes)
        # Rewrite column references through lf's projection so the
        # expression is self-contained and safe for fusion.
        existing = _extract_projection_exprs(lf)
        expr = _rewrite_column_refs(expr, existing)
    elif isinstance(value, LITEFRAME_TYPE):
        if len(value.columns) != 1:
            raise ValueError(
                f"Cannot setitem a LiteFrame with {len(value.columns)} columns "
                f"to a single column '{key}'"
            )
        col_name = value.columns[0]
        expr = LiteFrameColumn(name=col_name, dtype=value.dtypes[col_name])
    else:
        # Scalar or other value — infer dtype from the value itself
        dtype = pd.Series([value]).dtype
        expr = LiteFrameLiteral(value=value, dtype=dtype)

    # Build projections: all existing columns + new named expression
    existing_exprs = _extract_projection_exprs(lf)
    projections = []
    for c in lf.columns:
        if c != key:
            col_expr = existing_exprs.get(
                c, LiteFrameColumn(name=c, dtype=lf.dtypes[c])
            )
            # Wrap bare expressions in NamedExpr so _projection_name can extract the name
            if not isinstance(col_expr, (NamedExpr, LiteFrameColumn)):
                projections.append(NamedExpr(name=c, expr=col_expr))
            else:
                projections.append(col_expr)
    projections.append(NamedExpr(name=key, expr=expr))

    result = _build_fused_projection(lf, projections)

    # Detach old data and attach new data to avoid entity tracking issues
    lf._data.detach(lf)
    lf._data = result.data
    result.data.attach(lf)
