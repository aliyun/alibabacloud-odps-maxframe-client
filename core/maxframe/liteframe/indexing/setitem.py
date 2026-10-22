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
    _resolve_named_expr,
    _rewrite_column_refs,
)
from maxframe.liteframe.core import LITEFRAME_TYPE, LiteFrameData
from maxframe.liteframe.expressions import (
    LiteFrameColumn,
    LiteFrameExpr,
    LiteFrameLiteral,
    NamedExpr,
)
from maxframe.liteframe.operators.project import LiteFrameProjection
from maxframe.liteframe.utils import normalize_col_name


def liteframe_setitem(lf, key, value):
    """Add or replace a column in LiteFrame, or set rows by position.

    When *key* is a column name, the column is added or replaced.  The
    *value* may be a scalar, a :class:`LiteFrameExpr`, or a single-column
    LiteFrame; column references inside an expression are rewritten through
    the frame's projection so the assignment is self-contained.  When *key*
    is an integer slice, rows are set positionally via ``iloc``.

    The mutation is applied in place on the owning LiteFrame.

    Parameters
    ----------
    key : str or slice
        Column name to add or replace, or an integer slice selecting rows
        to set by position.
    value : scalar, LiteFrameExpr, or LiteFrame
        Value to assign.  A LiteFrame must have exactly one column.  When
        *key* is a slice, only scalar values are supported.

    Returns
    -------
    None
        The mutation is applied in place; nothing is returned.
    """
    if isinstance(lf, LiteFrameData):
        raise NotImplementedError(
            "item assignment is not supported on a LiteFrameData; mutate via "
            "the owning LiteFrame instead"
        )
    if isinstance(key, slice):
        edge = key.start if key.start is not None else key.stop
        if isinstance(edge, Integral):
            lf.iloc[key] = value
            return
        raise KeyError(f"Unsupported slice key: {key}")

    key = normalize_col_name(key)

    # Unwrap a single-column LiteFrame projection into its LiteFrameExpr,
    # so that lf["c"] = lf["a"] + 1 preserves the operation graph.
    from_liteframe_projection = (
        isinstance(value, LITEFRAME_TYPE)
        and isinstance(value.op, LiteFrameProjection)
        and len(value.op.projections) == 1
    )
    if from_liteframe_projection:
        value = value.op.projections[0]

    if isinstance(value, LiteFrameExpr):
        # Bind to resolve dtypes; if already a NamedExpr, use its inner expr
        # so the final NamedExpr(name=key, ...) wrapping below is correct.
        inner = value.expr if isinstance(value, NamedExpr) else value
        expr = inner.bind(lf.dtypes)
        # A projection-built expression (``lf["c"] = lf["a"] + 1``) is already
        # rooted at the root source: its column references name source
        # columns, so the fused projection below is self-contained.  Rewriting
        # it again against ``lf``'s projection would re-inline an
        # already-inlined self-derived column (``price := str.replace(price)``)
        # and duplicate its transform — ``transform(transform(col("price")))``
        # — crashing on a str op applied to the already-numeric result
        # (``InvalidOperationError: expected String type, got: f64``).
        if not from_liteframe_projection:
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

    # Build projections: all existing columns (preserving rename aliases)
    # plus the new named expression.
    projections = [
        _resolve_named_expr(lf, c, lf.dtypes[c]) for c in lf.columns if c != key
    ]
    projections.append(NamedExpr(name=key, expr=expr))

    result = _build_fused_projection(lf, projections)

    # Swap the underlying data on the owning LiteFrame.  LiteFrameData is
    # rejected above — it is not directly mutable; callers mutate via the
    # LiteFrame entity.
    lf._data.detach(lf)
    lf._data = result.data
    result.data.attach(lf)
