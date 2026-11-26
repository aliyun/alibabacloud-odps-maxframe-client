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
    _root_source_key,
)
from maxframe.liteframe.core import LITEFRAME_TYPE, LiteFrameData
from maxframe.liteframe.expressions import (
    LiteFrameColumn,
    LiteFrameExpr,
    LiteFrameLiteral,
    NamedExpr,
)
from maxframe.liteframe.operators.core import _alignment_id_of
from maxframe.liteframe.operators.merge import LiteFrameConcat
from maxframe.liteframe.operators.project import LiteFrameProjection
from maxframe.liteframe.utils import normalize_col_name


def _is_aligned(left, right):
    """True iff *left* and *right* are row-aligned (same non-None alignment_id)."""
    left_id = _alignment_id_of(left)
    right_id = _alignment_id_of(right)
    return left_id is not None and right_id is not None and left_id == right_id


def _build_aligned_setitem(lf, value, key):
    """Attach a single-column aligned *value* to *lf* as column *key*.

    Uses a shuffle-free horizontal concat: rename the value's column to
    *key* (a projection, which also drops the value's virtual range
    columns so they do not collide with *lf*'s), drop *key* from *lf*
    when it already exists, then ``concat(axis=1)``. The result stays
    row-aligned with *lf* (concat axis=1 inherits the shared id).
    """
    value_col = value.columns[0]
    value_dtype = value.dtypes[value_col]
    # Rename the value's single column to *key*; projecting also drops
    # the value's virtual range columns (they duplicate lf's).
    rename_proj = LiteFrameProjection(
        projections=[
            NamedExpr(name=key, expr=LiteFrameColumn(name=value_col, dtype=value_dtype))
        ]
    )
    value_renamed = rename_proj(value)

    # Drop an existing *key* column from lf so concat(axis=1) sees no
    # duplicated physical column names across inputs.
    if key in lf.columns:
        lf_projections = [
            _resolve_named_expr(lf, c, lf.dtypes[c]) for c in lf.columns if c != key
        ]
        lf_part = _build_fused_projection(lf, lf_projections)
    else:
        lf_part = lf

    return LiteFrameConcat(axis=1)([lf_part, value_renamed])


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
    # Only unwrap when the projection shares lf's root source (a same-frame
    # expression re-derivable from the source). A single-column projection
    # rooted at a different frame (e.g. a rank result) is foreign: it is
    # attached via the aligned-concat path below instead of being turned
    # into a (wrong) column reference against lf.
    from_liteframe_projection = (
        isinstance(value, LITEFRAME_TYPE)
        and isinstance(value.op, LiteFrameProjection)
        and len(value.op.projections) == 1
        and _root_source_key(value) == _root_source_key(lf)
    )
    if from_liteframe_projection:
        value = value.op.projections[0]

    # When *value* is a single-column LiteFrame that is row-aligned with
    # *lf* (same alignment_id), attach it with a shuffle-free horizontal
    # concat instead of a column reference / join. This is how a rank
    # result is set back onto its source frame: lf["rank"] = lf.rank(...).
    if (
        isinstance(value, LITEFRAME_TYPE)
        and not from_liteframe_projection
        and len(value.columns) == 1
        and _is_aligned(lf, value)
    ):
        result = _build_aligned_setitem(lf, value, key)
        lf._data.detach(lf)
        lf._data = result.data
        result.data.attach(lf)
        return

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
        if isinstance(value.op, LiteFrameProjection):
            # A foreign-rooted projection (e.g. lf.rechunk(...)["a"] + 1)
            # whose expression was not unwrapped above (different root
            # source, not aligned). Extract the inner expression and
            # rewrite its column refs through lf's projection so the
            # fused projection below is self-contained. This preserves
            # the operation graph (the +1) instead of silently
            # substituting a bare column reference.
            inner = value.op.projections[0]
            if isinstance(inner, NamedExpr):
                inner = inner.expr
            expr = inner.bind(lf.dtypes)
            existing = _extract_projection_exprs(lf)
            expr = _rewrite_column_refs(expr, existing)
        else:
            # Non-projection LiteFrame not row-aligned with *lf*: a bare
            # column reference into *lf* would silently bind *key* to itself
            # if col_name == key (discarding the UDF result).  Raise so the
            # caller re-aligns (placeback=True or manual re-split) first.
            col_name = value.columns[0]
            if col_name == key:
                raise ValueError(
                    f"Cannot assign unaligned LiteFrame column '{col_name}' "
                    f"to '{key}': different alignment_id. Use placeback=True "
                    f"or re-split/concat before assignment."
                )
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
