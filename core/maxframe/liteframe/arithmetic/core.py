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

import weakref

import pyarrow as pa

from maxframe.liteframe.core import LITEFRAME_TYPE, LiteFrame
from maxframe.liteframe.datatypes import (
    infer_decimal_arithmetic_dtype,
    infer_dtype_cached,
)
from maxframe.liteframe.expressions import (
    LiteFrameBinaryExpr,
    LiteFrameColumn,
    LiteFrameExpr,
    LiteFrameInvert,
    LiteFrameIsNA,
    LiteFrameLiteral,
    LiteFrameNaryExpr,
    LiteFrameUnaryExpr,
    LiteFrameWhere,
    NamedExpr,
    SourceCompatibilityError,
    _infer_literal_dtype,
)
from maxframe.liteframe.operators.project import (
    LiteFrameProjection,
    _append_hidden_projections,
)
from maxframe.utils import wrap_arrow_dtype

# Cache for _extract_projection_exprs: LiteFrameData -> dict[col_name, LiteFrameExpr]
_projection_expr_cache = weakref.WeakKeyDictionary()


def _extract_projection_exprs(lf):
    """Extract the per-column expressions from a LiteFrame's projection operator.

    Returns a dict mapping column name to the inner ``LiteFrameExpr``.
    When the input is not a ``LiteFrameProjection``, returns an empty dict.
    """
    data = lf._data if isinstance(lf, LiteFrame) else lf
    cached = _projection_expr_cache.get(data)
    if cached is not None:
        return cached
    op = getattr(lf, "op", None)
    if not isinstance(op, LiteFrameProjection):
        return {}
    result = {}
    for proj in op.projections:
        if isinstance(proj, NamedExpr):
            result[proj.name] = proj.expr
        elif isinstance(proj, LiteFrameExpr):
            # Bare expression (e.g. LiteFrameColumn) — use column name if available
            name = getattr(proj, "name", None)
            if name is not None:
                result[name] = proj
    _projection_expr_cache[data] = result
    return result


def _root_source_lf(lf):
    """Walk back through projection inputs to find the root source LiteFrame.

    Stops at the first tileable whose op is not a LiteFrameProjection.
    If the input itself is not a projection, returns it unchanged.
    """
    data = lf._data
    while isinstance(getattr(data, "op", None), LiteFrameProjection):
        inputs = data.inputs
        if not inputs:
            break
        data = inputs[0]
    return data


def _root_source_key(lf):
    """Walk back through projection inputs to find the root source key."""
    return _root_source_lf(lf).key


def _resolve_column_expr(lf, col_name, col_dtype):
    """Return the expression for a column, fusing from projection if possible.

    If lf is a LiteFrameProjection, extract the inner expression for col_name.
    Otherwise, create a fresh LiteFrameColumn.
    """
    if isinstance(lf.op, LiteFrameProjection):
        existing = _extract_projection_exprs(lf)
        return existing[col_name]
    return LiteFrameColumn(name=col_name, dtype=col_dtype)


def _resolve_named_expr(lf, col_name, col_dtype):
    """Like _resolve_column_expr but always returns a NamedExpr or LiteFrameColumn.

    Bare expressions (e.g. LiteFrameAdd) are wrapped in NamedExpr so that
    LiteFrameProjection.__call__ can extract the output name via _projection_name.
    """
    expr = _resolve_column_expr(lf, col_name, col_dtype)
    if isinstance(expr, (NamedExpr, LiteFrameColumn)):
        return expr
    return NamedExpr(name=col_name, expr=expr)


def _rewrite_column_refs(expr, existing_exprs):
    """Rewrite LiteFrameColumn leaves in an expression tree.

    Returns a new expression tree where each ``LiteFrameColumn`` whose ``name``
    appears in *existing_exprs* is replaced by a copy of the corresponding
    inner expression.  This is used when a user-provided expression (e.g.
    ``col("a") + 1``) must be evaluated against the root source rather than
    against a projection output — the column reference is replaced by the
    projection's inner expression so the final fused projection is
    self-contained.

    If *existing_exprs* is empty (non-projection source), *expr* is returned
    unchanged.
    """
    if not existing_exprs:
        return expr
    if isinstance(expr, LiteFrameColumn):
        return existing_exprs.get(expr.name, expr)
    if isinstance(expr, LiteFrameLiteral):
        return expr
    if isinstance(expr, LiteFrameBinaryExpr):
        return type(expr)(
            left=_rewrite_column_refs(expr.left, existing_exprs),
            right=_rewrite_column_refs(expr.right, existing_exprs),
            dtype=expr.dtype,
        )
    if isinstance(expr, LiteFrameUnaryExpr):
        new_expr = expr.copy()
        new_expr.operand = _rewrite_column_refs(expr.operand, existing_exprs)
        return new_expr
    if isinstance(expr, LiteFrameWhere):
        return type(expr)(
            condition=_rewrite_column_refs(expr.condition, existing_exprs),
            true_value=_rewrite_column_refs(expr.true_value, existing_exprs),
            false_value=_rewrite_column_refs(expr.false_value, existing_exprs),
            dtype=expr.dtype,
        )
    if isinstance(expr, LiteFrameNaryExpr):
        return type(expr)(
            operands=[_rewrite_column_refs(o, existing_exprs) for o in expr.operands],
            dtype=expr.dtype,
        )
    if isinstance(expr, NamedExpr):
        return type(expr)(
            name=expr.name,
            expr=_rewrite_column_refs(expr.expr, existing_exprs),
            dtype=expr.dtype,
        )
    # Fallback: return as-is (e.g. accessor expressions with extra fields)
    return expr


def _needs_materialize_range_cols(lf):
    """Check if any projection in the input chain has materialize_range_cols=True.

    When fusing through a projection that materializes range columns, the
    fused projection must also set ``materialize_range_cols=True`` so that
    the range columns are materialized at the root source.
    """
    data = lf._data
    while isinstance(getattr(data, "op", None), LiteFrameProjection):
        if getattr(data.op, "materialize_range_cols", False):
            return True
        inputs = data.inputs
        if not inputs:
            break
        data = inputs[0]
    return False


def _build_fused_projection(lf, projections, materialize_range_cols=False):
    """Create a LiteFrameProjection rooted at the source of *lf*.

    Appends hidden column pass-throughs from *lf*, then builds and executes
    a LiteFrameProjection whose input is the root source of *lf* so that
    fused expression trees resolve correctly.

    If any projection in the input chain has ``materialize_range_cols=True``,
    or *materialize_range_cols* is passed as True, the flag is set on the
    fused projection so that range columns are materialized at the root
    source.
    """
    _append_hidden_projections(lf, projections)
    materialize = materialize_range_cols or _needs_materialize_range_cols(lf)
    op = LiteFrameProjection(
        projections=projections,
        materialize_range_cols=materialize,
    )
    root_source = _root_source_lf(lf)
    return op(LiteFrame(root_source))


def _build_binary_projection(lf, expr_class, other, reverse=False):
    """Build a Projection with binary expressions for all columns.

    Parameters
    ----------
    lf : LiteFrame
        Left operand LiteFrame.
    expr_class : type
        The LiteFrameBinaryExpr subclass.
    other : LiteFrame or scalar
        Right operand.
    reverse : bool, default False
        When True, swap the scalar operand to the left side so that
        reverse ops (e.g. ``1.5 - lf``) produce
        ``LiteFrameSub(left=Literal(1.5), right=Column("a"))``.
        Ignored when *other* is a LiteFrame.
    """
    projections = []

    # Pre-validate source compatibility: for LiteFrame operands, both must
    # derive from the same root source. We walk back through projection
    # input chains to find the original source key, so that chained ops
    # like (lf + 10) / lf are accepted (both trace back to the same lf).
    if isinstance(other, LITEFRAME_TYPE):
        left_root = _root_source_key(lf)
        right_root = _root_source_key(other)
        if left_root != right_root:
            raise SourceCompatibilityError(left_root, right_root)

    for col_name in lf.columns:
        col_dtype = lf.dtypes[col_name]

        if isinstance(other, LITEFRAME_TYPE):
            left = _resolve_column_expr(lf, col_name, col_dtype)
            right = _resolve_column_expr(other, col_name, other.dtypes[col_name])
        else:
            lit = LiteFrameLiteral(value=other, dtype=_infer_literal_dtype(other))
            col = _resolve_column_expr(lf, col_name, col_dtype)
            if reverse:
                left, right = lit, col
            else:
                left, right = col, lit

        result_dtype = _infer_result_dtype(left.dtype, right.dtype, expr_class)
        expr = expr_class(left=left, right=right, dtype=result_dtype)

        projections.append(NamedExpr(name=col_name, expr=expr))

    # When either operand is a LiteFrame whose projection chain materializes
    # range columns, the fused binary projection must also materialize them
    # so that range columns are physical in both inputs.
    materialize_range_cols = _needs_materialize_range_cols(lf)
    if isinstance(other, LITEFRAME_TYPE):
        materialize_range_cols = (
            materialize_range_cols or _needs_materialize_range_cols(other)
        )

    # TODO: Some binary operators (e.g. FillNA) transform visible range
    # columns and need materialize_range_cols=True, but others (e.g. Add)
    # can keep the range virtual. Postpone per-operator handling to a
    # later patch.

    return _build_fused_projection(
        lf, projections, materialize_range_cols=materialize_range_cols
    )


def _build_unary_projection(lf, expr_class, **kwargs):
    """Build a Projection with unary expressions for all columns."""
    projections = []

    # Extract existing projection expressions from the input LiteFrame
    # so that chains like ~(lf["a"] > 50) produce (col("a") > 50).not_()
    # instead of col("a").not_()
    existing_exprs = _extract_projection_exprs(lf)

    # Range columns are virtual Int64; any unary transform (e.g. isna,
    # invert) generates pl.col(RANGE_COL_NAME).<op>(), which needs a
    # physical column to reference.
    range_col_names = (
        set(lf.frame_metadata.range_columns.keys())
        if lf.frame_metadata and lf.frame_metadata.range_columns
        else set()
    )
    materialize_range_cols = bool(range_col_names & set(lf.columns))

    for col_name in lf.columns:
        col_dtype = lf.dtypes[col_name]
        operand = existing_exprs.get(
            col_name, LiteFrameColumn(name=col_name, dtype=col_dtype)
        )
        result_dtype = _infer_unary_result_dtype(col_dtype, expr_class)
        expr = expr_class(operand=operand, dtype=result_dtype, **kwargs)
        projections.append(NamedExpr(name=col_name, expr=expr))

    return _build_fused_projection(
        lf, projections, materialize_range_cols=materialize_range_cols
    )


def _infer_result_dtype(left_dtype, right_dtype, expr_class):
    """Infer result dtype for binary expression.

    Uses Hive decimal formula for decimal operands, falling back to
    mock-based inference via :func:`infer_dtype_cached` for other types.

    Parameters
    ----------
    left_dtype : dtype
        Left operand dtype.
    right_dtype : dtype
        Right operand dtype.
    expr_class : type
        The LiteFrameBinaryExpr subclass (must have ``_operator``
        classproperty).

    Returns
    -------
    dtype
        The inferred result dtype, falling back to *left_dtype* on error.
    """
    try:
        opr = expr_class._operator
    except (AttributeError, NotImplementedError):
        return left_dtype
    # Try Hive decimal formula first — returns None if not applicable
    result = infer_decimal_arithmetic_dtype(left_dtype, right_dtype, opr)
    if result is not None:
        return result
    # Fall back to mock-based inference
    try:
        return infer_dtype_cached(left_dtype, right_dtype, opr)
    except Exception:
        return left_dtype


def _infer_unary_result_dtype(input_dtype, expr_class):
    """Infer result dtype for unary expression."""
    if isinstance(expr_class, type) and issubclass(expr_class, LiteFrameIsNA):
        return wrap_arrow_dtype(pa.bool_())
    if isinstance(expr_class, type) and issubclass(expr_class, LiteFrameInvert):
        return wrap_arrow_dtype(pa.bool_())
    return input_dtype
