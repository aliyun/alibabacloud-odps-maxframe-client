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

from maxframe.core.entity import get_tileable_data
from maxframe.liteframe.core import LITEFRAME_TYPE, LiteFrame
from maxframe.liteframe.datatypes import (
    infer_decimal_arithmetic_dtype,
    infer_dtype_cached,
)
from maxframe.liteframe.expressions import (
    LiteFrameBinaryExpr,
    LiteFrameBinUfuncExpr,
    LiteFrameColumn,
    LiteFrameExpr,
    LiteFrameInvert,
    LiteFrameIsNA,
    LiteFrameLiteral,
    LiteFrameLogicalNot,
    LiteFrameLShift,
    LiteFrameNaryExpr,
    LiteFrameRShift,
    LiteFrameUnaryExpr,
    LiteFrameUnaryUfuncExpr,
    LiteFrameWhere,
    NamedExpr,
    SourceCompatibilityError,
    _infer_literal_dtype,
    _infer_ufunc_result_dtype,
)
from maxframe.liteframe.operators.project import (
    LiteFrameProjection,
    _append_hidden_projections,
)
from maxframe.utils import wrap_arrow_dtype

# Shared boolean dtype used by comparison/logical expressions.
_BOOL = wrap_arrow_dtype(pa.bool_())

# Cache for _extract_projection_exprs: LiteFrameData -> dict[col_name, LiteFrameExpr]
_projection_expr_cache = weakref.WeakKeyDictionary()


def _extract_projection_exprs(lf):
    """Extract the per-column expressions from a LiteFrame's projection operator.

    Returns a dict mapping column name to the inner ``LiteFrameExpr``.
    """
    data = get_tileable_data(lf)
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
    data = get_tileable_data(lf)
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
    A LiteFrameColumn whose .name differs from col_name (e.g. after rename)
    is also wrapped so the output column name is preserved.
    """
    expr = _resolve_column_expr(lf, col_name, col_dtype)
    if isinstance(expr, NamedExpr):
        return expr
    if isinstance(expr, LiteFrameColumn) and expr.name == col_name:
        return expr
    # Pass col_dtype so that UDF projections (LiteFrameUDFExpr) and other
    # expressions without a resolved .dtype still carry the correct output
    # dtype after fusion.
    return NamedExpr(name=col_name, expr=expr, dtype=col_dtype)


def _rewrite_column_refs(expr, existing_exprs, expanding=frozenset()):
    """Rewrite LiteFrameColumn leaves in an expression tree.

    Returns a new expression tree where each ``LiteFrameColumn`` whose ``name``
    appears in *existing_exprs* is replaced by the corresponding inner
    expression.  This is used when a user-provided expression (e.g.
    ``col("a") + 1``) must be evaluated against the root source rather than
    against a projection output — the column reference is replaced by the
    projection's inner expression so the final fused projection is
    self-contained.

    Each column's expression is expanded at most once per path (*expanding*
    tracks the columns currently being inlined).  A column whose expression
    derives from itself (e.g. ``price := str.replace(col("price"))``) keeps
    its inner reference as a bare column: it refers to the **root source**
    column, not the projection output, and re-expanding it would duplicate the
    expression (``transform(transform(col("price")))``) and crash on type
    mismatches (a str op applied to an already-numeric column: e2e
    test_yunqi_demo ``InvalidOperationError: expected String type, got: f64``).

    If *existing_exprs* is empty (non-projection source), *expr* is returned
    unchanged.
    """
    if not existing_exprs:
        return expr
    if isinstance(expr, LiteFrameColumn):
        replacement = existing_exprs.get(expr.name)
        if replacement is None or expr.name in expanding:
            # Either not a projection-output reference, or the reference lies
            # inside the (already inlined) expression of a self-derived column
            # and therefore names the root source column — keep it as-is.
            return expr
        return _rewrite_column_refs(
            replacement, existing_exprs, expanding | {expr.name}
        )
    if isinstance(expr, LiteFrameLiteral):
        return expr
    if isinstance(expr, LiteFrameBinaryExpr):
        kwargs = dict(
            left=_rewrite_column_refs(expr.left, existing_exprs, expanding),
            right=_rewrite_column_refs(expr.right, existing_exprs, expanding),
            dtype=expr.dtype,
        )
        # Preserve extra fields on subclasses (e.g. ufunc_name)
        for attr in ("ufunc_name",):
            if hasattr(expr, attr):
                kwargs[attr] = getattr(expr, attr)
        return type(expr)(**kwargs)
    if isinstance(expr, LiteFrameUnaryExpr):
        new_expr = expr.copy()
        new_expr.operand = _rewrite_column_refs(expr.operand, existing_exprs, expanding)
        return new_expr
    if isinstance(expr, LiteFrameWhere):
        return type(expr)(
            condition=_rewrite_column_refs(expr.condition, existing_exprs, expanding),
            true_value=_rewrite_column_refs(expr.true_value, existing_exprs, expanding),
            false_value=_rewrite_column_refs(
                expr.false_value, existing_exprs, expanding
            ),
            dtype=expr.dtype,
        )
    if isinstance(expr, LiteFrameNaryExpr):
        return type(expr)(
            operands=[
                _rewrite_column_refs(o, existing_exprs, expanding)
                for o in expr.operands
            ],
            dtype=expr.dtype,
        )
    if isinstance(expr, NamedExpr):
        return type(expr)(
            name=expr.name,
            expr=_rewrite_column_refs(expr.expr, existing_exprs, expanding),
            dtype=expr.dtype,
        )
    # Fallback: return as-is (e.g. accessor expressions with extra fields)
    return expr


def _needs_materialize_range_cols(lf):
    """Check if any projection in the input chain requires materialization.

    Returns True if any predecessor's range_analysis has an entry with
    is_affine=False, references_range=True.
    """
    data = get_tileable_data(lf)
    while isinstance(getattr(data, "op", None), LiteFrameProjection):
        ra = getattr(data.op, "range_analysis", None)
        if ra:
            for result in ra.values():
                if result.references_range and not result.is_affine:
                    return True
        inputs = data.inputs
        if not inputs:
            break
        data = inputs[0]
    return False


def _build_fused_projection(lf, projections):
    """Create a LiteFrameProjection rooted at the source of *lf*.

    Appends hidden column pass-throughs from *lf*, then builds and executes
    a LiteFrameProjection whose input is the root source of *lf* so that
    fused expression trees resolve correctly.

    The analyzer in ``__call__`` runs on the fused expression tree to
    populate ``range_analysis``. If any predecessor in the chain requires
    materialization, the fused tree's analyzer will detect the same
    condition.
    """
    _append_hidden_projections(lf, projections)
    op = LiteFrameProjection(projections=projections)
    root_source = _root_source_lf(lf)
    return op(LiteFrame(root_source))


def _build_binary_projection(lf, expr_class, other, reverse=False, **kwargs):
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
    **kwargs
        Additional keyword arguments passed to the expression constructor
        (e.g. ``ufunc_name`` for :class:`LiteFrameBinUfuncExpr`).
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

    # LiteFrame-vs-LiteFrame pairing follows the numpy/polars model: output
    # takes the left operand's columns, right pairs positionally, a single
    # right column broadcasts; mismatched counts raise.  (lf + lf is the
    # positional case where positions match names.)
    other_is_lf = isinstance(other, LITEFRAME_TYPE)
    if other_is_lf:
        n_left = len(lf.columns)
        n_right = len(other.columns)
        if n_right == 1:
            # Broadcast: pair every left column with the sole right column.
            right_names = [other.columns[0]] * n_left
        elif n_right == n_left:
            # Positional: left[i] with right[i].
            right_names = list(other.columns)
        else:
            raise ValueError(
                f"Cannot align LiteFrames with {n_left} and {n_right} columns "
                f"for binary operation; right operand must have 1 column "
                f"(broadcast) or {n_left} columns (positional)."
            )

    for i, col_name in enumerate(lf.columns):
        col_dtype = lf.dtypes[col_name]

        if other_is_lf:
            left = _resolve_column_expr(lf, col_name, col_dtype)
            right_name = right_names[i]
            right = _resolve_column_expr(other, right_name, other.dtypes[right_name])
        else:
            lit = LiteFrameLiteral(value=other, dtype=_infer_literal_dtype(other))
            col = _resolve_column_expr(lf, col_name, col_dtype)
            if reverse:
                left, right = lit, col
            else:
                left, right = col, lit

        if issubclass(expr_class, LiteFrameBinUfuncExpr):
            result_dtype = _infer_ufunc_result_dtype(
                kwargs.get("ufunc_name"), left.dtype, right.dtype
            )
        else:
            result_dtype = _infer_result_dtype(left.dtype, right.dtype, expr_class)
        expr = expr_class(left=left, right=right, dtype=result_dtype, **kwargs)

        projections.append(NamedExpr(name=col_name, expr=expr))

    return _build_fused_projection(lf, projections)


def _build_unary_projection(lf, expr_class, **kwargs):
    """Build a Projection with unary expressions for all columns."""
    projections = []

    # Extract existing projection expressions from the input LiteFrame
    # so that chains like ~(lf["a"] > 50) produce (col("a") > 50).not_()
    # instead of col("a").not_()
    existing_exprs = _extract_projection_exprs(lf)

    is_ufunc = issubclass(expr_class, LiteFrameUnaryUfuncExpr)
    ufunc_name = kwargs.get("ufunc_name")

    for col_name in lf.columns:
        col_dtype = lf.dtypes[col_name]
        operand = existing_exprs.get(
            col_name, LiteFrameColumn(name=col_name, dtype=col_dtype)
        )
        if is_ufunc:
            result_dtype = _infer_ufunc_result_dtype(ufunc_name, col_dtype)
        else:
            result_dtype = _infer_unary_result_dtype(col_dtype, expr_class)
        expr = expr_class(operand=operand, dtype=result_dtype, **kwargs)
        projections.append(NamedExpr(name=col_name, expr=expr))

    return _build_fused_projection(lf, projections)


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
    if isinstance(expr_class, type) and issubclass(expr_class, LiteFrameBinUfuncExpr):
        # ufunc_name is not available here; bind() will re-infer.
        return left_dtype
    # Bitwise shift preserves the left operand's integer dtype. pandas Series
    # does not implement << / >> for the mock-based inference, so short-circuit
    # to the left dtype instead of falling back to string.
    if isinstance(expr_class, type) and issubclass(
        expr_class, (LiteFrameLShift, LiteFrameRShift)
    ):
        return left_dtype
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


def _is_bool_dtype(dtype):
    """Return True if *dtype* is an Arrow-backed boolean dtype."""
    pa_type = getattr(dtype, "pyarrow_dtype", None)
    try:
        return pa_type is not None and pa.types.is_boolean(pa_type)
    except Exception:
        return False


def _infer_unary_result_dtype(input_dtype, expr_class):
    """Infer result dtype for unary expression."""
    if isinstance(expr_class, type) and issubclass(expr_class, LiteFrameIsNA):
        return _BOOL
    if isinstance(expr_class, type) and issubclass(expr_class, LiteFrameLogicalNot):
        # np.logical_not always returns booleans for any input dtype.
        return _BOOL
    if isinstance(expr_class, type) and issubclass(expr_class, LiteFrameInvert):
        # Bitwise NOT: bool -> bool (logical not), integer -> preserve dtype.
        if _is_bool_dtype(input_dtype):
            return _BOOL
        return input_dtype
    return input_dtype
