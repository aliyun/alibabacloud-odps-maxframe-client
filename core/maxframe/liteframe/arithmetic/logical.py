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
    _build_unary_projection,
    _root_source_key,
)
from maxframe.liteframe.core import LITEFRAME_TYPE
from maxframe.liteframe.expressions import (
    LiteFrameAnd,
    LiteFrameEq,
    LiteFrameExpr,
    LiteFrameGe,
    LiteFrameGt,
    LiteFrameInvert,
    LiteFrameLe,
    LiteFrameLt,
    LiteFrameNe,
    LiteFrameOr,
    LiteFrameXor,
    NamedExpr,
    SourceCompatibilityError,
)
from maxframe.liteframe.operators.project import LiteFrameProjection
from maxframe.utils import wrap_arrow_dtype

# Expression types that produce boolean results
_BOOLEAN_EXPR_TYPES = (
    LiteFrameEq,
    LiteFrameNe,
    LiteFrameLt,
    LiteFrameLe,
    LiteFrameGt,
    LiteFrameGe,
    LiteFrameInvert,
    LiteFrameAnd,
    LiteFrameOr,
)


def _is_boolean_expr(expr):
    """Check if an expression tree produces a boolean result."""
    if isinstance(expr, _BOOLEAN_EXPR_TYPES):
        return True
    if isinstance(expr, NamedExpr):
        return _is_boolean_expr(expr.expr)
    return False


def _try_cross_column_logical(lf, other, expr_class):
    """Handle & and | between single-column boolean LiteFrames with different column names.

    When both operands are single-column boolean LiteFrames from different columns,
    extract their inner expressions and combine them with the logical operator.

    Returns a LiteFrame if cross-column handling applies, or None to fall back
    to _build_binary_projection.
    """
    if not isinstance(other, LITEFRAME_TYPE):
        return None

    # Pre-validate source compatibility — both must share the same root source
    left_root = _root_source_key(lf)
    right_root = _root_source_key(other)
    if left_root != right_root:
        raise SourceCompatibilityError(left_root, right_root)

    # Both must be single-column
    if len(lf.columns) != 1 or len(other.columns) != 1:
        return None

    lf_col = lf.columns[0]
    other_col = other.columns[0]

    # Same column name: let _build_binary_projection handle it
    if lf_col == other_col:
        return None

    # Both must be LiteFrameProjection operators (to extract expressions)
    if not isinstance(lf.op, LiteFrameProjection) or not isinstance(
        other.op, LiteFrameProjection
    ):
        return None

    # Extract inner expressions
    lf_proj = lf.op.projections[0]
    other_proj = other.op.projections[0]

    if isinstance(lf_proj, NamedExpr):
        lf_expr = lf_proj.expr
    elif isinstance(lf_proj, LiteFrameExpr):
        lf_expr = lf_proj
    else:
        return None

    if isinstance(other_proj, NamedExpr):
        other_expr = other_proj.expr
    elif isinstance(other_proj, LiteFrameExpr):
        other_expr = other_proj
    else:
        return None

    # Both expressions must produce boolean results
    if not _is_boolean_expr(lf_expr) or not _is_boolean_expr(other_expr):
        return None

    # Combine expressions
    combined = expr_class(
        left=lf_expr, right=other_expr, dtype=wrap_arrow_dtype(pa.bool_())
    )

    # Build a single-column projection with the combined expression
    out_col = lf_col
    projection = NamedExpr(name=out_col, expr=combined)
    return _build_fused_projection(lf, [projection])


def and_(lf, other):
    result = _try_cross_column_logical(lf, other, LiteFrameAnd)
    if result is not None:
        return result
    return _build_binary_projection(lf, LiteFrameAnd, other)


def or_(lf, other):
    result = _try_cross_column_logical(lf, other, LiteFrameOr)
    if result is not None:
        return result
    return _build_binary_projection(lf, LiteFrameOr, other)


def xor(lf, other):
    return _build_binary_projection(lf, LiteFrameXor, other)


def invert(lf):
    return _build_unary_projection(lf, LiteFrameInvert)
