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

from maxframe.liteframe.arithmetic.core import (
    _build_fused_projection,
    _resolve_named_expr,
    _root_source_key,
    _root_source_lf,
)
from maxframe.liteframe.core import LITEFRAME_TYPE, LiteFrame
from maxframe.liteframe.expressions import (
    LiteFrameAnd,
    LiteFrameBinaryExpr,
    LiteFrameEq,
    LiteFrameExpr,
    LiteFrameGe,
    LiteFrameGt,
    LiteFrameInvert,
    LiteFrameLe,
    LiteFrameLt,
    LiteFrameNe,
    LiteFrameOr,
    NamedExpr,
    SourceCompatibilityError,
)
from maxframe.liteframe.operators.filter import LiteFrameFilter
from maxframe.liteframe.operators.project import LiteFrameProjection
from maxframe.liteframe.utils import normalize_col_name

# Expression types that produce boolean results and can be used as filter predicates
_BOOLEAN_EXPR_TYPES = (
    LiteFrameEq,
    LiteFrameNe,
    LiteFrameLt,
    LiteFrameLe,
    LiteFrameGt,
    LiteFrameGe,
    LiteFrameInvert,
)


def _is_boolean_expr(expr):
    """Check if an expression tree produces a boolean result."""
    if isinstance(expr, _BOOLEAN_EXPR_TYPES):
        return True
    # Recurse into NamedExpr
    if isinstance(expr, NamedExpr):
        return _is_boolean_expr(expr.expr)
    # Check binary expressions that contain boolean sub-expressions (e.g., And, Or)
    if isinstance(expr, LiteFrameBinaryExpr):
        if isinstance(expr, (LiteFrameAnd, LiteFrameOr)):
            return True
    return False


def _try_extract_filter_expr(mask_lf, source_lf):
    """Try to extract a LiteFrameExpr predicate from a single-column boolean mask.

    Returns a bound LiteFrameExpr if extraction succeeds, or None if the mask
    should be treated as a boolean LiteFrame mask (fallback path).

    Parameters
    ----------
    mask_lf : LiteFrame
        The boolean mask LiteFrame passed to __getitem__.
    source_lf : LiteFrame
        The source LiteFrame being filtered.

    Returns
    -------
    LiteFrameExpr or None
    """
    # Only extract from single-column masks
    if len(mask_lf.columns) != 1:
        return None

    # Only extract from LiteFrameProjection operators
    op = mask_lf.op
    if not isinstance(op, LiteFrameProjection):
        return None

    projections = op.projections
    if len(projections) != 1:
        return None

    proj = projections[0]
    if isinstance(proj, NamedExpr):
        expr = proj.expr
    elif isinstance(proj, LiteFrameExpr):
        expr = proj
    else:
        return None

    # Only extract if the expression is a boolean-producing expression
    if not _is_boolean_expr(expr):
        return None

    # Validate source compatibility: the mask and the source must derive from
    # the same root source, otherwise left[right["a"] > 5] would silently
    # bind right's expression against left's schema.
    mask_root = _root_source_key(mask_lf)
    source_root = _root_source_key(source_lf)
    if mask_root != source_root:
        raise SourceCompatibilityError(mask_root, source_root)

    # Bind the expression to resolve any unresolved dtypes
    bound_expr = expr.bind(source_lf.dtypes)
    return bound_expr


def liteframe_getitem(lf, key):
    """Column projection or boolean row filtering for LiteFrame."""
    if isinstance(key, slice):
        # LiteFrame uses a range index, so all slices with integer or no
        # bounds are positional (matching pandas behavior on RangeIndex).
        # Reject non-integer bounds (label-based slicing not supported).
        for edge in (key.start, key.stop):
            if edge is not None and not isinstance(edge, Integral):
                raise KeyError(f"Unsupported slice key: {key}")
        return lf.iloc[key]
    elif isinstance(key, str):
        # Single column projection
        key = normalize_col_name(key)
        projections = [_resolve_named_expr(lf, key, lf.dtypes[key])]
        return _build_fused_projection(lf, projections)
    elif isinstance(key, list):
        # Multiple column projection
        key = [normalize_col_name(k) for k in key]
        projections = [_resolve_named_expr(lf, k, lf.dtypes[k]) for k in key]
        return _build_fused_projection(lf, projections)
    elif isinstance(key, LITEFRAME_TYPE):
        # Boolean row filtering
        expr = _try_extract_filter_expr(key, lf)
        if expr is not None:
            # The fused predicate references root source columns, so the
            # filter must be applied to the root source.  If *lf* is itself a
            # projection, re-apply that projection on top of the filtered
            # root source so the output schema matches lf.
            root_source = _root_source_lf(lf)
            filter_op = LiteFrameFilter(predicate=expr)
            filtered = filter_op(LiteFrame(root_source))
            if root_source is lf._data:
                return filtered
            # lf is a projection — re-apply it on the filtered root source
            return lf.op(filtered)
        else:
            op = LiteFrameFilter(predicate=key)
            return op(lf)
    else:
        raise KeyError(f"Unsupported getitem key type: {type(key)}")
