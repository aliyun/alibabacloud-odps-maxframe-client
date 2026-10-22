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

from maxframe.core.entity import get_tileable_data
from maxframe.liteframe.arithmetic.core import (
    _build_fused_projection,
    _root_source_key,
    _root_source_lf,
)
from maxframe.liteframe.core import LITEFRAME_TYPE, LiteFrame
from maxframe.liteframe.expressions import (
    LiteFrameExpr,
    LiteFrameInvert,
    NamedExpr,
    SourceCompatibilityError,
    _is_bool_arrow_dtype,
    is_always_boolean_expr,
)
from maxframe.liteframe.indexing.select import (
    _promote_hidden_in_metadata,
    _resolve_column_for_select,
)
from maxframe.liteframe.operators.filter import LiteFrameFilter
from maxframe.liteframe.operators.project import LiteFrameProjection
from maxframe.liteframe.utils import normalize_col_name


# Boolean predicate detection for filter masks. ``LiteFrameInvert`` (``~``)
# is dtype-aware bitwise NOT (bool -> bool, int -> int), so it is accepted
# only when its resolved dtype is boolean; all other boolean-producing
# expression classes are marked with ``_boolean_result`` in
# ``expressions.py`` and detected via ``is_always_boolean_expr``.
def _is_boolean_expr(expr):
    """Check if an expression tree produces a boolean result.

    ``LiteFrameInvert`` (``~``) is accepted only when its resolved dtype is
    boolean: ``~bool_col`` is a valid boolean predicate, but ``~int_col``
    is a bitwise (integer) expression and must not be fused into a filter
    predicate. Comparisons, ``IsInExpr``, ``LogicalNot``, and ``And``/``Or``
    are always boolean (see ``is_always_boolean_expr``).
    """
    if isinstance(expr, LiteFrameInvert):
        return _is_bool_arrow_dtype(getattr(expr, "dtype", None))
    if isinstance(expr, NamedExpr):
        return _is_boolean_expr(expr.expr)
    return is_always_boolean_expr(expr)


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
    """Column projection or boolean row filtering for LiteFrame.

    Supports several key types: a string selects a single column, a list of
    strings selects multiple columns, a slice selects rows positionally
    (delegating to ``iloc``), and a single-column boolean LiteFrame filters
    rows.  Single-column selection returns a LiteFrame (not a Series).

    Parameters
    ----------
    key : str, list of str, slice, or LiteFrame
        Selection key.  A string projects one column; a list of strings
        projects multiple columns; an integer slice selects rows by
        position; a boolean LiteFrame (single column) filters rows where
        the mask is ``True``.

    Returns
    -------
    LiteFrame
        Projected or filtered LiteFrame.  Column projection preserves row
        order; boolean filtering preserves the relative order of surviving
        rows.

    See Also
    --------
    LiteFrame.select : Explicit column selection.
    LiteFrame.iloc : Positional indexer for rows and columns.
    """
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
        projections = [_resolve_column_for_select(lf, key)]
        result = _build_fused_projection(lf, projections)
        promoted = {key} if key in lf._hidden_columns else set()
        _promote_hidden_in_metadata(result, promoted)
        return result
    elif isinstance(key, list):
        # Multiple column projection
        key = [normalize_col_name(k) for k in key]
        projections = [_resolve_column_for_select(lf, k) for k in key]
        result = _build_fused_projection(lf, projections)
        promoted = {k for k in key if k in lf._hidden_columns}
        _promote_hidden_in_metadata(result, promoted)
        return result
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
            lf_data = get_tileable_data(lf)
            if root_source is lf_data:
                return filtered
            # lf is a projection — re-apply it on the filtered root source
            return lf.op(filtered)
        else:
            op = LiteFrameFilter(predicate=key)
            return op(lf)
    else:
        raise KeyError(f"Unsupported getitem key type: {type(key)}")
