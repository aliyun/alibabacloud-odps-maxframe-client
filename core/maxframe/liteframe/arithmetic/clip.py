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

import numpy as np
import pandas as pd
import pyarrow as pa

from maxframe.liteframe.arithmetic.core import (
    _BOOL,
    _build_fused_projection,
    _extract_projection_exprs,
    _resolve_column_expr,
    _root_source_key,
)
from maxframe.liteframe.core import LITEFRAME_TYPE
from maxframe.liteframe.expressions import (
    LiteFrameBaseExpr,
    LiteFrameGt,
    LiteFrameLiteral,
    LiteFrameLt,
    LiteFrameWhere,
    NamedExpr,
    SourceCompatibilityError,
    _infer_literal_dtype,
    _is_integer_arrow_dtype,
    _promote_numpy_dtypes,
)
from maxframe.utils import wrap_arrow_dtype


def _clip_expr(col_ref, col_dtype, lower, upper):
    """Build a LiteFrameWhere expression tree that clips *col_ref* to
    ``[lower, upper]``.

    *lower* / *upper* may be ``None``, raw scalars (wrapped in
    ``LiteFrameLiteral``), or resolved LiteFrameExpr operands.  At least one
    must be non-None.

    The result dtype is promoted from *col_dtype* and the bound operands'
    dtypes so that, e.g., clipping an int column with a float LiteFrame
    bound yields a float result (matching the physical values and pandas).
    """

    def _as_operand(v):
        if v is None or isinstance(v, LiteFrameBaseExpr):
            return v
        return LiteFrameLiteral(value=v, dtype=_infer_literal_dtype(v))

    lower = _as_operand(lower)
    upper = _as_operand(upper)

    # Normalize scalar bounds to match pandas value-based semantics:
    #  * NaN scalar → None (no bound — pandas keeps column dtype/values).
    #  * Integer-valued float scalar on an int column → int literal so the
    #    physical execution doesn't promote the result to float64 (e.g.
    #    int64 + 0.0 → int64 literal 0, not float64 literal 0.0).
    #  * Infinity is never integer-valued (np.isfinite guard before
    #    int(val) avoids OverflowError — review follow-up: int(col)+inf
    #    crashed at build time).
    def _normalize_scalar(bound):
        if not isinstance(bound, LiteFrameLiteral):
            return bound
        val = bound.value
        if isinstance(val, float) and np.isnan(val):
            return None
        if (
            _is_integer_arrow_dtype(col_dtype)
            and pa.types.is_floating(bound.dtype.pyarrow_dtype)
            and np.isfinite(val)
            and val == int(val)
        ):
            return LiteFrameLiteral(
                value=int(val), dtype=_infer_literal_dtype(int(val))
            )
        return bound

    lower = _normalize_scalar(lower)
    upper = _normalize_scalar(upper)
    # Result-dtype promotion (scalar bounds were normalized above: NaN →
    # None, integer-valued float on int col → int literal).  Remaining
    # scalar-on-int-col cases: non-integer/non-finite float → float64,
    # or integer exceeding the column's range → widen.
    result_dtype = col_dtype
    for bound in (lower, upper):
        if bound is None or bound.dtype is None:
            continue
        if isinstance(bound, LiteFrameLiteral):
            bound_pa = bound.dtype.pyarrow_dtype
            col_np = pd.api.types.pandas_dtype(result_dtype).numpy_dtype
            if _is_integer_arrow_dtype(result_dtype):
                if pa.types.is_floating(bound_pa):
                    result_dtype = _promote_numpy_dtypes(
                        result_dtype, bound.dtype, result_dtype
                    )
                else:
                    val = bound.value
                    if not (np.iinfo(col_np).min <= val <= np.iinfo(col_np).max):
                        # np.result_type preserves the column's
                        # signedness: uint8 + 1000 → uint16 (not int16),
                        # and uint8 + -5 → int16 (negative forces signed).
                        # The original min_scalar_type(probe) trick
                        # always returned signed dtypes, dropping
                        # unsignedness (review follow-up).
                        result_np = np.result_type(col_np, val)
                        result_dtype = wrap_arrow_dtype(pa.from_numpy_dtype(result_np))
            elif pa.types.is_floating(bound_pa):
                # float col + float scalar: keep the column dtype only
                # when the scalar is finitely representable in it; an
                # out-of-range finite scalar (float32 + 1e40) or a
                # non-finite scalar (inf) overflows to inf in the
                # column dtype, so promote to float64 (pandas does the
                # same — review follow-up: [inf,inf] silently appeared).
                if np.issubdtype(col_np, np.floating):
                    val = bound.value
                    finfo = np.finfo(col_np)
                    if not (np.isfinite(val) and abs(val) <= finfo.max):
                        result_dtype = _promote_numpy_dtypes(
                            result_dtype, bound.dtype, result_dtype
                        )
        elif isinstance(bound, LiteFrameBaseExpr):
            result_dtype = _promote_numpy_dtypes(
                result_dtype, bound.dtype, result_dtype
            )
    if lower is None and upper is None:
        # Both bounds were NaN (treated as absent) — return the column
        # unchanged (pandas: clip(lower=nan) is a no-op).
        return col_ref
    if lower is not None and upper is not None:
        # where(col < lower, lower, where(col > upper, upper, col))
        inner = LiteFrameWhere(
            condition=LiteFrameGt(left=col_ref, right=upper, dtype=_BOOL),
            true_value=upper,
            false_value=col_ref,
            dtype=result_dtype,
        )
        return LiteFrameWhere(
            condition=LiteFrameLt(left=col_ref, right=lower, dtype=_BOOL),
            true_value=lower,
            false_value=inner,
            dtype=result_dtype,
        )
    if lower is not None:
        # where(col < lower, lower, col)
        return LiteFrameWhere(
            condition=LiteFrameLt(left=col_ref, right=lower, dtype=_BOOL),
            true_value=lower,
            false_value=col_ref,
            dtype=result_dtype,
        )
    # upper is not None
    # where(col > upper, upper, col)
    return LiteFrameWhere(
        condition=LiteFrameGt(left=col_ref, right=upper, dtype=_BOOL),
        true_value=upper,
        false_value=col_ref,
        dtype=result_dtype,
    )


def _to_python_scalar(v):
    """Normalize NumPy scalars (np.int64, np.float64, …) to Python scalars
    so they pass the isinstance-based swap check and build valid literals.
    LiteFrame and array-like bounds are returned unchanged."""
    if isinstance(v, np.generic):
        return v.item()
    return v


def clip(lf, lower=None, upper=None):
    """Trim values at input threshold(s).

    Values below *lower* are set to *lower*; values above *upper* are set
    to *upper*.  At least one of *lower* / *upper* must be provided.

    Parameters
    ----------
    lower : scalar or LiteFrame, optional
        Minimum threshold.  A LiteFrame must derive from the same source as
        *lf* (per-column bounds are matched by column name); a scalar is
        broadcast to every column.
    upper : scalar or LiteFrame, optional
        Maximum threshold.  Same rules as *lower*.

    Returns
    -------
    LiteFrame
        A new LiteFrame with clipped values.

    See Also
    --------
    where : Replace values where a condition is False.
    between : Return boolean LiteFrame for values within a range.

    Examples
    --------
    >>> import maxframe.liteframe as mlf
    >>> lf = mlf.LiteFrame({'a': [1, 2, 3, 4, 5]})
    >>> result = lf.clip(2, 4)
    >>> result.execute()
       a
    0  2
    1  2
    2  3
    3  4
    4  4
    """
    if lower is None and upper is None:
        raise ValueError("One of 'lower' or 'upper' must be set")

    lower_lf = isinstance(lower, LITEFRAME_TYPE)
    upper_lf = isinstance(upper, LITEFRAME_TYPE)

    # Validate source compatibility for LiteFrame bounds — unrelated frames
    # cannot be aligned implicitly (same rule as where/mask).  Without this
    # guard the bound expressions would silently rebind to lf's columns by
    # name and produce wrong results.
    lf_root = _root_source_key(lf)
    if lower_lf and _root_source_key(lower) != lf_root:
        raise SourceCompatibilityError(lf_root, _root_source_key(lower))
    if upper_lf and _root_source_key(upper) != lf_root:
        raise SourceCompatibilityError(lf_root, _root_source_key(upper))

    # Normalize scalar bounds (NumPy scalars → Python scalars).  LiteFrame
    # bounds are left as-is for per-column expression extraction below.
    if not lower_lf:
        lower = _to_python_scalar(lower)
    if not upper_lf:
        upper = _to_python_scalar(upper)

    # pandas normalizes inverted scalar bounds by swapping them, so
    # clip(lower=3, upper=1) behaves like clip(lower=1, upper=3) rather
    # than clamping every value to the lower branch.  Only plain scalar
    # bounds are normalized; LiteFrame bounds cannot be reordered.
    if (
        not lower_lf
        and not upper_lf
        and isinstance(lower, (int, float))
        and isinstance(upper, (int, float))
        and lower > upper
    ):
        lower, upper = upper, lower

    lower_exprs = _extract_projection_exprs(lower) if lower_lf else None
    upper_exprs = _extract_projection_exprs(upper) if upper_lf else None

    def _resolve_bound(bound, bound_lf, bound_exprs, col_name, col_dtype):
        """Resolve a bound to a per-column operand for _clip_expr.

        A LiteFrame bound must provide a matching column name; a missing
        column raises KeyError rather than silently falling back to the
        column reference (which would make the bound a no-op).  A scalar
        bound is returned as-is; _clip_expr wraps it in a literal.
        """
        if bound is None:
            return None
        if bound_lf:
            if col_name not in bound_exprs:
                raise KeyError(
                    f"LiteFrame bound has no column named {col_name!r}; "
                    f"clip cannot align a bound that is missing the target "
                    f"column. Available: {list(bound_exprs)!r}"
                )
            return bound_exprs[col_name]
        return bound

    projections = []
    for col_name in lf.columns:
        col_dtype = lf.dtypes[col_name]
        col_ref = _resolve_column_expr(lf, col_name, col_dtype)
        lower_op = _resolve_bound(lower, lower_lf, lower_exprs, col_name, col_dtype)
        upper_op = _resolve_bound(upper, upper_lf, upper_exprs, col_name, col_dtype)
        expr = _clip_expr(col_ref, col_dtype, lower_op, upper_op)
        projections.append(NamedExpr(name=col_name, expr=expr))

    return _build_fused_projection(lf, projections)
