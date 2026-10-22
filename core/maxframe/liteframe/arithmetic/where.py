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

from maxframe.liteframe.arithmetic.core import (
    _build_fused_projection,
    _extract_projection_exprs,
    _resolve_column_expr,
    _root_source_key,
)
from maxframe.liteframe.core import LITEFRAME_TYPE
from maxframe.liteframe.expressions import (
    LiteFrameColumn,
    LiteFrameLiteral,
    LiteFrameWhere,
    NamedExpr,
    SourceCompatibilityError,
    _infer_literal_dtype,
    _infer_result_dtype,
)


def where(lf, cond, other=np.nan):
    """Replace values where *cond* is False with *other*.

    Equivalent to ``pandas.DataFrame.where``: keep original values
    where *cond* is True, replace with *other* where False.

    Parameters
    ----------
    cond : LiteFrame
        Boolean LiteFrame (e.g. from ``lf > 2``) with the same columns.
    other : scalar or LiteFrame, default np.nan
        Value to substitute where *cond* is False.

    Returns
    -------
    LiteFrame

    See Also
    --------
    mask : Replace values where a condition is True.
    case_when : Replace values where conditions are True, in order.

    Examples
    --------
    >>> import maxframe.liteframe as mlf
    >>> lf = mlf.LiteFrame({'a': [1, 2, 3]})
    >>> result = lf.where(lf > 2, 0)
    >>> result.execute()
       a
    0  0
    1  0
    2  3
    """
    return _where_mask(lf, cond, other, replace_true=False)


def mask(lf, cond, other=np.nan):
    """Replace values where *cond* is True with *other*.

    Equivalent to ``pandas.DataFrame.mask``: replace original values
    where *cond* is True with *other*, keep where False.

    Parameters
    ----------
    cond : LiteFrame
        Boolean LiteFrame (e.g. from ``lf > 2``) with the same columns.
    other : scalar or LiteFrame, default np.nan
        Value to substitute where *cond* is True.

    Returns
    -------
    LiteFrame

    See Also
    --------
    where : Replace values where a condition is False.
    case_when : Replace values where conditions are True, in order.

    Examples
    --------
    >>> import maxframe.liteframe as mlf
    >>> lf = mlf.LiteFrame({'a': [1, 2, 3]})
    >>> result = lf.mask(lf > 2, 0)
    >>> result.execute()
       a
    0  1
    1  2
    2  0
    """
    return _where_mask(lf, cond, other, replace_true=True)


def _where_mask(lf, cond, other, replace_true):
    """Build a per-column Where projection implementing where/mask.

    *replace_true*=False → where (keep where cond True, replace where False).
    *replace_true*=True  → mask  (replace where cond True, keep where False).
    """
    # Validate source compatibility — cond and any LiteFrame-typed other
    # must share lf's root source.  Unrelated frames cannot be aligned
    # implicitly (design doc: "Unrelated LiteFrames cannot be combined
    # implicitly"); without this guard the cond/replacement expressions
    # would silently rebind to lf's columns by name and produce wrong results.
    lf_root = _root_source_key(lf)
    if _root_source_key(cond) != lf_root:
        raise SourceCompatibilityError(lf_root, _root_source_key(cond))
    if isinstance(other, LITEFRAME_TYPE) and _root_source_key(other) != lf_root:
        raise SourceCompatibilityError(lf_root, _root_source_key(other))

    cond_exprs = _extract_projection_exprs(cond)
    other_is_lf = isinstance(other, LITEFRAME_TYPE)
    other_exprs = _extract_projection_exprs(other) if other_is_lf else None

    projections = []
    for col_name in lf.columns:
        col_dtype = lf.dtypes[col_name]
        col_ref = _resolve_column_expr(lf, col_name, col_dtype)
        cond_expr = cond_exprs.get(
            col_name, LiteFrameColumn(name=col_name, dtype=col_dtype)
        )

        if other_is_lf:
            other_val = other_exprs.get(
                col_name, LiteFrameColumn(name=col_name, dtype=col_dtype)
            )
        else:
            other_val = LiteFrameLiteral(value=other, dtype=_infer_literal_dtype(other))

        if replace_true:
            # mask: Where(cond, other, col)
            true_val, false_val = other_val, col_ref
        else:
            # where: Where(cond, col, other)
            true_val, false_val = col_ref, other_val

        result_dtype = _infer_result_dtype(
            true_val.dtype, false_val.dtype, LiteFrameWhere
        )
        expr = LiteFrameWhere(
            condition=cond_expr,
            true_value=true_val,
            false_value=false_val,
            dtype=result_dtype,
        )
        projections.append(NamedExpr(name=col_name, expr=expr))

    return _build_fused_projection(lf, projections)


def case_when(lf, caselist):
    """Replace values where conditions are True, evaluated in order.

    Mirrors ``pandas.Series.case_when``: each ``(cond, replacement)`` pair
    is evaluated sequentially; the first condition that is True determines
    the replacement value for that row.  Rows where no condition is True
    keep their original value.

    Both *cond* and *replacement* may be callables; when callable they are
    invoked with *lf* and should return a LiteFrame (for *cond*) or a
    LiteFrame / scalar (for *replacement*), matching the pandas contract
    where callables are computed on the Series itself.

    Parameters
    ----------
    caselist : list of (condition, replacement)
        Ordered list of ``(condition, replacement)`` pairs.  *condition*
        is a boolean LiteFrame, or a callable returning one.  *replacement*
        is a scalar, a LiteFrame, or a callable returning either.

    Returns
    -------
    LiteFrame
        A LiteFrame with the same columns as *lf*.

    See Also
    --------
    where : Replace values where a condition is False.
    mask : Replace values where a condition is True.

    Examples
    --------
    >>> import maxframe.liteframe as mlf
    >>> lf = mlf.LiteFrame({'a': [1, 2, 3, 4]})
    >>> result = lf.case_when([(lf < 2, 0), (lf > 3, 10)])
    >>> result.execute()
        a
    0   0
    1   2
    2   3
    3  10
    """
    if not isinstance(caselist, list):
        raise TypeError(
            f"The caselist argument should be a list; instead got {type(caselist)}"
        )
    if not caselist:
        raise ValueError(
            "provide at least one boolean condition, "
            "with a corresponding replacement."
        )
    for i, entry in enumerate(caselist):
        if not isinstance(entry, tuple) or len(entry) != 2:
            raise ValueError(
                f"Argument {i} must be a (condition, replacement) tuple "
                f"of length 2; instead got {entry!r}."
            )

    # Resolve callables once: callable(cond) -> cond(lf), callable(repl) -> repl(lf).
    resolved = [
        (c(lf) if callable(c) else c, r(lf) if callable(r) else r) for c, r in caselist
    ]

    projections = []
    for col_name in lf.columns:
        col_dtype = lf.dtypes[col_name]
        col_ref = _resolve_column_expr(lf, col_name, col_dtype)

        # Collect per-case condition + replacement expressions for this column.
        case_exprs = []
        for cond, replacement in resolved:
            cond_exprs = _extract_projection_exprs(cond)
            cond_expr = cond_exprs.get(
                col_name, LiteFrameColumn(name=col_name, dtype=col_dtype)
            )

            if isinstance(replacement, LITEFRAME_TYPE):
                repl_exprs = _extract_projection_exprs(replacement)
                repl_val = repl_exprs.get(
                    col_name, LiteFrameColumn(name=col_name, dtype=col_dtype)
                )
            else:
                repl_val = LiteFrameLiteral(
                    value=replacement, dtype=_infer_literal_dtype(replacement)
                )
            case_exprs.append((cond_expr, repl_val))

        # Build nested Where from the inside out:
        #   Where(c0, r0, Where(c1, r1, col))
        # First match wins, fall through to the original column.
        result_dtype = col_ref.dtype
        expr = col_ref
        for cond_expr, repl_val in reversed(case_exprs):
            result_dtype = _infer_result_dtype(
                repl_val.dtype, result_dtype, LiteFrameWhere
            )
            expr = LiteFrameWhere(
                condition=cond_expr,
                true_value=repl_val,
                false_value=expr,
                dtype=result_dtype,
            )

        projections.append(NamedExpr(name=col_name, expr=expr))

    return _build_fused_projection(lf, projections)
