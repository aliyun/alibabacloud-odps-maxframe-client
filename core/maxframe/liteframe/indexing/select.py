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

from maxframe.liteframe.arithmetic.core import (
    _build_fused_projection,
    _extract_projection_exprs,
    _resolve_named_expr,
    _rewrite_column_refs,
    _root_source_lf,
)
from maxframe.liteframe.core import LITEFRAME_TYPE, FrameMetadata, LiteFrame
from maxframe.liteframe.expressions import (
    LiteFrameColumn,
    LiteFrameExpr,
    LiteFrameLiteral,
    NamedExpr,
    _infer_literal_dtype,
)
from maxframe.liteframe.operators.project import LiteFrameProjection
from maxframe.liteframe.utils import normalize_col_name, validate_errors


def _resolve_column_for_select(lf, c):
    """Resolve a column name to a projection, promoting hidden columns."""
    if c in lf.dtypes.index:
        return _resolve_named_expr(lf, c, lf.dtypes[c])
    if c in lf._physical_dtypes.index:
        # Promote hidden column to visible
        return LiteFrameColumn(name=c, dtype=lf._physical_dtypes[c])
    raise KeyError(f"Column '{c}' not found")


def _promote_hidden_in_metadata(result, promoted):
    """Remove promoted columns from the result's hidden_columns set."""
    if not promoted or not result._data.frame_metadata:
        return
    old_meta = result._data.frame_metadata
    new_hidden = [h for h in (old_meta.hidden_columns or []) if h not in promoted]
    new_hidden = new_hidden or None
    if old_meta.range_columns or new_hidden:
        result._data.frame_metadata = FrameMetadata(
            range_columns=old_meta.range_columns,
            hidden_columns=new_hidden,
        )
    else:
        result._data.frame_metadata = None
    result._data.__dict__.pop("dtypes", None)
    result._data.__dict__.pop("columns", None)


def select(lf, columns):
    """Explicit column selection."""
    columns = [normalize_col_name(c) for c in columns]
    projections = [_resolve_column_for_select(lf, c) for c in columns]
    result = _build_fused_projection(lf, projections)
    promoted = {c for c in columns if c in lf._hidden_columns}
    _promote_hidden_in_metadata(result, promoted)
    return result


def select_expr(lf, *exprs):
    """Select columns using LiteFrameExpr objects, binding dtypes from lf."""
    existing_exprs = _extract_projection_exprs(lf)
    bound = []
    for i, expr in enumerate(exprs):
        bound_expr = expr.bind(lf.dtypes)
        # Capture the logical output name before rewriting column references.
        # _rewrite_column_refs inlines the root-source expression, which may
        # carry a different .name (e.g. after rename); the output column must
        # still use the name the user selected.
        output_name = getattr(bound_expr, "name", None) or str(i)
        # Rewrite column references through lf's projection so the
        # expression is self-contained and safe for later fusion.
        bound_expr = _rewrite_column_refs(bound_expr, existing_exprs)
        if isinstance(bound_expr, NamedExpr):
            bound.append(bound_expr)
        elif isinstance(bound_expr, LiteFrameColumn) and bound_expr.name == output_name:
            bound.append(bound_expr)
        else:
            bound.append(NamedExpr(name=output_name, expr=bound_expr))
    _reject_duplicate_output_names(bound)
    return _build_fused_projection(lf, bound)


def _reject_duplicate_output_names(projections):
    """Reject duplicate final projection names before schema construction.

    A positional fallback name (``str(i)`` for an anonymous expression) can
    collide with an explicitly selected integer-string column (e.g.
    ``col("1")`` alongside ``col("a") + 1`` at position 1).  Name-keyed
    dicts in ``LiteFrameProjection`` metadata / SPE would silently overwrite
    one entry, while DPE emits duplicate Polars aliases and fails.  Enforce
    uniqueness here so the collision is rejected consistently and early.
    """
    seen = {}
    for i, proj in enumerate(projections):
        name = getattr(proj, "name", None)
        if name is None:
            continue
        name = normalize_col_name(name)
        if name in seen:
            raise ValueError(
                f"Duplicate output column name {name!r} in select(): "
                f"position {seen[name]} and position {i} both resolve to it. "
                f"Use rename() or an explicit NamedExpr to disambiguate."
            )
        seen[name] = i


def select_dispatch(self, *args):
    """Dispatch to select or select_expr based on argument types.

    If the first argument is a :class:`LiteFrameExpr`, delegates to
    :func:`select_expr` to select computed columns; otherwise delegates to
    :func:`select` with the column names.  A single string argument is
    treated as one column; multiple arguments are collected into a list.

    Parameters
    ----------
    args : LiteFrameExpr or str or list of str
        Column expressions or column names to select.  Mixes are not
        supported: if the first argument is an expression, all arguments
        are treated as expressions.

    Returns
    -------
    LiteFrame
        LiteFrame containing only the selected columns, preserving row
        order.
    """
    if args and isinstance(args[0], LiteFrameExpr):
        return select_expr(self, *args)
    # Normalize: single string → list, multiple args → list
    if len(args) == 1:
        columns = [args[0]] if isinstance(args[0], str) else args[0]
    else:
        columns = list(args)
    return select(self, columns)


def drop(lf, columns, errors="raise"):
    """Drop columns from LiteFrame.

    Removes the specified columns and returns a LiteFrame over the
    remaining columns.  Hidden columns named in *columns* are also
    dropped.  Row order is preserved.

    Parameters
    ----------
    columns : str or list of str
        Column name(s) to drop.
    errors : {"raise", "ignore"}, default "raise"
        When ``"raise"``, a ``KeyError`` is raised for unknown columns;
        when ``"ignore"``, unknown columns are silently skipped.

    Returns
    -------
    LiteFrame
        LiteFrame with the specified columns removed.

    See Also
    --------
    LiteFrame.select : Explicit column selection.

    Examples
    --------
    >>> import maxframe.liteframe as mlf
    >>> lf = mlf.LiteFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    >>> result = lf.drop(columns=['a'])
    >>> result.execute()
       b
    0  4
    1  5
    2  6
    """
    validate_errors(errors, allow_coerce=False)
    if isinstance(columns, str):
        columns = [columns]
    columns = [normalize_col_name(c) for c in columns]
    columns_set = set(columns)

    # Separate into visible drops and hidden drops
    visible_drops = columns_set & set(lf.dtypes.index)
    hidden_drops = columns_set & lf._hidden_columns
    unknown = columns_set - visible_drops - hidden_drops

    if errors == "raise" and unknown:
        raise KeyError(f"Column(s) {unknown} not found")

    # Build projection over remaining visible columns
    remaining_visible = [c for c in lf.columns if c not in visible_drops]
    projections = [_resolve_named_expr(lf, c, lf.dtypes[c]) for c in remaining_visible]

    # Append surviving hidden columns (hidden_drops are simply omitted)
    surviving_hidden = lf._hidden_columns - hidden_drops
    for hc in sorted(surviving_hidden):
        projections.append(LiteFrameColumn(name=hc, dtype=lf._physical_dtypes[hc]))

    # Build projection directly — hidden columns already handled above,
    # so we bypass _build_fused_projection which would re-add all hidden columns.
    op = LiteFrameProjection(projections=projections)
    root_source = _root_source_lf(lf)
    return op(LiteFrame(root_source))


def rename(lf, columns):
    """Rename columns in LiteFrame.

    Applies a mapping of old column names to new names; columns not in the
    mapping keep their current name.  Row order and data are unchanged.

    Parameters
    ----------
    columns : dict
        Mapping ``{old_name: new_name}`` describing the renames to apply.

    Returns
    -------
    LiteFrame
        LiteFrame with renamed columns.

    See Also
    --------
    LiteFrame.add_prefix : Prefix all column labels.
    LiteFrame.add_suffix : Suffix all column labels.

    Examples
    --------
    >>> import maxframe.liteframe as mlf
    >>> lf = mlf.LiteFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    >>> result = lf.rename(columns={'a': 'x'})
    >>> result.execute()
       x  b
    0  1  4
    1  2  5
    2  3  6
    """
    # Each existing column is carried forward as a single-output projection.
    # Multi-output UDF projections (transform/transform_chunk) are never
    # reached bare here -- they are always wrapped in a
    # LiteFrameMultiColumnSubExpr (which carries .dtype) per output column.
    # Only single-output map() UDFs surface as a bare LiteFrameUDFExpr,
    # whose .dtype property exposes the single output dtype.  So a bare
    # multi-output UDF expr (which has no single .dtype) is never accessed
    # through rename() / assign().
    # Normalize rename dict keys (old names) and values (new names)
    normalized_columns = {
        normalize_col_name(k): normalize_col_name(v) for k, v in columns.items()
    }
    existing_exprs = _extract_projection_exprs(lf)
    projections = []
    for c in lf.columns:
        col_expr = existing_exprs.get(c, LiteFrameColumn(name=c, dtype=lf.dtypes[c]))
        if c in normalized_columns:
            projections.append(NamedExpr(name=normalized_columns[c], expr=col_expr))
        else:
            projections.append(_resolve_named_expr(lf, c, lf.dtypes[c]))

    return _build_fused_projection(lf, projections)


def add_prefix(lf, prefix):
    """Prefix all column labels with *prefix*.

    Parameters
    ----------
    prefix : str
        String to prepend to every column name.

    Returns
    -------
    LiteFrame
        LiteFrame with prefixed column names.

    See Also
    --------
    LiteFrame.add_suffix : Suffix all column labels.
    LiteFrame.rename : Rename columns in LiteFrame.

    Examples
    --------
    >>> import maxframe.liteframe as mlf
    >>> lf = mlf.LiteFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    >>> result = lf.add_prefix('col_')
    >>> result.execute()
       col_a  col_b
    0      1      4
    1      2      5
    2      3      6
    """
    return rename(lf, {c: f"{prefix}{c}" for c in lf.columns})


def add_suffix(lf, suffix):
    """Suffix all column labels with *suffix*.

    Parameters
    ----------
    suffix : str
        String to append to every column name.

    Returns
    -------
    LiteFrame
        LiteFrame with suffixed column names.

    See Also
    --------
    LiteFrame.add_prefix : Prefix all column labels.
    LiteFrame.rename : Rename columns in LiteFrame.

    Examples
    --------
    >>> import maxframe.liteframe as mlf
    >>> lf = mlf.LiteFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    >>> result = lf.add_suffix('_val')
    >>> result.execute()
       a_val  b_val
    0      1      4
    1      2      5
    2      3      6
    """
    return rename(lf, {c: f"{c}{suffix}" for c in lf.columns})


def _resolve_assign_value(name, value, lf):
    """Resolve an assign value to a (projection, dtype) pair.

    Returns (LiteFrameExpr, dtype) where the projection's output name is `name`.
    """
    if callable(value):
        value = value(lf)

    # Unwrap a single-column LiteFrame projection into its LiteFrameExpr,
    # so that assign(col=df[col].astype(...)) preserves the operation graph.
    if (
        isinstance(value, LITEFRAME_TYPE)
        and isinstance(value.op, LiteFrameProjection)
        and len(value.op.projections) == 1
    ):
        value = value.op.projections[0]

    if isinstance(value, LiteFrameExpr):
        bound = value.bind(lf.dtypes)
        # Rewrite column references through lf's projection so the
        # expression is self-contained and safe for fusion.
        existing = _extract_projection_exprs(lf)
        bound = _rewrite_column_refs(bound, existing)
        # The bound expression may itself be a NamedExpr carrying a different
        # output name (e.g. from .select("a")); unwrap it so the assign target
        # name is applied uniformly.  Wrap in a NamedExpr named `name` directly
        # (rather than .rename()) so this also covers a bare LiteFrameUDFExpr
        # (map mode), which has no .rename() method.
        inner = bound.expr if isinstance(bound, NamedExpr) else bound
        dtype = inner.dtype
        return NamedExpr(name=normalize_col_name(name), expr=inner, dtype=dtype), dtype

    if isinstance(value, LITEFRAME_TYPE):
        if len(value.columns) != 1:
            raise ValueError(
                f"Cannot assign a LiteFrame with {len(value.columns)} columns "
                f"to a single column '{name}'"
            )
        col_name = value.columns[0]
        expr = LiteFrameColumn(name=col_name, dtype=value.dtypes[col_name])
        return expr.rename(name), value.dtypes.iloc[0]

    # numpy temporal dtypes lack .pyarrow_dtype; infer Arrow directly so
    # the temporal codegen branch can read .pyarrow_dtype.unit.
    dtype = (
        _infer_literal_dtype(value)
        if isinstance(value, (np.datetime64, np.timedelta64))
        else pd.Series([value]).dtype
    )
    expr = LiteFrameLiteral(value=value, dtype=dtype)
    return expr.rename(name), dtype


def assign(lf, **kwargs):
    """Add new columns to LiteFrame.

    Existing columns whose names appear in *kwargs* are replaced in place
    (preserving their position); new columns are appended in insertion
    order.  Each value may be a scalar, a callable receiving the frame, a
    :class:`LiteFrameExpr`, or a single-column LiteFrame.

    Parameters
    ----------
    **kwargs
        Column assignments as ``name=value`` pairs.  Names are normalized
        and used as the output column names.

    Returns
    -------
    LiteFrame
        LiteFrame with the new or replaced columns.  Row order is
        preserved.

    See Also
    --------
    LiteFrame.select : Explicit column selection.
    LiteFrame.eval : Evaluate a string expression to add columns.

    Examples
    --------
    >>> import maxframe.liteframe as mlf
    >>> lf = mlf.LiteFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    >>> result = lf.assign(c=lambda x: x['a'] + x['b'])
    >>> result.execute()
       a  b  c
    0  1  4  5
    1  2  5  7
    2  3  6  9

    Assign with a column expression:

    >>> from maxframe.liteframe import col
    >>> result = lf.assign(c=col('a') * 2)
    >>> result.execute()
       a  b  c
    0  1  4  2
    1  2  5  4
    2  3  6  6
    """
    # Like rename(), only single-output projections flow through here; a
    # bare multi-output UDF expr (no single .dtype) is never accessed -- see
    # the comment in rename() and LiteFrameUDFExpr.dtype.
    normalized_kwargs = {normalize_col_name(k): v for k, v in kwargs.items()}
    existing_names = set(lf.columns)

    # Build projections in original column order, replacing overwritten
    # columns in-place so they keep their position; new columns are appended.
    projections = []

    for c in lf.columns:
        if c in normalized_kwargs:
            proj, dtype = _resolve_assign_value(c, normalized_kwargs[c], lf)
        else:
            proj = _resolve_named_expr(lf, c, lf.dtypes[c])
        projections.append(proj)

    for name, value in normalized_kwargs.items():
        if name not in existing_names:
            proj, dtype = _resolve_assign_value(name, value, lf)
            projections.append(proj)

    return _build_fused_projection(lf, projections)
