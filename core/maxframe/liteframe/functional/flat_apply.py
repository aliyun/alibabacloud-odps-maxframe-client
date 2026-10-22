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

"""LiteFrame.flat_apply() -- row-expanding UDF application (operator-based)."""

import pandas as pd

from maxframe.liteframe.functional.type_infer import (
    _parse_explicit_dtypes,
    infer_output_dtypes,
)
from maxframe.liteframe.functional.utils import (
    build_flat_apply_output_order,
    build_functional_output,
    effective_groupby_input_dtypes,
    groupby_key_output_dtypes,
    pop_check_output_dtypes,
    wrap_inference_func,
)
from maxframe.liteframe.operators.functional import LiteFrameFunctional
from maxframe.liteframe.utils import normalize_col_name


def _normalize_subset(subset, visible_cols):
    """Normalize the ``subset`` argument to a list of validated column names.

    Returns the full visible column list when ``subset`` is ``None``.
    Raises ``KeyError`` for columns absent from the visible schema.
    """
    if subset is None:
        return list(visible_cols)
    if isinstance(subset, str):
        subset = [subset]
    else:
        subset = list(subset)
    # LiteFrame lowercases stored column names; normalize the caller's names so
    # case-insensitive selection (lf["Val"]) stays consistent with flat_apply,
    # and reject duplicate names after normalization (mirrors the other
    # selection APIs).  Without normalization subset=["Val"] would raise a
    # spurious KeyError against the lowercased schema.
    subset = [normalize_col_name(c) for c in subset]
    valid = set(visible_cols)
    missing = [c for c in subset if c not in valid]
    dupes = sorted({c for c in subset if subset.count(c) > 1})
    if dupes:
        raise ValueError(
            f"flat_apply subset columns must be unique after normalization; "
            f"duplicates: {dupes}"
        )
    if missing:
        raise KeyError(
            f"flat_apply subset column(s) {missing} not found in visible "
            f"columns {list(visible_cols)}"
        )
    return subset


def flat_apply(
    self,
    func,
    *,
    dtypes=None,
    skip_infer=False,
    input_frame_type="pandas",
    subset=None,
    args=(),
    **kwargs,
):
    """Apply ``func`` to each row, expanding it to 0-N output rows.

    ``flat_apply`` is the LiteFrame analogue of ``mf.flatmap``: ``func``
    receives one row (a ``pandas.Series`` for ``input_frame_type="pandas"``,
    a single-row ``polars.DataFrame`` for ``input_frame_type="polars"``) and
    returns / yields 0-N new rows or a new DataFrame.  Each returned row is
    aligned to the function-output columns (dict / named-Series by key/label,
    list / tuple positionally, scalar to a single column).

    Columns **not** in ``subset`` are *broadcast* (repeated for every output
    row produced from the same input row), mirroring ``pandas.DataFrame.
    explode``.  The function-output columns are inserted at the position
    where the first ``subset`` column originally located; broadcast columns
    keep their relative order around the func-out block.

    Parameters
    ----------
    func : callable or callable class
        Function taking one row and returning / yielding 0-N rows (a
        DataFrame, generator, list-like, dict, Series, or scalar) or a tuple
        of such results (multiple outputs).

        If ``func`` is a class, it is instantiated once per chunk,
        ``__call__`` is invoked per row, and ``close()`` (if defined)
        is always called at the end of the chunk.
    dtypes : dict, pd.Series, or list, optional
        Expected output dtypes for the **function-output** columns only (the
        broadcast columns inherit their input dtypes automatically).  A list
        of dicts/pd.Series declares multiple outputs.  If omitted, inferred
        from ``func``'s return annotation or a mock call.
    skip_infer : bool, default False
        If True, skip dtype inference and produce an untyped output.
    input_frame_type : {"pandas", "polars"}, default "pandas"
        Frame type passed to ``func``.
    subset : list-like of str, optional
        Columns passed to ``func``.  ``None`` (default) passes all visible
        columns (no broadcast columns, pure flatmap behaviour).  Columns not
        in ``subset`` are broadcast across the expanded output rows.
    args : tuple
        Extra positional arguments forwarded to ``func``.
    **kwargs
        Extra keyword arguments forwarded to ``func``.

    Returns
    -------
    LiteFrame or tuple of LiteFrame
        A new LiteFrame (or tuple of LiteFrames for multiple outputs) whose
        schema inserts the function-output columns at the position of the
        first ``subset`` column, with broadcast columns keeping their
        relative order.

    Notes
    -----
    The row count may change (1:N expansion), so -- like ``apply(axis=0)``
    and ``apply_chunk`` -- hidden columns are dropped from the UDF input and
    not carried into the output.  Users who need a hidden column to travel
    through ``flat_apply`` should ``select`` it first.  Range columns are
    materialized to physical ``int64`` before the UDF runs; a broadcast range
    column becomes a regular ``int64`` column in the output.

    See Also
    --------
    apply_chunk : Apply a function per execution chunk.

    Examples
    --------
    >>> import maxframe.liteframe as mlf
    >>> lf = mlf.LiteFrame({'a': [1, 2], 'b': ['x', 'y']})

    Expand each row into multiple rows (1:N), like ``explode``:

    >>> result = lf.flat_apply(lambda row: [{'n': row['a']}, {'n': row['a'] * 10}],
    ...                        subset=['a'], dtypes={'n': int})
    >>> result.execute()
        n  b
    0   1  x
    1  10  x
    2   2  y
    3  20  y

    Use ``subset`` to broadcast non-subset columns (``B`` is repeated for
    every expanded row):

    >>> result = lf.flat_apply(lambda row: [row['a'], row['a'] + 1],
    ...                        subset=['a'], dtypes={'n': int})
    >>> result.execute()
       n  b
    0  1  x
    1  2  x
    2  2  y
    3  3  y

    Return ``None`` to drop an input row (0 output rows):

    >>> result = lf.flat_apply(
    ...     lambda row: [{'n': row['a']}] if row['a'] > 1 else None,
    ...     subset=['a'], dtypes={'n': int})
    >>> result.execute()
       n  b
    0  2  y
    """
    kwargs = dict(kwargs) if kwargs else {}
    check_output_dtypes = pop_check_output_dtypes(kwargs)
    hidden = sorted(self._hidden_columns)
    inp = self.drop(hidden) if hidden else self
    return _build_flat_apply_op(
        func,
        dtypes=dtypes,
        skip_infer=skip_infer,
        input_frame_type=input_frame_type,
        subset=subset,
        args=args,
        kwargs=kwargs,
        check_output_dtypes=check_output_dtypes,
        effective_dtypes=self.dtypes,
        inp=inp,
        groupby_params=None,
    )


def _build_flat_apply_op(
    func,
    *,
    dtypes,
    skip_infer,
    input_frame_type,
    subset,
    args,
    kwargs,
    check_output_dtypes,
    effective_dtypes,
    inp,
    groupby_params,
):
    """Shared op construction for ``flat_apply`` and ``groupby_flat_apply``.

    ``effective_dtypes`` is the schema the UDF sees (the visible columns for
    the non-groupby path; the groupby-adjusted columns -- key-excluded /
    selected -- for the groupby path).  ``subset`` is normalized against it;
    the remaining effective columns are broadcast.  The full output schema
    inserts the function-output columns at the position where the first
    subset column originally located (broadcast cols keep their relative
    order around the func-out block), and is what the operator exposes and
    the executor validates/aligns against.
    """
    visible_cols = list(effective_dtypes.index)
    subset_cols = _normalize_subset(subset, visible_cols)
    subset_set = set(subset_cols)
    # For groupby flat_apply, exclude the grouping keys from the broadcast
    # schema: the group key is governed solely by the ``group_keys`` contract
    # (prepended by ``groupby_key_output_dtypes`` when True, omitted when
    # False), never by the broadcast path -- mirroring the SPE/DPE runtime
    # exclusion.  The key may still be a subset column (explicit UDF input);
    # only its broadcast role is removed.
    by_cols = []
    if groupby_params is not None:
        by = groupby_params.get("by")
        by_cols = by if isinstance(by, (list, tuple)) else [by]
    broadcast_cols = [
        c for c in visible_cols if c not in subset_set and c not in by_cols
    ]
    broadcast_dtypes = effective_dtypes[broadcast_cols]
    subset_dtypes = effective_dtypes[subset_cols]

    # Infer the *function-output* dtypes (broadcast columns are interleaved
    # afterwards to form the full output schema).
    if dtypes is not None:
        func_dtypes_list = _parse_explicit_dtypes(dtypes)
    elif not skip_infer:
        infer_func = wrap_inference_func(
            func, "flat_apply", input_frame_type, args, kwargs
        )
        func_dtypes_list = infer_output_dtypes(
            infer_func,
            subset_dtypes,
            dtypes=None,
            skip_infer=False,
            args=(),
            kwargs={},
            input_frame_type=input_frame_type,
        )
    else:
        func_dtypes_list = None

    if func_dtypes_list is None:
        # skip_infer: untyped output (column count is unknown because the
        # function-output schema is unknown).
        op = LiteFrameFunctional(
            func=func,
            mode="flat_apply",
            subset=subset_cols,
            input_frame_type=input_frame_type,
            check_output_dtypes=check_output_dtypes,
            output_dtypes=None,
            args=args,
            kwargs=kwargs if kwargs else None,
            groupby_params=groupby_params,
        )
        return build_functional_output(op, [inp], None)

    # A function-output column name must not collide with a broadcast column
    # name (the two blocks are interleaved, so a clash would be ambiguous and
    # the UDF cannot produce a broadcast column).
    for fd in func_dtypes_list:
        conflict = [c for c in fd.index if c in broadcast_cols]
        if conflict:
            raise ValueError(
                f"flat_apply output column(s) {conflict} collide with "
                f"broadcast (non-subset) column(s) {broadcast_cols}; the "
                f"function-output columns must not reuse a broadcast column "
                f"name. Rename the output column or include it in `subset`."
            )

    # Full output schema: func-out cols inserted at the position of the
    # first subset column; broadcast cols keep their relative order.
    # ``exclude_cols=by_cols`` keeps the groupby keys out of the broadcast
    # schema (their position still anchors the func-out block when
    # ``subset`` includes a key); the keys are prepended separately by
    # ``groupby_key_output_dtypes`` when ``group_keys=True``.
    full_dtypes_list = [
        pd.concat([broadcast_dtypes, fd]).reindex(
            build_flat_apply_output_order(
                visible_cols,
                subset_cols,
                list(fd.index),
                exclude_cols=by_cols,
            )
        )
        for fd in func_dtypes_list
    ]

    # groupby flat_apply follows the apply-family key contract: the grouping
    # keys are prepended to every output when ``group_keys=True`` (default)
    # and omitted when ``False`` -- matching the executors.
    if groupby_params is not None and groupby_params.get("group_keys", True):
        full_dtypes_list = groupby_key_output_dtypes(
            inp, groupby_params, full_dtypes_list
        )

    op = LiteFrameFunctional(
        func=func,
        mode="flat_apply",
        subset=subset_cols,
        input_frame_type=input_frame_type,
        check_output_dtypes=check_output_dtypes,
        output_dtypes=full_dtypes_list,
        args=args,
        kwargs=kwargs if kwargs else None,
        groupby_params=groupby_params,
    )
    return build_functional_output(op, [inp], full_dtypes_list)


def groupby_flat_apply(
    self,
    func,
    *,
    dtypes=None,
    skip_infer=False,
    input_frame_type="pandas",
    subset=None,
    args=(),
    **kwargs,
):
    """Apply ``func`` to each row within each group, expanding it to 0-N rows.

    The groupby analogue of :func:`flat_apply`: the data is partitioned by the
    group keys, then ``func`` is applied per row *within each group* (not per
    group), returning / yielding 0-N rows.  Non-subset columns are broadcast
    (repeated per output row) just like the non-groupby variant, and the
    per-group expanded frames are concatenated across groups in group-iteration
    order.

    The effective columns (the schema the UDF sees) follow groupby semantics:
    groupby key columns are excluded by default (pandas 3.0), and an explicit
    ``groupby('k')[selection]`` restricts the UDF input to ``selection``.
    ``subset`` is normalized against those effective columns; the rest are
    broadcast.

    Parameters
    ----------
    func : callable or callable class
        Function taking one row and returning / yielding 0-N rows (a
        DataFrame, generator, list-like, dict, Series, or scalar) or a tuple
        of such results (multiple outputs).

        If ``func`` is a class, it is instantiated once per group,
        ``__call__`` is invoked per row, and ``close()`` (if defined)
        is always called at the end of each group.
    dtypes : dict, pd.Series, or list, optional
        Expected output dtypes for the **function-output** columns only.  A
        list of dicts/pd.Series declares multiple outputs.
    skip_infer : bool, default False
        If True, skip dtype inference and produce an untyped output.
    input_frame_type : {"pandas", "polars"}, default "pandas"
        Frame type passed to ``func``.
    subset : list-like of str, optional
        Columns passed to ``func`` (relative to the effective groupby
        columns).  Columns not in ``subset`` are broadcast.
    args : tuple
        Extra positional arguments forwarded to ``func``.
    **kwargs
        Extra keyword arguments forwarded to ``func``.

    Returns
    -------
    LiteFrame or tuple of LiteFrame
        A new LiteFrame (or tuple of LiteFrames for multiple outputs) whose
        schema inserts the function-output columns at the position of the
        first ``subset`` column, with broadcast columns keeping their
        relative order; group keys are prepended when ``group_keys=True``.

    Notes
    -----
    The row count may change, so -- like ``groupby.apply`` -- hidden columns
    are dropped from the UDF input and not carried into the output.  A
    callable-class UDF is instantiated fresh per group and its ``end`` flag
    fires on the last row of each group.  Range columns follow the groupby
    path (they are dropped by the groupby shuffle, like other groupby
    functional ops).

    See Also
    --------
    flat_apply : Apply a function per row, expanding to 0-N rows.
    groupby_apply : Apply a function to each group as a whole frame.

    Examples
    --------
    >>> import maxframe.liteframe as mlf
    >>> lf = mlf.LiteFrame({'key': ['a', 'a', 'b'], 'val': [1, 2, 3]})

    Expand each row within its group (the group key ``key`` is excluded
    from the UDF input by default, so it is broadcast):

    >>> result = lf.groupby('key').flat_apply(
    ...     lambda row: [{'n': row['val']}, {'n': row['val'] * 10}],
    ...     dtypes={'n': int})
    >>> result.execute()
      key   n
    0   a   1
    1   a  10
    2   a   2
    3   a  20
    4   b   3
    5   b  30
    """
    kwargs = dict(kwargs) if kwargs else {}
    check_output_dtypes = pop_check_output_dtypes(kwargs)
    effective_dtypes = effective_groupby_input_dtypes(
        self._liteframe, self.groupby_params
    )
    hidden = sorted(self._liteframe._hidden_columns)
    inp = self._liteframe.drop(hidden) if hidden else self._liteframe
    return _build_flat_apply_op(
        func,
        dtypes=dtypes,
        skip_infer=skip_infer,
        input_frame_type=input_frame_type,
        subset=subset,
        args=args,
        kwargs=kwargs,
        check_output_dtypes=check_output_dtypes,
        effective_dtypes=effective_dtypes,
        inp=inp,
        groupby_params=self.groupby_params,
    )
