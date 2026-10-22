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

"""LiteFrame.apply() -- operator-based (LiteFrameFunctional) UDF application."""

import numpy as np

from maxframe.liteframe.core import FrameMetadata
from maxframe.liteframe.functional.utils import (
    build_functional_output,
    create_functional_op,
    effective_groupby_input_dtypes,
    groupby_key_output_dtypes,
    merge_hidden_into_physical,
    propagate_hidden_metadata,
    validate_single_output,
)
from maxframe.tensor.utils import validate_axis


def apply(
    self,
    func,
    axis=0,
    *,
    result_type=None,
    dtypes=None,
    skip_infer=False,
    input_frame_type="pandas",
    args=(),
    **kwargs,
):
    """Apply ``func`` per column (axis=0) or per row (axis=1).

    Mirrors ``pandas.DataFrame.apply``:

    - ``axis=0``: ``func`` receives each column as a Series and may return a
      scalar, Series, or DataFrame.  All row chunks are concatenated first
      so the UDF sees the whole column.
    - ``axis=1``: ``func`` receives each row as a Series and may return a
      scalar, Series, or dict.  Results are combined one output row per
      input row.  Because per-row UDFs are not mock-inferable from the whole
      frame, ``axis=1`` generally requires explicit ``dtypes`` or a return
      type annotation.

    Parameters
    ----------
    func : callable or callable class
        Function taking a Series (a column for ``axis=0``, a row for
        ``axis=1``) and returning a scalar, Series, dict, or DataFrame.

        If ``func`` is a class, it is instantiated once per chunk,
        ``__call__`` is invoked per column (``axis=0``) or per row
        (``axis=1``), and ``close()`` (if defined) is always called at
        the end of the chunk — even when ``__call__`` receives single
        rows/columns.
    axis : {0, 1}, default 0
        ``axis=0`` iterates over columns; ``axis=1`` iterates over rows.
    result_type : {"expand", "reduce", "broadcast", None}, default None
        Only applies when ``axis=1``.  Controls how list-like results
        (dict, list, Series) are combined into the output frame:

        * ``None`` (default): ``Series`` results are auto-expanded to
          columns (matching pandas, where a per-row ``Series`` is combined
          into a multi-column frame).  ``dict`` and ``list`` results are
          kept as a single object column (one cell per row) — pandas 2.3.3
          does not expand dict keys or lists unless ``result_type='expand'``
          is given; declaring multiple columns for a dict / list result
          raises.
        * ``"expand"``: dict / list / Series results are always expanded
          to columns (dict keys / positional indices / Series index
          become column names).
        * ``"reduce"``: dict / list results are kept as a single column
          of objects (one cell per row), matching pandas
          ``result_type='reduce'``.  ``Series`` results are still expanded
          to columns (pandas behavior).
        * ``"broadcast"``: results are broadcast to the original input
          shape — the output has the same column names as the input.
          Requires list-like results whose length matches the input
          column count.
    dtypes : dict or pd.Series, optional
        Expected output dtypes.  If omitted, inferred from ``func``'s
        return annotation or a mock call.
    skip_infer : bool, default False
        If True, skip dtype inference and produce an untyped output.
    input_frame_type : {"pandas", "polars"}, default "pandas"
        Frame type passed to ``func``.
    args : tuple
        Extra positional arguments forwarded to ``func``.
    **kwargs
        Extra keyword arguments forwarded to ``func``.

    Returns
    -------
    LiteFrame
        A new LiteFrame whose dtypes match the (inferred or explicit)
        output schema.

    See Also
    --------
    apply_chunk : Apply a function per execution chunk.
    transform : Transform per column or per row.
    map : Apply a function element-wise.

    Examples
    --------
    >>> import maxframe.liteframe as mlf
    >>> lf = mlf.LiteFrame({'a': [1, 2, 3], 'b': [10, 20, 30]})

    Apply a reduction per column (``axis=0``, default) — ``func`` receives
    each column as a Series:

    >>> result = lf.apply(lambda col: col.sum())
    >>> result.execute()
        a   b
    0   6  60

    Apply per row (``axis=1``) — ``func`` receives each row as a Series.
    Declare the output schema with ``dtypes``:

    >>> result = lf.apply(lambda row: {'sum': row['a'] + row['b']},
    ...                   axis=1, dtypes={'sum': int})
    >>> result.execute()
       sum
    0   11
    1   22
    2   33

    With ``axis=1`` and ``result_type='expand'``, dict results are expanded
    to columns:

    >>> result = lf.apply(lambda row: {'a2': row['a'] ** 2, 'b2': row['b'] ** 2},
    ...                   axis=1, result_type='expand',
    ...                   dtypes={'a2': int, 'b2': int})
    >>> result.execute()
       a2    b2
    0   1   100
    1   4   400
    2   9   900
    """
    # Validate axis up front so an unsupported value (e.g. axis=2) raises a
    # clear error instead of silently falling through to the executor's
    # whole-frame path.  LiteFrame is always 2-D, so only {0, 1} are valid.
    axis = validate_axis(self.ndim, axis, argname="axis")
    if result_type is not None and result_type not in {"expand", "reduce", "broadcast"}:
        raise ValueError(
            f"Invalid result_type='{result_type}'; expected one of "
            f"{{None, 'expand', 'reduce', 'broadcast'}}."
        )
    if axis == 1:
        # apply(axis=1) preserves row count (one result per row), so hidden
        # columns can be preserved — stripped from the UDF input and
        # re-appended after execution, matching the transform-family preserve
        # policy.  The operator input is ``self`` (with hidden columns) so the
        # executor can re-append them; the tiler carries the hidden names onto
        # the opfunc so the executor strips them before calling the UDF.
        op, inferred = create_functional_op(
            func,
            mode="apply",
            input_dtypes=self.dtypes,
            dtypes=dtypes,
            skip_infer=skip_infer,
            input_frame_type=input_frame_type,
            args=args,
            kwargs=kwargs,
            axis=axis,
            result_type=result_type,
        )
        if inferred is None:
            # skip_infer (untyped): no known visible schema, so hidden columns
            # cannot be carried into the output.  Drop them from the UDF input.
            hidden = sorted(self._hidden_columns)
            inp = self.drop(hidden) if hidden else self
            return build_functional_output(op, [inp], inferred)
        if len(inferred) > 1:
            # Multi-output: apply(axis=1) preserves hidden columns, carrying
            # them into each output's schema independently (one output row
            # per input row holds for every tuple element).  Pass the full
            # input (with hidden) plus the propagated hidden metadata so
            # build_functional_output merges hidden into each output.
            return build_functional_output(
                op, [self], inferred, frame_metadata=propagate_hidden_metadata(self)
            )
        out_dtypes = inferred[0]
        out_meta = propagate_hidden_metadata(self)
        physical_dtypes, out_meta, n_cols = merge_hidden_into_physical(
            self, out_dtypes, out_meta
        )
        return op.new_liteframe(
            [self],
            shape=(self.shape[0], n_cols),
            physical_dtypes=physical_dtypes,
            frame_metadata=out_meta,
        )
    else:
        # apply(axis=0): row count may change (reduction), so hidden columns
        # are dropped from the input and not carried into the output.
        hidden = sorted(self._hidden_columns)
        inp = self.drop(hidden) if hidden else self
        op, inferred = create_functional_op(
            func,
            mode="apply",
            input_dtypes=self.dtypes,
            dtypes=dtypes,
            skip_infer=skip_infer,
            input_frame_type=input_frame_type,
            args=args,
            kwargs=kwargs,
            axis=axis,
            result_type=result_type,
        )
        return build_functional_output(op, [inp], inferred)


def groupby_apply(
    self,
    func,
    *,
    dtypes=None,
    skip_infer=False,
    input_frame_type="pandas",
    args=(),
    **kwargs,
):
    """Apply ``func`` to each group as a whole frame.

    The user function receives one DataFrame per group (not per column or
    row) and may return a DataFrame, Series, or scalar.

    apply may change the row count / row order, so hidden columns are not
    propagated -- they only survive if the UDF explicitly returns them.

    Parameters
    ----------
    func : callable or callable class
        Function taking a group DataFrame and returning a DataFrame,
        Series, or scalar.

        If ``func`` is a class, it is instantiated once per group,
        ``__call__(batch, end=True)`` is invoked once with ``end=True``
        (each call is a complete invocation that must flush its output),
        and ``close()`` (if defined) is always called at the end of each
        group.
    dtypes : dict or pd.Series, optional
        Expected output dtypes.
    skip_infer : bool, default False
        If True, skip dtype inference.
    input_frame_type : {"pandas", "polars"}, default "pandas"
        Frame type passed to ``func``.
    args : tuple
        Extra positional arguments forwarded to ``func``.
    **kwargs
        Extra keyword arguments forwarded to ``func``.

    Returns
    -------
    LiteFrame
        A new LiteFrame with the results concatenated across groups.

    See Also
    --------
    groupby_transform : Per-group column transform.
    groupby_apply_chunk : Per-group-chunk application.

    Examples
    --------
    >>> import maxframe.liteframe as mlf
    >>> lf = mlf.LiteFrame({'key': ['a', 'a', 'b', 'b'], 'val': [1, 2, 3, 4]})

    Return one row per group (reduction):

    >>> result = lf.groupby('key').apply(lambda g: g['val'].sum(),
    ...                                 dtypes={'val': int})
    >>> result.execute()
      key  val
    0   a    3
    1   b    7

    Return all rows of each group (identity):

    >>> result = lf.groupby('key')[['key', 'val']].apply(
    ...     lambda g: g, dtypes={'key': str, 'val': int})
    >>> result.execute()
      key  val
    0   a    1
    1   a    2
    2   b    3
    3   b    4
    """
    return groupby_apply_op(
        self, func, "apply", dtypes, skip_infer, input_frame_type, args, kwargs
    )


def groupby_apply_op(
    self, func, mode, dtypes, skip_infer, input_frame_type, args, kwargs
):
    """Shared helper for groupby_apply and groupby_apply_chunk.

    Both create a ``LiteFrameFunctional`` with ``groupby_params`` and build
    the output without propagating hidden metadata (apply-family drop policy:
    row count / order may change, so hidden columns cannot be safely
    re-appended).

    Hidden columns are dropped from the operator input at the API (where the
    tileable carries ``frame_metadata``) so the executor's input is
    visible-only -- matching the non-groupby ``apply`` / ``apply_chunk`` drop
    policy.  Without this the operator received the original physical
    LiteFrame whose ``_physical_dtypes`` includes hidden columns, so they
    leaked into the UDF input at execution even though inference excluded
    them.  Users who need a hidden column must ``select`` it first.
    """
    effective_dtypes = effective_groupby_input_dtypes(
        self._liteframe, self.groupby_params
    )
    hidden = sorted(self._liteframe._hidden_columns)
    inp = self._liteframe.drop(hidden) if hidden else self._liteframe
    op, inferred = create_functional_op(
        func,
        mode=mode,
        input_dtypes=effective_dtypes,
        dtypes=dtypes,
        skip_infer=skip_infer,
        input_frame_type=input_frame_type,
        args=args,
        kwargs=kwargs,
        groupby_params=self.groupby_params,
    )
    if inferred is not None:
        # groupby apply includes the grouping keys in its output (pandas
        # groupby.apply(...).reset_index() semantics): the executor prefixes
        # each per-group result with the group's key values, so the declared
        # schema must carry the key columns too.  The operator carries the
        # per-chunk output dtypes (the tiler copies op fields onto every
        # chunk opfunc), so update it as well -- otherwise chunks validate /
        # materialize against the key-less schema.  group_keys=False omits
        # the keys entirely ("no key columns" contract).
        if self.groupby_params.get("group_keys", True):
            inferred = groupby_key_output_dtypes(
                self._liteframe, self.groupby_params, inferred
            )
        op.output_dtypes = inferred
    return build_functional_output(op, [inp], inferred)


def groupby_transform_op(
    self, func, mode, api_name, dtypes, skip_infer, input_frame_type, args, kwargs
):
    """Shared helper for groupby_transform and groupby_transform_chunk.

    Both follow the same pattern: compute effective input dtypes (excluding
    key columns / applying selection), create the operator, validate single
    output, merge hidden columns into physical_dtypes, and build the result.
    """
    effective_dtypes = effective_groupby_input_dtypes(
        self._liteframe, self.groupby_params
    )
    op, inferred = create_functional_op(
        func,
        mode=mode,
        input_dtypes=effective_dtypes,
        dtypes=dtypes,
        skip_infer=skip_infer,
        input_frame_type=input_frame_type,
        args=args,
        kwargs=kwargs,
        groupby_params=self.groupby_params,
        axis=0,
    )
    validate_single_output(inferred, api_name)
    if inferred is None:
        # skip_infer: the output schema is unknown until execution.  Do NOT
        # advertise the input schema as the output (the UDF may return fewer
        # columns), otherwise a downstream select on a non-produced column
        # (e.g. result[['key']] when the UDF only returned 'val') would be
        # accepted at graph construction and then fail with ColumnNotFoundError
        # at execution.  Build an untyped output (physical_dtypes=None) so the
        # schema stays unknown until the UDF runs -- mirroring
        # build_functional_output's skip_infer path.  Hidden columns cannot be
        # carried without a known visible schema, so the output carries no
        # hidden metadata.
        return op.new_liteframe(
            [self._liteframe],
            shape=(self._liteframe.shape[0], np.nan),
            physical_dtypes=None,
            frame_metadata=FrameMetadata(),
        )
    out_dtypes = inferred[0]
    out_meta = propagate_hidden_metadata(self._liteframe)
    physical_dtypes, out_meta, n_cols = merge_hidden_into_physical(
        self._liteframe, out_dtypes, out_meta
    )
    return op.new_liteframe(
        [self._liteframe],
        shape=(self._liteframe.shape[0], n_cols),
        physical_dtypes=physical_dtypes,
        frame_metadata=out_meta,
    )
