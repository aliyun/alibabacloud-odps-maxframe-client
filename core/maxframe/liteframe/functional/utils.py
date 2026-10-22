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

"""Shared utilities for the LiteFrame functional (UDF) API module.

These helpers are used by both the expression-based UDF APIs (``map``,
``transform_chunk``, ``transform(axis=1)``) and the operator-based UDF APIs
(``apply``, ``apply_chunk``, ``transform(axis=0)``) that use the
``LiteFrameFunctional`` operator.  The operator-based helpers live here so the
API modules (``apply.py``, ``apply_chunk.py``, ``transform.py``) stay thin.
"""

import contextlib
import inspect
import types

import numpy as np
import pandas as pd

from maxframe.core import OutputType
from maxframe.liteframe.core import FrameMetadata
from maxframe.liteframe.expressions import (
    LiteFrameColumn,
    LiteFrameMultiColumnSubExpr,
    LiteFrameUDFExpr,
    NamedExpr,
)
from maxframe.liteframe.functional.type_infer import infer_output_dtypes
from maxframe.liteframe.operators.functional import LiteFrameFunctional
from maxframe.liteframe.operators.project import _append_hidden_projections

try:
    import polars as pl
except ImportError:  # pragma: no cover
    pl = None

from maxframe.core.operator.base import SchedulingHint
from maxframe.liteframe.functional.type_infer import _normalize_to_arrow_dtype
from maxframe.udf import discover_marked_functions
from maxframe.utils import unwrap_function


def pop_check_output_dtypes(kwargs):
    """Pop ``_check_output_dtypes`` from ``kwargs``, defaulting to ``"raises"``."""
    return kwargs.pop("_check_output_dtypes", "raises")


def validate_single_output(inferred, api_name):
    """Reject multi-output schemas for single-output transform APIs.

    ``transform()`` and ``transform_chunk()`` are single-output APIs. If type
    inference returns multiple output schemas (e.g. user passed a list of
    dtypes dicts), raise instead of silently taking ``inferred[0]``.
    """
    if inferred is not None and len(inferred) > 1:
        raise ValueError(
            f"{api_name} is a single-output API but received "
            f"{len(inferred)} output schemas. Use apply_chunk() or "
            f"apply() for multi-output UDFs."
        )


def _build_scheduling_hint_from_func(func):
    """Extract a SchedulingHint from a (possibly MarkedFunction) UDF.

    Public UDF builders (``transform_chunk`` etc.) receive the user function
    which may be wrapped by ``@with_running_options`` / ``@with_resources``
    as a ``MarkedFunction``.  The projection tiler reads
    ``LiteFrameUDFExpr.scheduling_hint`` for color-grouping, so the hint must
    be set on the expression for the splitter to see incompatible running
    options.  Returns ``None`` when the function has no running options.
    """
    marked_funcs = discover_marked_functions(func)
    if not marked_funcs:
        return None
    mf = marked_funcs[0]
    has_options = any(
        (
            mf.expect_engine,
            mf.expect_resources,
            mf.gpu,
            mf.fs_mount,
            mf.image_options,
        )
    )
    if not has_options:
        return None
    return SchedulingHint(
        gpu=mf.gpu or None,
        expect_engine=mf.expect_engine or None,
        expect_resources=dict(mf.expect_resources) if mf.expect_resources else None,
        fs_mount=list(mf.fs_mount) if mf.fs_mount else None,
        image_options=dict(mf.image_options) if mf.image_options else None,
    )


def build_udf_projections(
    lf, func, mode, out_dtypes, input_frame_type, check_output_dtypes, args, kwargs
):
    """Build ``LiteFrameProjection`` projections for expression-based UDFs.

    The ``operands`` on the ``LiteFrameUDFExpr`` are the input
    ``LiteFrameColumn`` nodes the UDF reads from.  They double as the
    expression-tree binding anchor (so binding succeeds even when the UDF
    produces columns absent from the input schema) and as the column set
    the executor selects before calling the UDF.

    For ``transform_chunk`` / ``transform_row`` we pass all input columns
    because the UDF may reference any of them.  For ``map`` (handled in
    ``map.py``), each per-column UDF only needs its own column.
    """
    output_names = list(out_dtypes.index)
    dtypes = dict(out_dtypes)
    input_columns = list(lf.dtypes.index)
    # Operands reference *input* columns (which exist in the frame schema),
    # so expression-tree binding never raises for UDF-produced output columns
    # that are absent from the input schema.
    input_operands = [
        LiteFrameColumn(name=c, dtype=lf.dtypes[c]) for c in input_columns
    ]

    scheduling_hint = _build_scheduling_hint_from_func(func)
    udf_expr = LiteFrameUDFExpr(
        func=func,
        mode=mode,
        operands=input_operands,
        output_names=output_names,
        dtypes=dtypes,
        input_frame_type=input_frame_type,
        check_output_dtypes=check_output_dtypes,
        args=args,
        kwargs=kwargs,
        scheduling_hint=scheduling_hint,
    )

    projections = []
    for col_name in output_names:
        sub_expr = LiteFrameMultiColumnSubExpr(
            multi_expr=udf_expr,
            output_name=col_name,
            dtype=dtypes[col_name],
        )
        projections.append(
            NamedExpr(name=col_name, expr=sub_expr, dtype=dtypes[col_name])
        )

    _append_hidden_projections(lf, projections)
    return projections


def _invoke_udf(underlying, inputs, args, kwargs):
    """Invoke a (possibly callable-class) UDF on one or more inputs.

    Mirrors the executor's ``make_callable_wrapper`` / ``call_user_func``: for
    callable classes, **one** instance is created and reused across all inputs,
    with ``end=True`` passed on every invocation (each call is a complete
    invocation that must flush its output) so inference sees the same
    output dtypes the runtime does.  ``close()`` runs in a ``finally``
    block.  Plain functions are called directly per input.
    """
    if isinstance(underlying, type):
        instance = underlying()
        sig = inspect.signature(instance.__call__)
        has_end = "end" in sig.parameters
        try:
            results = []
            for data in inputs:
                if has_end:
                    results.append(instance(data, *args, end=True, **kwargs))
                else:
                    results.append(instance(data, *args, **kwargs))
            return results
        finally:
            if hasattr(instance, "close"):
                instance.close()
    return [underlying(data, *args, **kwargs) for data in inputs]


def _invoke_udf_once(underlying, data, args, kwargs):
    """Invoke a (possibly callable-class) UDF once on *data*, closing on exit.

    Thin wrapper around :func:`_invoke_udf` for the single-input case.
    """
    return _invoke_udf(underlying, [data], args, kwargs)[0]


def _normalize_row_result_to_frame(res):
    """Turn a single per-row/per-column UDF result into a one-row frame.

    A row or column UDF may return a scalar, ``dict``, ``pd.Series``,
    ``pl.Series`` or a single-row frame; each is normalized so dtype
    extraction can read the output columns/dtypes.  Dtypes are converted to
    LiteFrame's Arrow-backed convention to match the chunk-wise inference
    path.
    """
    if pl is not None and isinstance(res, (pl.DataFrame, pl.LazyFrame)):
        return res
    if isinstance(res, pd.DataFrame):
        return res
    if pl is not None and isinstance(res, pl.Series):
        return res.to_frame()
    if isinstance(res, pd.Series):
        frame = res.to_frame().T
    elif isinstance(res, dict):
        frame = pd.DataFrame([res])
    else:
        frame = pd.DataFrame({0: [res]})
    arrow_dtypes = {}
    for col, dt in frame.dtypes.items():
        try:
            arrow_dtypes[col] = _normalize_to_arrow_dtype(dt)
        except Exception:
            arrow_dtypes[col] = dt
    try:
        frame = frame.astype(arrow_dtypes)
    except Exception:
        pass
    return frame


def _normalize_result_to_frame(res):
    """Normalize a UDF result (or tuple of results) into frame(s) for inference.

    Handles multi-output UDFs that return a tuple: each element is normalized
    individually via :func:`_normalize_row_result_to_frame` so that
    :func:`~maxframe.liteframe.functional.type_infer.infer_output_dtypes` can
    extract dtypes from each output.
    """
    if isinstance(res, tuple):
        return tuple(_normalize_row_result_to_frame(r) for r in res)
    return _normalize_row_result_to_frame(res)


def _is_flat_apply_iterable(res):
    """Return True if *res* is a generator or a (non-string/dict) list-like.

    Such results are flattened element-by-element at runtime (each element is
    one output row), so inference only needs the first element's schema.
    """
    if isinstance(res, types.GeneratorType):
        return True
    if isinstance(res, (str, bytes, dict)):
        return False
    if pl is not None and isinstance(res, (pl.Series, pl.DataFrame, pl.LazyFrame)):
        return False
    if isinstance(res, (pd.DataFrame, pd.Series)):
        return False
    return hasattr(res, "__iter__")


def _normalize_flat_apply_result_to_frame(res):
    """Normalize a per-row ``flat_apply`` result into frame(s) for inference.

    ``flat_apply`` calls ``func`` once per input row.  A **top-level tuple**
    denotes multiple outputs and is normalized element-wise: each element is
    that output's per-row result, flattened independently by its type
    (DataFrame -> its own rows; generator / list-like -> one row per element,
    so the *first* element's schema is used; dict / Series / scalar -> one
    output row).  This makes the row-selection criterion per-output and
    type-based, so different outputs may use different result forms.

    A top-level tuple is *only* multi-output; a single declared output must
    not return a top-level tuple (the executor rejects it), and tuple rows
    are valid only when nested inside a list/generator.  Only the schema
    (column names + dtypes) matters for inference, mirroring the executor's
    flattening at runtime.
    """
    if isinstance(res, tuple):
        return tuple(_normalize_flat_apply_result_to_frame(r) for r in res)
    if res is None:
        return pd.DataFrame()
    if _is_flat_apply_iterable(res):
        items = list(res)
        if not items:
            return pd.DataFrame()
        return _normalize_row_result_to_frame(items[0])
    return _normalize_row_result_to_frame(res)


def _normalize_transform_result_to_frame(
    res, input_frame_type="pandas", input_col_names=None
):
    """Normalize a groupby *transform* UDF result for mock type inference.

    Transform modes have a different Series contract than apply: a per-group
    Series becomes **one output column** with one value per group row (mirroring
    the executor's ``_normalize_transform_group_result``), whereas apply turns
    a Series into a one-row, multi-column frame (Series index -> columns via
    :func:`_normalize_row_result_to_frame`).  Using the apply normalization for
    a transform UDF inferred the wrong column count (e.g. a 2-element Series
    -> columns {0,1}) and then execution raised because a Series can only be a
    single output column.

    Only the column count and dtypes matter for inference (not the row count).
    When ``input_col_names`` is provided (the mock frame's column names):

    - A **scalar** result is broadcast to *all* input columns (pandas
      ``groupby.transform(lambda g: 999)`` fills every column), so the inferred
      schema carries one column per input column.
    - A **Series** whose length matches the number of input columns (e.g.
      ``g.mean()``) is treated as one value per column: each value's dtype
      defines one output column.
    - A **Series** whose length does not match the input column count is
      treated as a single output column (one value per group row).

    DataFrame results are returned as-is; dict results fall back to the
    apply-style one-row normalization (transform dict results are an edge case
    deferred to concat at runtime).
    """
    if isinstance(res, tuple):
        return tuple(
            _normalize_transform_result_to_frame(r, input_frame_type, input_col_names)
            for r in res
        )
    if pl is not None and isinstance(res, (pl.DataFrame, pl.LazyFrame)):
        return res
    if isinstance(res, pd.DataFrame):
        return res

    n_input = len(input_col_names) if input_col_names else 0

    # A polars Series has no index labels, so it is inherently positional and
    # is treated as a per-row result (single output column) -- per-column
    # results in polars come as a DataFrame (e.g. g.mean()).  Convert to a
    # pandas Series so the shared label-based classification below applies.
    if pl is not None and isinstance(res, pl.Series):
        res = pd.Series(res.to_list())

    if isinstance(res, pd.Series):
        # Per-column only when the Series labels match the input columns (e.g.
        # g.mean() returns a Series indexed by the input column names).  A
        # positional (RangeIndex) Series is a single per-row output column,
        # even when its length happens to equal the input column count (e.g.
        # (g["v1"] + g["v2"]).reset_index(drop=True) on a 2-row, 2-column
        # group) -- otherwise inference/execution disagree and the per-row
        # result is broadcast across the wrong columns.
        if (
            n_input >= 1
            and len(res) == n_input
            and set(res.index) == set(input_col_names)
        ):
            # Per-column Series (e.g. g.mean()): one dtype per input column.
            # Labels may be in any order (aligned by name, not position).
            col_dtypes = {}
            for col in input_col_names:
                dt = pd.Series([res[col]]).dtype
                try:
                    dt = _normalize_to_arrow_dtype(dt)
                except Exception:
                    pass
                col_dtypes[col] = dt
            return pd.DataFrame(
                {c: pd.Series(dtype=dt) for c, dt in col_dtypes.items()}
            )
        frame = res.to_frame()
    elif isinstance(res, dict):
        frame = pd.DataFrame([res])
    else:
        # scalar -> broadcast to all input columns (pandas groupby.transform
        # behavior).  When input_col_names is unknown, fall back to a single
        # column.
        if n_input > 0:
            dt = pd.Series([res]).dtype
            try:
                dt = _normalize_to_arrow_dtype(dt)
            except Exception:
                pass
            return pd.DataFrame({c: pd.Series(dtype=dt) for c in input_col_names})
        frame = pd.DataFrame({0: [res]})
    arrow_dtypes = {}
    for col, dt in frame.dtypes.items():
        try:
            arrow_dtypes[col] = _normalize_to_arrow_dtype(dt)
        except Exception:
            arrow_dtypes[col] = dt
    try:
        frame = frame.astype(arrow_dtypes)
    except Exception:
        pass
    return frame


def _normalize_col_result_to_frame(res, col_names):
    """Build a frame with input column names and the result's dtype.

    For per-column (axis=0) UDFs the runner builds ``{col_name: udf_result}``
    and constructs the frame column-wise, so the output column names are the
    *input* column names and the dtype comes from the UDF result.  A single
    representative column is used for inference, so the result dtype is
    assumed uniform across all columns.
    """
    if isinstance(res, tuple):
        return tuple(_normalize_col_result_to_frame(r, col_names) for r in res)

    # If the UDF already returned a frame, use it as-is.
    if isinstance(res, pd.DataFrame):
        return res
    if pl is not None and isinstance(res, (pl.DataFrame, pl.LazyFrame)):
        return res

    # Extract the element dtype from the representative result.
    if isinstance(res, pd.Series):
        dt = res.dtype
    elif pl is not None and isinstance(res, pl.Series):
        dt = res.to_pandas().dtype
    else:
        # scalar
        dt = pd.Series([res]).dtype

    # Convert to Arrow-backed dtype (LiteFrame convention).
    try:
        arrow_dt = _normalize_to_arrow_dtype(dt)
    except Exception:
        arrow_dt = dt

    # Build a (zero-row) frame whose dtypes carry the input column names.
    return pd.DataFrame({col: pd.Series(dtype=arrow_dt) for col in col_names})


def _extract_result_dtype(res):
    """Extract the dtype from a single per-column UDF result."""
    if isinstance(res, pd.Series):
        dt = res.dtype
    elif pl is not None and isinstance(res, pl.Series):
        dt = res.to_pandas().dtype
    else:
        # scalar
        dt = pd.Series([res]).dtype
    try:
        return _normalize_to_arrow_dtype(dt)
    except Exception:
        return dt


def _normalize_col_results_to_frame(results, col_names):
    """Build per-output frames from per-column UDF results.

    Each element of *results* is the UDF's return value for one input column
    (a scalar, Series, or -- for multi-output UDFs -- a tuple with one element
    per output).  Each output column carries its own result dtype, mirroring
    the ``Axis0FunctionalRunner`` which iterates every column at runtime.
    """
    if any(isinstance(r, tuple) for r in results):
        # Multi-output: each column returned a tuple.  Collect per-output.
        n_outputs = len(results[0])
        frames = []
        for out_i in range(n_outputs):
            col_dtypes = {}
            for col_name, res in zip(col_names, results):
                col_dtypes[col_name] = _extract_result_dtype(res[out_i])
            frames.append(
                pd.DataFrame({c: pd.Series(dtype=dt) for c, dt in col_dtypes.items()})
            )
        return tuple(frames)

    # Single-output: each result is a scalar or Series.
    col_dtypes = {}
    for col_name, res in zip(col_names, results):
        col_dtypes[col_name] = _extract_result_dtype(res)
    return pd.DataFrame({c: pd.Series(dtype=dt) for c, dt in col_dtypes.items()})


def _stringify_col_labels(frame):
    """Convert integer column labels to strings (polars convention).

    Polars always stores column names as strings, so a pandas frame with
    integer column labels (e.g. from a scalar UDF result) must be
    stringified to match the runtime output.
    """
    if isinstance(frame, pd.DataFrame):
        frame = frame.copy()
        frame.columns = [str(c) for c in frame.columns]
    return frame


def _stringify_frame_labels(frame):
    """Stringify column labels on a frame or tuple of frames (boundary contract).

    LiteFrame column names are always strings; a scalar/dict UDF result is
    normalized to a pandas frame with an integer label (0), which must be
    stringified to "0" so the inferred schema matches the polars runtime
    and survives the ODPS temp-table round trip.
    """
    if isinstance(frame, tuple):
        return tuple(_stringify_col_labels(f) for f in frame)
    return _stringify_col_labels(frame)


def _normalize_apply_row_result(res, result_type, mock_df):
    """Normalize a per-row apply(axis=1) UDF result for mock type inference.

    Respects the ``result_type`` argument, mirroring
    ``pandas.DataFrame.apply(func, axis=1, result_type=...)``:

    * ``"expand"``: dict / list / Series results are expanded to columns.
    * ``"reduce"``: dict / list results become a single object column.
      Series results are still expanded (pandas behavior).
    * ``"broadcast"``: the output shape matches the input; inference uses the
      input column names so the declared schema aligns with the runtime.
    * ``None`` (default): dict results are kept as a single object column
      (matching pandas 2.3.3, where ``apply(axis=1)`` returns a Series of
      dict objects unless ``result_type='expand'`` is given); Series results
      auto-expand to columns; list results stay as a single object column.

    Only the column count and dtypes matter for inference (not the row count).
    """
    # Normalize a native polars Series to a positional pandas Series so the
    # result-type rules below apply uniformly: pl.Series.to_frame() is vertical
    # (one column, N rows), which would wrongly infer a single column for a
    # multi-element Series, and the broadcast branch relies on list()/to_list().
    if pl is not None and isinstance(res, pl.Series):
        res = pd.Series(res.to_list())

    # Determine whether res is list-like-but-not-Series/dict (i.e. a list/tuple).
    is_list_like = isinstance(res, (list, tuple)) or (
        pl is not None and isinstance(res, pl.Series)
    )

    if result_type == "reduce":
        # reduce: dict / list-like results stay as a single object column.
        # Series results are still expanded (pandas behavior).
        if isinstance(res, dict) or is_list_like:
            return pd.DataFrame({0: pd.Series(dtype=object)})
        return _normalize_result_to_frame(res)

    if result_type == "broadcast":
        # broadcast: output shape matches input; use input column names.
        col_names = list(mock_df.columns)
        # For a scalar result, broadcast its dtype to all input columns.
        if isinstance(res, dict):
            # dict cannot broadcast — will error at execution.  For inference,
            # use input column names with object dtype so the graph can be built.
            return pd.DataFrame({c: pd.Series(dtype=object) for c in col_names})
        if isinstance(res, (pd.Series, list, tuple)) or (
            pl is not None and isinstance(res, pl.Series)
        ):
            # List-like: extract per-element dtypes if length matches.
            vals = (
                res.to_list()
                if (pl is not None and isinstance(res, pl.Series))
                else list(res)
            )
            if len(vals) == len(col_names):
                col_dtypes = {}
                for c, v in zip(col_names, vals):
                    dt = pd.Series([v]).dtype
                    try:
                        dt = _normalize_to_arrow_dtype(dt)
                    except Exception:
                        pass
                    col_dtypes[c] = dt
                return pd.DataFrame(
                    {c: pd.Series(dtype=dt) for c, dt in col_dtypes.items()}
                )
            # Length mismatch — will error at execution.  Use input names.
            return pd.DataFrame({c: pd.Series(dtype=object) for c in col_names})
        # Scalar: broadcast its dtype to all input columns.
        dt = pd.Series([res]).dtype
        try:
            dt = _normalize_to_arrow_dtype(dt)
        except Exception:
            pass
        return pd.DataFrame({c: pd.Series(dtype=dt) for c in col_names})

    # result_type is None (default) or "expand":
    # expand forces list-like results to expand to columns too.
    if result_type == "expand" and is_list_like:
        # Treat a list/tuple as a positional Series so it expands to columns.
        if pl is not None and isinstance(res, pl.Series):
            return res.to_frame()
        return pd.Series(res).to_frame().T
    # Default (result_type is None): pandas keeps dict results as a single
    # object column (one cell per row); only result_type='expand' turns dict
    # keys into columns.  Series results still auto-expand (pandas behavior).
    # Mirror that so inference agrees with the executor, otherwise a no-dtypes
    # apply(axis=1) returning dicts infers multiple columns and then fails at
    # execution when each dict is kept as one object value.
    if result_type is None and isinstance(res, dict):
        return pd.DataFrame({0: pd.Series(dtype=object)})
    return _normalize_result_to_frame(res)


def wrap_inference_func(
    func, mode, input_frame_type="pandas", args=(), kwargs=None, result_type=None
):
    """Wrap a user UDF so mock type inference respects its execution contract.

    :func:`~maxframe.liteframe.functional.type_infer.infer_output_dtypes`
    always calls ``func(mock_df)`` and extracts dtypes from the returned
    frame; it stays free of mode-specific calling logic.  The wrapper owns the
    execution contract -- what data the UDF receives and how the result is
    normalized -- so that mock inference faithfully reflects the runtime
    calling convention for every mode.

    Supported modes:

    - ``"transform_row"`` / ``"apply_row"``: the UDF receives a single row (a
      ``pandas.Series`` in pandas mode, a single-row ``polars.DataFrame`` in
      polars mode).  The wrapper invokes the UDF on a representative row and
      normalizes the per-row result into a frame.
    - ``"apply_col"``: the UDF receives a single column as a Series.  The
      wrapper invokes the UDF on *every* column (not just a representative
      one) so each output column carries its own result dtype, mirroring the
      ``Axis0FunctionalRunner`` at runtime.
    - ``"groupby"``: the UDF receives a whole group frame and may return a
      DataFrame, Series, dict, scalar, or tuple (multi-output).  The wrapper
      invokes the UDF on the whole mock frame and normalizes the result via
      :func:`_normalize_result_to_frame`, mirroring the executor's
      ``_normalize_group_result``.
    - anything else (e.g. ``"transform_chunk"`` / ``"apply_chunk"``): the UDF
      receives the whole frame; no wrapping is needed beyond invocation.

    Callable-class UDFs are instantiated and closed here (on both success and
    failure paths).

    The original function's return annotation is preserved (via
    ``__wrapped__``) so annotation-based inference still resolves type hints
    in the user's own module globals.
    """
    kwargs = kwargs or {}
    underlying = unwrap_function(func)

    if mode in ("transform_row", "apply_row"):
        # Per-row contract: the UDF receives a single row (a pandas.Series in
        # pandas mode, a single-row polars.DataFrame in polars mode) -- not the
        # whole frame.  Invoke on a representative row and normalize the result.
        # For ``apply_row`` the ``result_type`` argument (expand / reduce /
        # broadcast) controls how dict / list results are normalized, matching
        # ``pandas.DataFrame.apply(func, axis=1, result_type=...)``:
        #
        # * ``"expand"``: dict / list / Series results are expanded to columns
        #   (dict keys / positional indices / Series index become columns).
        # * ``"reduce"``: dict / list results become a single object column.
        #   Series results are still expanded (pandas behavior).
        # * ``"broadcast"``: the output shape matches the input (one column per
        #   input column, same names); inference uses the input column names.
        # * ``None`` (default): dict results are kept as a single object column
        #   (pandas 2.3.3 keeps per-row dicts as a Series of objects unless
        #   ``result_type='expand'``); Series results are always expanded
        #   (pandas auto-expand behavior); list results stay as one column.
        is_apply_row = mode == "apply_row"

        def wrapper(mock_df):
            if (
                input_frame_type == "polars"
                and pl is not None
                and isinstance(mock_df, (pl.DataFrame, pl.LazyFrame))
            ):
                row = (
                    mock_df.slice(0, 1)
                    if isinstance(mock_df, pl.DataFrame)
                    else mock_df.collect().slice(0, 1)
                )
            else:
                row = mock_df.iloc[0]
            res = _invoke_udf_once(underlying, row, args, kwargs)
            frame = _normalize_apply_row_result(
                res, result_type if is_apply_row else None, mock_df
            )
            return _stringify_frame_labels(frame)

    elif mode == "apply_col":
        # Per-column contract: the UDF receives a single column as a Series.
        # Invoke the UDF on *every* column (not just a representative one) so
        # each output column carries its own result dtype.  Mirrors the
        # Axis0FunctionalRunner, which iterates every column at runtime.
        def wrapper(mock_df):
            col_names = list(mock_df.columns)
            if (
                input_frame_type == "polars"
                and pl is not None
                and isinstance(mock_df, (pl.DataFrame, pl.LazyFrame))
            ):
                cols = [mock_df[c] for c in col_names]
            else:
                cols = [mock_df.iloc[:, i] for i in range(len(col_names))]
            results = _invoke_udf(underlying, cols, args, kwargs)
            return _normalize_col_results_to_frame(results, col_names)

    elif mode == "groupby_transform":
        # GroupBy *transform* contract: like groupby the UDF receives a whole
        # group frame, but a Series result is ONE output column (one value per
        # group row), not a transposed multi-column frame.  Use the transform
        # normalization so inference matches the executor's
        # ``_normalize_transform_group_result`` (otherwise a Series-returning
        # transform infers the wrong column count and execution raises).
        # Pass the mock frame's column names so scalar/Series-per-column
        # broadcasts infer the correct output column count (matching the
        # executor's broadcast semantics).
        def wrapper(mock_df):
            res = _invoke_udf_once(underlying, mock_df, args, kwargs)
            input_col_names = list(mock_df.columns)
            frame = _normalize_transform_result_to_frame(
                res, input_frame_type, input_col_names
            )
            return _stringify_frame_labels(frame)

    elif mode == "groupby":
        # GroupBy contract: the UDF receives a whole group frame (not a
        # single row or column) and may return a DataFrame, Series, dict,
        # scalar, or a tuple of these (multi-output).  Invoke on the whole
        # mock frame and normalize the result into frame(s) so dtype
        # extraction can read the output columns/dtypes -- mirroring the
        # executor's ``_normalize_group_result``.
        def wrapper(mock_df):
            res = _invoke_udf_once(underlying, mock_df, args, kwargs)
            frame = _normalize_result_to_frame(res)
            return _stringify_frame_labels(frame)

    elif mode == "flat_apply":
        # Per-row row-expanding contract: the UDF receives a single row (a
        # ``pandas.Series`` in pandas mode, a single-row ``polars.DataFrame``
        # in polars mode) and may return a DataFrame, generator, list-like,
        # dict, Series, scalar, or a tuple of these (multi-output).  Each
        # result flattens to 0-N output rows at runtime, but inference only
        # needs the output *schema* (column names + dtypes), so the first
        # row/item is normalized into a frame via
        # :func:`_normalize_flat_apply_result_to_frame`.
        def wrapper(mock_df):
            if (
                input_frame_type == "polars"
                and pl is not None
                and isinstance(mock_df, (pl.DataFrame, pl.LazyFrame))
            ):
                row = (
                    mock_df.slice(0, 1)
                    if isinstance(mock_df, pl.DataFrame)
                    else mock_df.collect().slice(0, 1)
                )
            else:
                row = mock_df.iloc[0]
            res = _invoke_udf_once(underlying, row, args, kwargs)
            frame = _normalize_flat_apply_result_to_frame(res)
            # LiteFrame column names are always strings (boundary contract);
            # a scalar/positional UDF result normalizes to a pandas frame
            # with an integer label (0), which must be stringified to "0"
            # so the inferred schema matches the polars runtime and the SPE
            # validation path — regardless of input_frame_type.
            return _stringify_frame_labels(frame)

    else:
        # Whole-frame contract (transform_chunk / apply_chunk): the UDF
        # already receives the whole frame.
        def wrapper(mock_df):
            return _invoke_udf_once(underlying, mock_df, args, kwargs)

    # Preserve the underlying callable's annotations so the annotation-based
    # inference path resolves type hints against the user's module globals
    # (``_get_return_annotation`` follows ``__wrapped__``).
    hints_target = underlying.__call__ if isinstance(underlying, type) else underlying
    try:
        wrapper.__wrapped__ = hints_target
        wrapper.__annotations__ = getattr(hints_target, "__annotations__", {})
    except Exception:
        pass
    return wrapper


# --- Operator-based UDF helpers (LiteFrameFunctional) ---


def create_functional_op(
    func,
    *,
    mode,
    input_dtypes,
    dtypes,
    skip_infer,
    input_frame_type,
    args,
    kwargs,
    axis=None,
    groupby_params=None,
    result_type=None,
):
    """Infer dtypes and build a :class:`LiteFrameFunctional` operator.

    Shared by ``apply``, ``apply_chunk`` and ``transform(axis=0)``.  The
    ``_check_output_dtypes`` private kwarg is popped from ``kwargs`` (it is
    not forwarded to the user function) and stored on the operator.

    The inference function is wrapped via :func:`wrap_inference_func` so that
    mock inference respects the mode's execution contract (per-column,
    per-row, or whole-frame).  ``infer_output_dtypes`` itself stays
    mode-agnostic.
    """
    kwargs = dict(kwargs) if kwargs else {}
    check_output_dtypes = pop_check_output_dtypes(kwargs)

    # Determine the inference wrapper mode from the operator mode + axis.
    # GroupBy UDFs receive a whole group frame and may return non-DataFrame
    # results (Series/scalar/dict), so they need a result-normalizing wrapper
    # (unlike plain apply_chunk, which is DataFrame-only).  Transform modes
    # use a distinct Series contract (one column, not a transposed frame), so
    # they get the "groupby_transform" wrapper; apply modes use "groupby".
    if groupby_params is not None and mode in ("transform", "transform_chunk"):
        infer_mode = "groupby_transform"
    elif groupby_params is not None:
        infer_mode = "groupby"
    elif mode == "apply" and axis == 1:
        infer_mode = "apply_row"
    elif mode in ("apply", "transform") and axis == 0:
        infer_mode = "apply_col"
    else:
        infer_mode = "chunk"

    infer_func = wrap_inference_func(
        func, infer_mode, input_frame_type, args, kwargs, result_type=result_type
    )
    inferred = infer_output_dtypes(
        infer_func,
        input_dtypes,
        dtypes=dtypes,
        skip_infer=skip_infer,
        args=(),
        kwargs={},
        input_frame_type=input_frame_type,
    )
    op = LiteFrameFunctional(
        func=func,
        mode=mode,
        axis=axis,
        groupby_params=groupby_params,
        input_frame_type=input_frame_type,
        check_output_dtypes=check_output_dtypes,
        output_dtypes=inferred,
        args=args,
        kwargs=kwargs if kwargs else None,
        result_type=result_type,
    )
    return op, inferred


def propagate_hidden_metadata(liteframe):
    """Build :class:`FrameMetadata` preserving hidden columns from the input."""
    inp_meta = liteframe.frame_metadata
    return FrameMetadata(
        hidden_columns=list(inp_meta.hidden_columns)
        if inp_meta and inp_meta.hidden_columns
        else None
    )


def effective_groupby_input_dtypes(liteframe, groupby_params):
    """Compute the dtypes the UDF will actually see for a groupby operation.

    Following pandas 3.0 semantics, groupby key columns are excluded from
    the UDF input by default.  If a column selection was made via
    ``groupby(cols)[selection]``, only the selected columns are passed to
    the UDF (key columns are included if explicitly selected).
    """
    dtypes = liteframe.dtypes
    selection = groupby_params.get("selection")
    if selection is not None:
        # Explicit selection (incl. an empty []): use exactly the selected
        # columns.  An empty selection yields zero UDF-visible columns; it
        # must not fall through to the key-exclusion default (truthiness would
        # wrongly treat [] as "no selection").
        return dtypes[selection]
    # No selection: exclude key columns (pandas 3.0 default)
    by = groupby_params.get("by")
    by_cols = by if isinstance(by, (list, tuple)) else [by]
    return dtypes.drop([c for c in by_cols if c in dtypes.index])


def build_flat_apply_output_order(
    visible_cols, subset_cols, func_out_cols, exclude_cols=None
):
    """Compute the flat_apply output column order: func-out cols inserted at
    the position where the first subset column originally located.

    The new contract (replacing "broadcast + func-out append") inserts the
    function-output columns at the position of the first ``subset`` column in
    the original visible schema, so broadcast columns keep their relative
    order *around* the func-out block instead of being hoisted to the front.

    The anchorless case ``subset=[]`` (zero-col UDF input, every visible
    column broadcast) has no first subset column to position against, so the
    func-out block is *appended* after the broadcast columns — preserving the
    pre-MR caller-visible order (e.g. ``["a", "b", "x"]``) rather than
    moving it to the front.  This is the documented fallback for the
    anchorless case: the "insert at first subset position" rule requires a
    subset anchor; with none, the func-out block follows the broadcast cols.

    When ``subset`` is ``None`` (all columns passed to the UDF) there are no
    broadcast columns, so func-out cols simply replace the full input schema
    (their natural order).

    ``exclude_cols`` (e.g. groupby key columns) are never emitted as
    broadcast columns; their position in ``visible_cols`` still serves as the
    subset anchor when ``subset`` includes one of them.  This lets the caller
    pass the full ``visible_cols`` (incl. excluded cols) so the anchor
    position is preserved without the excluded cols leaking into the output.

    Returns a list of column names in the output order.  ``func_out_cols``
    may be ``None`` (untyped / skip_infer path); the func-out block is then
    omitted and only broadcast columns are returned (in relative order).
    """
    exclude_set = set(exclude_cols or [])
    subset_set = set(subset_cols)
    broadcast_cols = [
        c for c in visible_cols if c not in subset_set and c not in exclude_set
    ]
    if not broadcast_cols:
        # Every column goes to the UDF (subset=None or subset=all); no
        # broadcast cols, so func-out cols replace the full input schema.
        return list(func_out_cols) if func_out_cols else []
    if not subset_cols:
        # subset=[] (anchorless): no first subset column to position
        # against, so append func-out after the broadcast columns —
        # preserving the pre-MR caller-visible order.
        insert_at = len(broadcast_cols)
    elif subset_cols[0] in visible_cols:
        first_subset_pos = visible_cols.index(subset_cols[0])
        insert_at = sum(
            1
            for c in visible_cols[:first_subset_pos]
            if c not in subset_set and c not in exclude_set
        )
    else:
        insert_at = len(broadcast_cols)
    if func_out_cols:
        return (
            broadcast_cols[:insert_at]
            + list(func_out_cols)
            + broadcast_cols[insert_at:]
        )

    return broadcast_cols


def classify_flat_apply_result(res, n_outputs, func_out_cols_per_output):
    """Validate a per-row ``flat_apply`` UDF result and yield per-output items.

    Returns a list of ``(sub_res, out_cols)`` tuples (one per output).  Raises
    ``ValueError`` for arity mismatches or a top-level tuple in single-output
    mode.  Tuple *rows* nested inside a list/generator remain valid.
    """
    if n_outputs > 1:
        if not isinstance(res, tuple):
            raise ValueError(
                f"flat_apply multi-output UDF returned a "
                f"{type(res).__name__} but {n_outputs} output(s) were "
                f"declared; a multi-output UDF must return a tuple with "
                f"exactly {n_outputs} elements."
            )
        if len(res) != n_outputs:
            raise ValueError(
                f"flat_apply multi-output UDF returned a tuple of "
                f"{len(res)} element(s) but {n_outputs} output(s) were "
                f"declared; a multi-output UDF must return a tuple with "
                f"exactly {n_outputs} elements."
            )
        return list(zip(res, func_out_cols_per_output))
    if isinstance(res, tuple):
        raise ValueError(
            "flat_apply single-output UDF returned a top-level tuple; "
            "a top-level tuple denotes multiple outputs. To yield "
            "tuple rows from a single output, wrap them in a list or "
            "generator, e.g. ``return [(a, b), ...]``."
        )
    return [(res, func_out_cols_per_output[0])]


def assemble_flat_apply_output(
    frames,
    func_out_cols,
    func_out_dtypes,
    broadcast_pdf,
    broadcast_cols,
    visible_cols,
    subset_cols,
    exclude_cols=None,
):
    """Assemble one output's pandas DataFrame from per-row frames.

    Concatenates per-row frames, casts func-out cols to declared dtypes,
    repeats broadcast cols positionally, and enforces column order via
    :func:`build_flat_apply_output_order`.
    """

    if frames:
        func_out = pd.concat(frames, ignore_index=True)
    else:
        func_out = (
            pd.DataFrame(columns=func_out_cols)
            if func_out_cols is not None
            else pd.DataFrame()
        )
    if func_out_dtypes:
        for col, dt in func_out_dtypes.items():
            if col in func_out.columns:
                try:
                    func_out[col] = func_out[col].astype(dt)
                except (ValueError, TypeError):
                    pass
    counts = [len(f) for f in frames] if frames else []
    if broadcast_pdf is not None and counts:
        rep_idx = np.repeat(np.arange(len(broadcast_pdf)), counts)
        broadcast_out = broadcast_pdf.iloc[rep_idx].reset_index(drop=True)
        result = pd.concat([broadcast_out, func_out.reset_index(drop=True)], axis=1)
    elif broadcast_pdf is not None:
        result = pd.concat([broadcast_pdf.iloc[:0], func_out.iloc[:0]], axis=1)
    else:
        result = func_out
    func_out_effective = func_out_cols or [
        c for c in result.columns if c not in broadcast_cols
    ]
    if func_out_effective:
        declared = build_flat_apply_output_order(
            visible_cols, subset_cols, func_out_effective, exclude_cols=exclude_cols
        )
        extras = [c for c in result.columns if c not in declared]
        result = result[declared + extras] if extras else result[declared]
    return result


def build_empty_flat_apply_frame(
    order_cols,
    subset_cols,
    func_out_cols,
    func_out_dtypes,
    by_cols,
    group_keys,
    key_dtypes,
    in_dtypes,
    declared_dtypes=None,
):
    """Build a 0-row DataFrame for an empty flat_apply partition.

    Carries the declared column order and dtypes so the empty partition does
    not collapse to zero columns or default to ``object``.  ``key_dtypes`` is
    a dict of key→dtype (prepended when ``group_keys``); ``in_dtypes`` is a
    dict of input-column→dtype for broadcast cols; ``declared_dtypes`` (from
    the declared output schema) takes priority for all columns.
    """

    declared_dtypes = declared_dtypes or {}
    key_cols = list(key_dtypes) if group_keys and by_cols else []
    ordered = build_flat_apply_output_order(
        order_cols, subset_cols, func_out_cols, exclude_cols=by_cols
    )
    data = {}
    for c in key_cols + ordered:
        if c in declared_dtypes:
            data[c] = pd.Series(dtype=declared_dtypes[c])
        elif c in key_dtypes:
            data[c] = pd.Series(dtype=key_dtypes[c])
        elif c in (func_out_dtypes or {}):
            data[c] = pd.Series(dtype=func_out_dtypes[c])
        elif c in in_dtypes:
            data[c] = pd.Series(dtype=in_dtypes[c])
        else:
            data[c] = pd.Series(dtype=object)
    return pd.DataFrame(data)


def groupby_key_output_dtypes(liteframe, groupby_params, inferred):
    """Prepend the groupby key dtypes to each inferred output schema.

    ``groupby.apply`` / ``groupby.apply_chunk`` include the grouping keys in
    their output: each per-group UDF result is prefixed with the group's key
    values, matching pandas ``groupby.apply(...).reset_index()``.  The
    declared output schema must therefore carry the key columns too --
    otherwise the runtime-emitted key columns disagree with the declared
    schema at the tiler boundary, and the fetched result silently drops the
    keys.  Key columns already present in the inferred schema (e.g. the UDF
    returned them explicitly) are not duplicated.

    ``inferred`` is a list of per-output dtypes (one per declared output);
    each entry gets the keys prepended independently.
    """
    by = groupby_params.get("by")
    by_cols = by if isinstance(by, (list, tuple)) else [by]
    key_dtypes = liteframe.dtypes[by_cols]
    merged = []
    for dts in inferred:
        missing = key_dtypes[~key_dtypes.index.isin(dts.index)]
        merged.append(pd.concat([missing, dts]) if not missing.empty else dts)
    return merged


def prepend_group_key_columns(frames, group_df, declared_dtypes, by_cols):
    """Prepend the group's key values to each frame when the columns are absent.

    Runtime half of the ``groupby_key_output_dtypes`` contract for groupby
    ``flat_apply``: the declared schema carries the prepended grouping keys,
    and the executor must materialize them on every per-group output unless
    the keys are already present (a UDF that returned them explicitly is not
    duplicated).  Shared by the SPE codegen adapter and the DPE executor so
    their per-group expansion behaves identically.

    ``group_df`` is the group's frame (pandas or polars), used to read the
    key values; ``declared_dtypes`` is the declared output schema for the
    first output (or None), so the prepended columns carry the declared key
    dtypes and validate against the schema.  Returns the frames unchanged
    when nothing is missing.
    """
    if pl is not None and isinstance(group_df, pl.DataFrame):
        key_lookup = lambda c: group_df[c][0]
    else:
        key_lookup = lambda c: group_df[c].iloc[0]
    key_dtypes = (
        {c: declared_dtypes[c] for c in by_cols if c in declared_dtypes.index}
        if declared_dtypes is not None
        else {}
    )
    # Compute the missing key set per output frame: a multi-output flat_apply
    # may have output 0 carry a key-named column while output 1 does not, so
    # deriving missing from frames[0] alone would skip the prepend for ALL
    # outputs and let output 1 diverge from its declared schema.  Mirrors
    # the per-output logic in ``groupby_key_output_dtypes``.
    prepended = []
    for o in frames:
        missing = [c for c in by_cols if c not in o.columns]
        if not missing:
            prepended.append(o)
            continue
        key_cols = [
            pd.Series([key_lookup(c)] * len(o), dtype=key_dtypes.get(c), name=c)
            for c in missing
        ]
        prepended.append(pd.concat(key_cols + [o], axis=1))
    return prepended


def merge_hidden_into_physical(liteframe, visible_dtypes, frame_metadata):
    """Carry preserved hidden columns into ``physical_dtypes``.

    The project operator includes hidden columns in ``physical_dtypes``
    (hidden names are a subset of the physical index); ``LiteFrameData.dtypes``
    then drops them to expose only the visible schema.  Functional APIs that
    preserve hidden metadata must do the same -- otherwise
    ``dtypes.drop(hidden)`` raises ``KeyError`` for hidden columns that are
    absent from ``physical_dtypes``.

    Under the apply(axis=1) / transform preserve policy, *all* hidden columns
    physically present in the input are carried through to the output as
    hidden -- they are stripped from the UDF input and re-appended after
    execution, regardless of the output's visible schema.  A hidden column
    cannot also be declared as a visible output column (via ``dtypes`` or a
    return annotation): the UDF never sees hidden columns, so it cannot
    produce one; naming a hidden column as an output is a contract error.
    Returns ``(physical_dtypes, reconciled_frame_metadata, n_physical_cols)``
    where ``n_physical_cols = len(physical_dtypes)`` (visible + carried
    hidden), suitable for ``shape[1]``.
    """
    hidden = (
        list(frame_metadata.hidden_columns)
        if frame_metadata and frame_metadata.hidden_columns
        else []
    )
    if not hidden or liteframe is None:
        return visible_dtypes, frame_metadata, len(visible_dtypes)

    # A hidden column declared as a visible output (via ``dtypes`` or a return
    # annotation) is a contract error: the UDF cannot produce it (hidden
    # columns are stripped from its input), and hidden columns are carried
    # through automatically regardless of the output schema.
    conflict = [h for h in hidden if h in visible_dtypes.index]
    if conflict:
        raise ValueError(
            f"hidden column(s) {conflict} cannot be declared as output "
            f"column(s) via `dtypes` or a return annotation: apply(axis=1) / "
            f"transform strip hidden columns from the UDF input and carry "
            f"them through to the output automatically. To let the UDF "
            f"produce a column, include it as a visible input column "
            f"(e.g. lf[col]) instead of naming it in the output schema."
        )

    in_physical = liteframe._physical_dtypes
    hidden_dtypes = in_physical[in_physical.index.isin(hidden)]
    if hidden_dtypes.empty:
        # No carryable hidden columns: drop the metadata so it stays
        # consistent with physical_dtypes.
        return visible_dtypes, None, len(visible_dtypes)

    physical_dtypes = pd.concat([visible_dtypes, hidden_dtypes])
    out_meta = FrameMetadata(hidden_columns=list(hidden_dtypes.index))
    return physical_dtypes, out_meta, len(physical_dtypes)


def build_functional_output(op, inputs, inferred, frame_metadata=None):
    """Build single or multiple output LiteFrames from inferred dtypes.

    When ``inferred`` is ``None`` (``skip_infer``), the output is untyped.
    A single-element ``inferred`` list produces one LiteFrame; a multi-element
    list produces one tileable per output (``n_output`` is set on the op).
    """
    if frame_metadata is None:
        frame_metadata = FrameMetadata()

    if inferred is None:
        # skip_infer: unknown output
        return op.new_liteframe(
            inputs,
            shape=(np.nan, np.nan),
            physical_dtypes=None,
            frame_metadata=frame_metadata,
        )

    in_lf = inputs[0] if inputs else None

    if len(inferred) == 1:
        out_dtypes = inferred[0]
        physical_dtypes, out_meta, n_cols = merge_hidden_into_physical(
            in_lf, out_dtypes, frame_metadata
        )
        return op.new_liteframe(
            inputs,
            shape=(np.nan, n_cols),
            physical_dtypes=physical_dtypes,
            frame_metadata=out_meta,
        )
    else:
        # Multiple outputs -- each output gets its own FrameMetadata copy
        # to avoid shared mutable state across outputs.  Hidden columns are
        # carried into each output's physical_dtypes (consistently with the
        # single-output path and the project operator).
        kws = []
        for out_dtypes in inferred:
            physical_dtypes, out_meta, n_cols = merge_hidden_into_physical(
                in_lf, out_dtypes, frame_metadata
            )
            kws.append(
                dict(
                    shape=(np.nan, n_cols),
                    physical_dtypes=physical_dtypes,
                    frame_metadata=out_meta,
                )
            )
        op.n_output = len(kws)
        op.output_types = [OutputType.liteframe] * len(kws)
        return op.new_tileables(inputs, kws=kws)


@contextlib.contextmanager
def make_callable_wrapper(func, args, kwargs):
    """Instantiate a callable class once, yielding a ``call`` closure.

    The class is instantiated once, ``__call__`` is invoked per
    element/row/chunk with ``end=True`` (each call is a complete
    invocation that must flush its output), and ``close()`` runs in the
    ``finally`` block — so cleanup is guaranteed even when ``__call__``
    raises.  ``end=True`` means "produce output now"; ``close()`` means
    "release resources".  For plain functions ``close`` is a no-op.

    A ``MarkedFunction``-wrapped class (from ``@with_resources`` /
    ``@with_python_requirements``) is unwrapped via ``unwrap_function``
    before the callable-class check so the class is instantiated and
    called via ``__call__``; scheduling metadata on ``opfunc.func`` is
    preserved.
    """
    underlying = unwrap_function(func)
    if isinstance(underlying, type):
        instance = underlying()
        sig = inspect.signature(instance.__call__)
        has_end = "end" in sig.parameters
        if has_end:

            def call(data, end=True):
                return instance(data, *args, end=end, **kwargs)

        else:

            def call(data, end=True):
                return instance(data, *args, **kwargs)

        close = getattr(instance, "close", None)
    else:

        def call(data, end=True):
            return func(data, *args, **kwargs)

        close = None
    try:
        yield call
    finally:
        if close is not None:
            close()


def call_user_func(func, data, args, kwargs):
    """Call a user UDF, handling the callable-class protocol.

    If ``func`` is a class, it is instantiated once, called with ``data``
    (passing ``end=True`` when ``__call__`` accepts ``end`` — signaling
    the UDF to flush its output), and ``close()`` is called if present —
    from a ``finally`` block so cleanup runs even when ``__call__``
    raises.  ``end=True`` means "produce output now"; ``close()`` means
    "release resources".  Plain functions are called directly.
    """
    kwargs = kwargs or {}

    underlying = unwrap_function(func)
    if isinstance(underlying, type):
        instance = underlying()
        sig = inspect.signature(instance.__call__)
        has_end = "end" in sig.parameters
        close = getattr(instance, "close", None)

        try:
            if has_end:
                result = instance(data, *args, end=True, **kwargs)
            else:
                result = instance(data, *args, **kwargs)
        finally:
            if close is not None:
                close()
        return result
    else:
        return func(data, *args, **kwargs)
