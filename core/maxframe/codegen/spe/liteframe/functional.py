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

import types
from typing import List, Optional

import pandas as pd

try:
    import polars as _pl
except ImportError:  # pragma: no cover
    _pl = None

from maxframe.codegen.spe.core import (
    SPECodeContext,
    SPEOperatorAdapter,
    register_op_adapter,
)
from maxframe.codegen.spe.liteframe.expressions import _materialize_range_columns
from maxframe.liteframe.functional.utils import (
    _stringify_frame_labels,
    assemble_flat_apply_output,
    build_empty_flat_apply_frame,
    call_user_func,
    classify_flat_apply_result,
    make_callable_wrapper,
    prepend_group_key_columns,
)
from maxframe.liteframe.operators.functional import LiteFrameFunctional
from maxframe.utils import unwrap_function, validate_and_align_output


def _flat_apply_is_iterable(res):
    """True if *res* is a generator or (non-string/dict/Series/Frame) list-like."""

    if isinstance(res, types.GeneratorType):
        return True
    if isinstance(res, (str, bytes, dict)):
        return False
    if _pl is not None and isinstance(res, (_pl.Series, _pl.DataFrame, _pl.LazyFrame)):
        return False
    if isinstance(res, (pd.DataFrame, pd.Series)):
        return False
    return hasattr(res, "__iter__")


def _flat_apply_row_to_dict(item, out_cols):
    """Normalize one flattened row-item to a ``{col: value}`` dict.

    Positional results align to ``out_cols`` with an exact width check; named
    results (dict / named Series) keep their natural keys.  ``out_cols=None``
    (untyped) uses positional integers / natural names.
    """

    def _positional(values, kind):
        if out_cols is None:
            return {i: v for i, v in enumerate(values)}
        if len(values) != len(out_cols):
            raise ValueError(
                f"flat_apply positional row ({kind}) has {len(values)} values "
                f"but {len(out_cols)} output columns were declared; the positional "
                f"width must match the declared output schema."
            )
        return {out_cols[i]: values[i] for i in range(len(out_cols))}

    if isinstance(item, dict):
        return dict(item)
    if isinstance(item, pd.Series):
        if list(item.index) == list(range(len(item))):
            return _positional(item.tolist(), "Series")
        return dict(item)
    if _pl is not None and isinstance(item, _pl.Series):
        return _positional(item.to_list(), "polars Series")
    if isinstance(item, (list, tuple)):
        return _positional(item, "list/tuple")
    if out_cols is None:
        return {0: item}
    if out_cols:
        return {out_cols[0]: item}
    return {}


def _flatten_flat_apply_result(res, out_cols):
    """Flatten one per-row ``flat_apply`` result into a 0-N row pandas DataFrame.

    Polars frames are converted to pandas first.  Natural column names are
    preserved so the caller's dtype alignment can reject/reorder; only the
    empty-result cases use ``out_cols`` for a stable zero-row schema.
    """

    def _empty():
        return (
            pd.DataFrame(columns=out_cols) if out_cols is not None else pd.DataFrame()
        )

    if res is None:
        return _empty()
    if _pl is not None and isinstance(res, (_pl.DataFrame, _pl.LazyFrame)):
        res = res.collect() if isinstance(res, _pl.LazyFrame) else res
        res = res.to_pandas(use_pyarrow_extension_array=True)
    if isinstance(res, pd.DataFrame):
        return res
    items = list(res) if _flat_apply_is_iterable(res) else [res]
    if not items:
        return _empty()
    return pd.DataFrame([_flat_apply_row_to_dict(it, out_cols) for it in items])


def _flat_apply_plan(
    subset, visible_cols, output_dtypes, n_outputs, by_cols=None, group_keys=True
):
    """Derive ``(subset_cols, broadcast_cols, func_out_cols, func_out_dtypes)``.

    Function-output cols/dtypes are the declared full output schema minus the
    broadcast columns, so they line up with the columns the UDF produces.
    For groupby flat_apply the declared schema also carries the prepended
    grouping keys (group_keys=True); they are excluded from the function
    output and re-attached by the executor instead.
    """
    if subset is None:
        subset_cols = list(visible_cols)
        broadcast_cols = []
    else:
        subset_cols = list(subset)
        broadcast_set = set(subset_cols)
        broadcast_cols = [c for c in visible_cols if c not in broadcast_set]
    broadcast_set = set(broadcast_cols) | (set(by_cols or []) if group_keys else set())

    func_out_cols = []
    func_out_dtypes = []
    for i in range(n_outputs):
        dt = (
            output_dtypes[i]
            if output_dtypes is not None and i < len(output_dtypes)
            else None
        )
        if dt is not None and len(dt):
            cols = [c for c in dt.index if c not in broadcast_set]
            func_out_cols.append(cols)
            func_out_dtypes.append({c: dt[c] for c in cols})
        else:
            func_out_cols.append(None)
            func_out_dtypes.append(None)
    return subset_cols, broadcast_cols, func_out_cols, func_out_dtypes


def _flat_apply_expand_frame(
    frame,
    visible_cols,
    subset_cols,
    broadcast_cols,
    func,
    args,
    kwargs,
    input_frame_type,
    n_outputs,
    func_out_cols_per_output,
    func_out_dtypes_per_output,
    exclude_cols=None,
):
    """Expand one frame's rows via per-row ``flat_apply``; return per-output PDFs.

    Each per-row result is flattened to 0-N rows; broadcast columns are
    repeated (``np.repeat``), function-output columns cast to declared
    dtypes, and column order enforced: func-out cols inserted at the first
    subset column's position, broadcast cols keep relative order.
    """

    n_rows = len(frame)
    udf_data = frame[subset_cols] if subset_cols else frame.iloc[:, :0]
    broadcast_pdf = frame[broadcast_cols] if broadcast_cols else None

    # Build the per-row inputs in the user-requested frame type.
    if input_frame_type == "polars" and _pl is not None:
        udf_pl = (
            _pl.from_pandas(udf_data)
            if not isinstance(udf_data, _pl.DataFrame)
            else udf_data
        )
        row_iter = (udf_pl.slice(i, 1) for i in range(n_rows))
    else:
        row_iter = (udf_data.iloc[i] for i in range(n_rows))

    kw = kwargs or {}
    per_row_frames = [[] for _ in range(n_outputs)]
    with make_callable_wrapper(func, args, kw) as call:
        for _i, row in enumerate(row_iter):
            res = call(row)
            for out_idx, (sub_res, out_cols) in enumerate(
                classify_flat_apply_result(res, n_outputs, func_out_cols_per_output)
            ):
                per_row_frames[out_idx].append(
                    _flatten_flat_apply_result(sub_res, out_cols)
                )

    return tuple(
        assemble_flat_apply_output(
            per_row_frames[out_idx],
            func_out_cols_per_output[out_idx],
            func_out_dtypes_per_output[out_idx],
            broadcast_pdf,
            broadcast_cols,
            visible_cols,
            subset_cols,
            exclude_cols=exclude_cols,
        )
        for out_idx in range(n_outputs)
    )


def _run_flat_apply(
    df,
    func,
    args,
    kwargs,
    groupby_params,
    subset,
    visible_cols,
    output_dtypes,
    n_outputs,
    input_frame_type,
):
    """Execute ``flat_apply``: a per-row UDF that expands each row to 0-N rows.

    Mirrors the DPE ``FlatApplyRunner`` / ``GroupByFlatApplyRunner``: the UDF
    receives one row built from ``subset``; the remaining columns are
    broadcast.  A tuple result denotes multiple outputs.
    """
    by_cols = None
    if groupby_params is not None:
        by = groupby_params["by"]
        by_cols = [by] if isinstance(by, str) else list(by)
        # Exclude the grouping keys from the effective columns so they are
        # never broadcast: for flat_apply the group key is governed solely
        # by the ``group_keys`` contract (prepended / omitted), never by the
        # broadcast path -- even when an explicit selection includes it.
        selection = groupby_params.get("selection")
        if selection is not None:
            by_set = set(by_cols)
            effective_cols = [c for c in selection if c not in by_set]
        else:
            effective_cols = [c for c in visible_cols if c not in set(by_cols)]
    else:
        effective_cols = visible_cols
    # Columns used for output ordering: the selection (incl. the key) when
    # explicit, otherwise the full visible cols.  The key's position here
    # anchors the func-out block; ``exclude_cols=by_cols`` removes it from
    # the broadcast result.
    order_cols = (
        list(selection)
        if groupby_params is not None and selection is not None
        else visible_cols
    )
    subset_cols, broadcast_cols, func_out_cols, func_out_dtypes = _flat_apply_plan(
        subset,
        effective_cols,
        output_dtypes,
        n_outputs,
        by_cols=by_cols,
        group_keys=(
            groupby_params.get("group_keys", True)
            if groupby_params is not None
            else True
        ),
    )

    def _expand_block(frame):
        return _flat_apply_expand_frame(
            frame,
            order_cols,
            subset_cols,
            broadcast_cols,
            func,
            args,
            kwargs,
            input_frame_type,
            n_outputs,
            func_out_cols,
            func_out_dtypes,
            exclude_cols=by_cols,
        )

    if groupby_params is not None:
        dropna = groupby_params.get("dropna", True)
        group_keys = groupby_params.get("group_keys", True)
        sort = groupby_params.get("sort", True)
        results_per_output = [[] for _ in range(n_outputs)]
        for _key, g in df.groupby(by_cols, dropna=dropna, sort=sort):
            outs = _expand_block(g)
            if group_keys:
                # apply-family key contract: prepend the group's key values to
                # each expanded output unless group_keys=False (keys the frames
                # already carry are not duplicated).
                declared0 = (
                    output_dtypes[0]
                    if output_dtypes is not None and len(output_dtypes)
                    else None
                )
                outs = prepend_group_key_columns(outs, g, declared0, by_cols)
            for out_idx in range(n_outputs):
                results_per_output[out_idx].append(outs[out_idx])
        outputs = []
        for out_idx in range(n_outputs):
            frames = results_per_output[out_idx]
            if frames:
                outputs.append(pd.concat(frames, ignore_index=True))
            else:
                declared_i = (
                    output_dtypes[out_idx]
                    if output_dtypes is not None and out_idx < len(output_dtypes)
                    else None
                )
                key_dtypes = {}
                if group_keys and by_cols and declared_i is not None:
                    key_dtypes = {
                        c: declared_i[c] for c in by_cols if c in declared_i.index
                    }
                elif group_keys and by_cols:
                    key_dtypes = {c: df[c].dtype for c in by_cols if c in df.columns}
                in_dtypes = {c: df[c].dtype for c in df.columns}
                declared_dict = dict(declared_i) if declared_i is not None else None
                outputs.append(
                    build_empty_flat_apply_frame(
                        order_cols,
                        subset_cols,
                        func_out_cols[out_idx],
                        func_out_dtypes[out_idx],
                        by_cols,
                        group_keys,
                        key_dtypes,
                        in_dtypes,
                        declared_dtypes=declared_dict,
                    )
                )
        return tuple(outputs) if n_outputs > 1 else outputs[0]

    outs = _expand_block(df)
    return tuple(outs) if n_outputs > 1 else outs[0]


def _normalize_group_result(result):
    """Normalize a per-group apply UDF result into a DataFrame.

    Mirrors the DPE ``_normalize_group_result``: a groupby apply UDF may
    return a DataFrame, Series, dict, or scalar per group.  Each is coerced
    to a DataFrame so ``pd.concat`` never raises ``TypeError`` on a bare
    scalar/dict (the previous code only handled ``pd.Series``).
    """
    if isinstance(result, pd.DataFrame):
        return result
    if isinstance(result, pd.Series):
        # A named Series -> one column; an indexed Series -> columns from index
        return result.to_frame().T if result.name else pd.DataFrame([result])
    if isinstance(result, dict):
        return pd.DataFrame([result])
    # scalar
    return pd.DataFrame([result])


def _resolve_groupby_selection(group_df, groupby_params):
    """Resolve the UDF-visible columns for a groupby apply/apply_chunk frame.

    Following pandas 3.0 semantics (``docs/designs/liteframe/operators/
    functional.md``): an unset ``selection`` excludes the groupby key columns;
    an explicit selection (incl. an empty ``[]``) passes exactly those
    columns.  The ``is not None`` check (not truthiness) is critical so an
    explicit empty selection yields zero UDF-visible columns instead of
    falling through to the key-exclusion default.
    """
    selection = groupby_params.get("selection")
    if selection is not None:
        return group_df[list(selection)] if selection else group_df.iloc[:, :0]
    by = groupby_params.get("by")
    by_cols = by if isinstance(by, (list, tuple)) else [by]
    return group_df[[c for c in group_df.columns if c not in by_cols]]


def _run_groupby_functional(
    df,
    func,
    mode,
    axis,
    args,
    kwargs,
    groupby_params,
    hidden_columns,
    input_frame_type="pandas",
    output_dtypes=None,
    check_output_dtypes="raises",
    n_outputs=1,
):
    """Execute groupby apply / apply_chunk / transform via pandas groupby.

    Mirrors the DPE runner semantics:
    - apply / apply_chunk: per-group UDF, row count/order may change.
    - transform: row-count-preserving, output aligned to input order.

    ``input_frame_type`` is applied at the UDF boundary for apply/apply_chunk
    so a grouped ``input_frame_type="polars"`` UDF receives a polars frame,
    matching the non-grouped path (mirrors DPE GroupByFunctionalRunner).

    When ``n_outputs > 1`` the UDF is expected to return a tuple of frames per
    group; each output is concatenated across groups and the function returns a
    tuple.  An arity mismatch (non-tuple or wrong length) raises ``ValueError``.
    """
    by = groupby_params["by"]
    by_cols = [by] if isinstance(by, str) else list(by)
    dropna = groupby_params.get("dropna", True)
    # Forward sort so groupby(..., sort=False) preserves group-encounter order
    # instead of inheriting pandas' sort=True default (which would reorder
    # group keys, e.g. ["b","a"] -> ["a","b"]).
    sort = groupby_params.get("sort", True)
    kw = kwargs or {}
    grouped = df.groupby(by_cols, dropna=dropna, sort=sort)
    want_polars = input_frame_type == "polars" and _pl is not None
    adapter = LiteFrameFunctionalAdapter

    if mode in ("apply", "apply_chunk"):
        group_keys = groupby_params.get("group_keys", True)
        multi_output = n_outputs > 1

        # Normalize a per-group UDF result, prepend missing group keys.
        def _normalize_and_prepend(elem, g):
            result = _normalize_group_result(adapter._from_polars_result(elem))
            missing = [c for c in by_cols if c not in result.columns]
            if missing and group_keys:
                key_df = pd.DataFrame(
                    {c: g[c].iloc[0] for c in missing}, index=result.index
                )
                result = pd.concat([key_df, result], axis=1)
            return result

        def _empty_from_dtypes(single_dtypes=None):
            """Zero-row frame from declared output schema (or ``df.iloc[:0]``)."""
            if single_dtypes is not None:
                return pd.DataFrame(
                    {c: pd.Series(dtype=dt) for c, dt in single_dtypes.items()}
                )
            return df.iloc[:0]

        results_per_output = [[] for _ in range(n_outputs)] if multi_output else None
        results = []
        for _key, g in grouped:
            udf_data = _resolve_groupby_selection(g, groupby_params)
            if want_polars:
                udf_data = adapter._to_polars_frame(udf_data)
            res = call_user_func(func, udf_data, args, kw)
            res = adapter._from_polars_result(res)
            if multi_output:
                if not isinstance(res, tuple) or len(res) != n_outputs:
                    raise ValueError(
                        f"groupby multi-output UDF returned {type(res).__name__} "
                        f"with {len(res) if isinstance(res, tuple) else 0} "
                        f"elements but {n_outputs} outputs were declared; "
                        f"every group must return a tuple of exactly "
                        f"{n_outputs} elements."
                    )
                for i, elem in enumerate(res):
                    results_per_output[i].append(_normalize_and_prepend(elem, g))
            else:
                results.append(_normalize_and_prepend(res, g))

        if multi_output:
            if not results_per_output[0]:
                return tuple(
                    _empty_from_dtypes(output_dtypes[i] if output_dtypes else None)
                    for i in range(n_outputs)
                )
            return tuple(
                pd.concat(frames, ignore_index=True) for frames in results_per_output
            )
        if not results:
            single_dtypes = (
                output_dtypes[0]
                if output_dtypes and isinstance(output_dtypes, list)
                else output_dtypes
            )
            return _empty_from_dtypes(single_dtypes)
        return pd.concat(results, ignore_index=True)

    if mode == "transform":
        # Transform is row-count-preserving and aligned to input order.
        # The grouping keys are excluded from the operation (pandas 3.0
        # ``include_groups=False`` semantics; ``DataFrameGroupBy.transform``
        # never passes them to the UDF) and from the output -- the result
        # carries only the transformed non-key columns, matching the DPE
        # ``GroupByTransformRunner`` and the declared schema.
        if axis == 0:
            # Respect the gb[...] selection (pandas 3.0 semantics): an unset
            # selection transforms all non-key columns; an explicit selection
            # (incl. []) transforms exactly those columns.  Mirrors the apply
            # branch's use of ``_resolve_groupby_selection`` and the DPE
            # ``GroupByTransformRunner``.
            selection = groupby_params.get("selection")
            if selection is not None:
                transform_cols = list(selection)
            else:
                transform_cols = [c for c in df.columns if c not in by_cols]
            col_results = {}
            if isinstance(unwrap_function(func), type):
                # Callable class: fresh instance + close() per group,
                # mirroring DPE GroupByTransformRunner (call_user_func per
                # group).  Pandas' grouped[col].transform would share one
                # instance across all groups.
                for _key, g in grouped:
                    with make_callable_wrapper(func, args, kw) as call:
                        for col in transform_cols:
                            res = call(g[col])
                            col_results.setdefault(col, []).append(res)
                for col in transform_cols:
                    # Reindex to original input order: group-iteration
                    # order (a: [0,2], b: [1,3]) reorders interleaved rows.
                    col_results[col] = pd.concat(col_results[col]).reindex(df.index)
            else:
                with make_callable_wrapper(func, args, kw) as call:
                    for col in transform_cols:
                        col_results[col] = grouped[col].transform(call)
            return pd.DataFrame(col_results)
        # axis=1 transform: apply per row then group-transform is unusual;
        # fall back to per-group apply with realignment.
        results = []
        with make_callable_wrapper(func, args, kw) as call:
            for _key, g in grouped:
                res = g.apply(call, axis=1)
                if isinstance(res, pd.Series):
                    res = res.to_frame()
                results.append(res)
        return pd.concat(results)

    raise ValueError(f"Unsupported groupby functional mode: {mode!r} axis={axis!r}")


@register_op_adapter(LiteFrameFunctional)
class LiteFrameFunctionalAdapter(SPEOperatorAdapter):
    @staticmethod
    def _to_polars_frame(pdf):
        """Convert a pandas frame to polars for a polars-mode UDF boundary.

        polars is an optional core dependency (lazily imported); a user that
        explicitly requests ``input_frame_type="polars"`` gets an ImportError
        at execution if it is absent, matching the convention in
        ``projection.py`` and ``utils.py``.
        """
        import polars as pl

        return pl.from_pandas(pdf)

    @staticmethod
    def _from_polars_result(result):
        """Normalize a UDF result back to pandas (SPE's storage form).

        A polars-mode UDF may return a polars DataFrame/Series; convert it to
        pandas so downstream hidden-column reattachment and output validation
        operate on SPE's native frame type.
        """
        if _pl is not None and isinstance(result, _pl.DataFrame):
            return result.to_pandas(use_pyarrow_extension_array=True)
        if _pl is not None and isinstance(result, _pl.Series):
            return result.to_frame().to_pandas(use_pyarrow_extension_array=True)
        return result

    @staticmethod
    def _validate_and_align(result, output_dtypes, check_output_dtypes):
        """Validate/align a functional UDF result against the declared schema.

        Routes the result through the shared ``validate_and_align_output`` so
        a UDF returning wrong column names or incompatible dtypes raises under
        the default ``"raises"`` policy instead of silently passing through --
        mirroring the DPE ``FunctionalUDFRunner._validate_output`` lifecycle
        and the Output Alignment contract.  ``output_dtypes`` is a list of
        per-output ``pd.Series``; the single-output schema is used (SPE's
        apply/transform paths produce one frame per call).  No declared schema
        or a non-DataFrame result skips validation.

        ``normalize_str_columns=True`` is always passed: LiteFrame column names
        are strings, but SPE scalar paths (``apply(axis=1)``, groupby-apply)
        produce pandas frames with an integer label (0).  Canonicalization
        runs inside ``validate_and_align_output`` before the dtype-policy early
        exit, so even ``"ignore"`` returns a string-labeled frame.

        When no schema is declared (``output_dtypes`` is ``None`` / empty --
        the public ``skip_infer=True`` path), the frame is still returned to
        the user, so its column labels must be canonicalized to strings here
        too: a scalar UDF result normalizes to a pandas frame with an integer
        label (0), which would survive unchanged without this boundary guard.
        """
        if not isinstance(result, pd.DataFrame):
            return result
        # No declared schema (skip_infer): still canonicalize column labels to
        # strings at the frame boundary so a scalar/positional UDF result (whose
        # pandas frame carries an integer label 0) returns "0" to the user.
        if not output_dtypes:
            return _stringify_frame_labels(result)
        expected = output_dtypes[0]
        if expected is None or len(expected) == 0:
            return _stringify_frame_labels(result)
        return validate_and_align_output(
            result,
            expected,
            check_output_dtypes=check_output_dtypes,
            normalize_str_columns=True,
        )

    @staticmethod
    def _validate_outputs(result, output_dtypes, check_output_dtypes):
        """Validate a single or multi-output result at the return boundary.

        Multi-output ``output_dtypes`` is a per-output list; validate output
        ``i`` against its own schema (wrapped as a single-element list, which
        ``_validate_and_align`` indexes at ``[0]``).
        """
        if isinstance(result, tuple):
            return tuple(
                LiteFrameFunctionalAdapter._validate_and_align(
                    r,
                    [output_dtypes[i]] if output_dtypes else None,
                    check_output_dtypes,
                )
                for i, r in enumerate(result)
            )
        return LiteFrameFunctionalAdapter._validate_and_align(
            result, output_dtypes, check_output_dtypes
        )

    @staticmethod
    def _broadcast_hidden(result, df, hidden_columns, hidden_set):
        """Append hidden columns from ``df`` when the UDF did not return them.

        Row-count-preserving modes (apply_chunk, apply/transform axis=0,
        apply axis=1) broadcast hidden columns into the result when the UDF
        dropped them; row-expanding modes (flat_apply) drop hidden columns at
        the API and never reach here.  Only appends when the result is a
        frame of the same length as ``df``.
        """
        if hidden_set and isinstance(result, pd.DataFrame) and len(result) == len(df):
            for hc in hidden_columns:
                if hc in df.columns and hc not in result.columns:
                    result[hc] = df[hc].to_numpy()
        return result

    @staticmethod
    def _run_functional(
        df: pd.DataFrame,
        func,
        mode: str,
        axis: int,
        args: tuple,
        kwargs: Optional[dict],
        result_type=None,
        hidden_columns=None,
        range_columns: dict = None,
        groupby_params: Optional[dict] = None,
        subset=None,
        output_dtypes=None,
        input_frame_type: str = "pandas",
        n_output: Optional[int] = None,
        check_output_dtypes: str = "raises",
    ):
        df = _materialize_range_columns(df, range_columns)

        hidden_set = set(hidden_columns) if hidden_columns else set()
        visible_cols = [c for c in df.columns if c not in hidden_set]
        kw = kwargs or {}
        want_polars = input_frame_type == "polars" and _pl is not None

        if mode == "flat_apply":
            n_outputs = n_output or 1
            result = _run_flat_apply(
                df,
                func,
                args,
                kw,
                groupby_params,
                subset,
                visible_cols,
                output_dtypes,
                n_outputs,
                input_frame_type,
            )
            return LiteFrameFunctionalAdapter._validate_outputs(
                result, output_dtypes, check_output_dtypes
            )

        if groupby_params is not None:
            n_outputs = n_output or 1
            result = _run_groupby_functional(
                df,
                func,
                mode,
                axis,
                args,
                kw,
                groupby_params,
                hidden_columns,
                input_frame_type,
                output_dtypes,
                check_output_dtypes,
                n_outputs,
            )
            return LiteFrameFunctionalAdapter._validate_outputs(
                result, output_dtypes, check_output_dtypes
            )

        if mode == "apply_chunk":
            udf_data = df[visible_cols] if hidden_set else df
            if want_polars:
                udf_data = LiteFrameFunctionalAdapter._to_polars_frame(udf_data)
            result = call_user_func(func, udf_data, args, kw)
            result = LiteFrameFunctionalAdapter._from_polars_result(result)
            if isinstance(result, pd.Series):
                result = result.to_frame().T
            result = LiteFrameFunctionalAdapter._broadcast_hidden(
                result, df, hidden_columns, hidden_set
            )
            return LiteFrameFunctionalAdapter._validate_and_align(
                result, output_dtypes, check_output_dtypes
            )

        if mode in ("apply", "transform") and axis == 0:
            col_results = {}
            with make_callable_wrapper(func, args, kw) as call:
                for col in visible_cols:
                    col_data = df[col]
                    if want_polars:
                        col_data = LiteFrameFunctionalAdapter._to_polars_frame(
                            col_data.to_frame()
                        )[col]
                    col_results[col] = call(col_data)
            # All-scalar results: pd.DataFrame({col: scalar}) raises
            # "If using all scalar values, you must pass an index"; build a
            # one-row frame instead (mirrors DPE Axis0FunctionalRunner).
            if all(not isinstance(v, pd.Series) for v in col_results.values()):
                result = pd.DataFrame([col_results])
            else:
                result = pd.DataFrame(col_results)
            result = LiteFrameFunctionalAdapter._broadcast_hidden(
                result, df, hidden_columns, hidden_set
            )
            return LiteFrameFunctionalAdapter._validate_and_align(
                result, output_dtypes, check_output_dtypes
            )

        if mode == "apply" and axis == 1:
            udf_data = df[visible_cols] if hidden_set else df
            with make_callable_wrapper(func, args, kw) as call:
                if want_polars:
                    udf_data = LiteFrameFunctionalAdapter._to_polars_frame(udf_data)
                    # Polars has no axis=1 apply; iterate rows as single-row
                    # frames (mirrors DPE apply_polars_rowwise).
                    rows = []
                    for i in range(len(udf_data)):
                        rows.append(call(udf_data.slice(i, 1)))
                    result = pd.Series(rows)
                elif result_type is not None:
                    result = udf_data.apply(call, axis=1, result_type=result_type)
                else:
                    result = udf_data.apply(call, axis=1)
            result = LiteFrameFunctionalAdapter._from_polars_result(result)
            if isinstance(result, pd.Series):
                result = result.to_frame()
            result = LiteFrameFunctionalAdapter._broadcast_hidden(
                result, df, hidden_columns, hidden_set
            )
            return LiteFrameFunctionalAdapter._validate_and_align(
                result, output_dtypes, check_output_dtypes
            )

        raise ValueError(f"Unsupported functional mode: {mode!r} axis={axis!r}")

    def generate_code(self, op, context: SPECodeContext) -> List[str]:
        cls_name = type(self).__name__
        context.register_import(__name__, from_item=cls_name)
        input_var = context.get_input_tileable_variable(op.inputs[0])
        n_output = getattr(op, "n_output", None) or 1
        # flat_apply and groupby apply/apply_chunk can produce a tuple for
        # multi-output; transform and non-groupby axis-0 apply produce a
        # single frame.
        is_multi_output_mode = op.mode == "flat_apply" or (
            op.mode in ("apply", "apply_chunk") and op.groupby_params is not None
        )
        multi_output = is_multi_output_mode and n_output > 1
        res_vars = [context.get_output_tileable_variable(out) for out in op.outputs]

        range_columns = None
        in_tileable = op.inputs[0]
        if hasattr(in_tileable, "frame_metadata") and in_tileable.frame_metadata:
            if in_tileable.frame_metadata.range_columns:
                range_columns = {}
                for name, info in in_tileable.frame_metadata.range_columns.items():
                    range_columns[name] = (info.start, info.stop, info.step)

        hidden_columns = getattr(op, "hidden_columns", None)

        args = self._translate_call_args(
            context,
            op.func,
            op.mode,
            op.axis,
            op.args,
            op.kwargs,
            op.result_type,
            hidden_columns,
            range_columns=range_columns,
            groupby_params=op.groupby_params,
            subset=op.subset,
            output_dtypes=op.output_dtypes,
            input_frame_type=op.input_frame_type,
            n_output=n_output if multi_output else None,
            check_output_dtypes=getattr(op, "check_output_dtypes", "raises"),
        )
        call = f"{cls_name}._run_functional({input_var}, {', '.join(args)})"
        if multi_output:
            # Multi-output: _run_functional returns a tuple; unpack into the
            # per-output variables.  A single trailing comma is needed when
            # there is exactly one output var to keep tuple-unpacking syntax.
            targets = ", ".join(res_vars)
            if len(res_vars) == 1:
                targets += ","
            return [f"{targets} = {call}"]
        return [f"{res_vars[0]} = {call}"]
