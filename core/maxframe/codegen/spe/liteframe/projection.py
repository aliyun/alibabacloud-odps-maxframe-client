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

from typing import List, Optional

import pandas as pd

from maxframe.codegen.spe.core import (
    SPECodeContext,
    SPEOperatorAdapter,
    register_op_adapter,
)
from maxframe.codegen.spe.liteframe.expressions import (
    LiteFrameExprEvaluator,
    _materialize_range_columns,
)
from maxframe.liteframe.expressions import (
    LiteFrameColumn,
    LiteFrameExpr,
    LiteFrameMultiColumnExpr,
    LiteFrameMultiColumnSubExpr,
    LiteFrameUDFExpr,
    NamedExpr,
)
from maxframe.liteframe.functional.utils import make_callable_wrapper
from maxframe.liteframe.operators.project import LiteFrameProjection, _projection_name
from maxframe.utils import validate_and_align_output


def _extract_input_range_columns(tileable) -> dict:
    """Return ``{name: (start, stop, step)}`` for the input's virtual range
    columns, or ``None`` when the input carries no range columns."""
    metadata = getattr(tileable, "frame_metadata", None)
    if not metadata or not metadata.range_columns:
        return None
    return {
        name: (info.start, info.stop, info.step)
        for name, info in metadata.range_columns.items()
    }


# ---------------------------------------------------------------------------
# UDF projection helpers
# ---------------------------------------------------------------------------


def _get_udf_expr(proj) -> Optional[LiteFrameUDFExpr]:
    """Extract a ``LiteFrameUDFExpr`` from a projection if it references one.

    ``map()`` wraps ``LiteFrameUDFExpr`` directly in ``NamedExpr``;
    ``transform_chunk`` / ``transform_row`` wrap via
    ``LiteFrameMultiColumnSubExpr``.
    """
    expr = proj.expr if isinstance(proj, NamedExpr) else proj
    if isinstance(expr, LiteFrameUDFExpr):
        return expr
    if isinstance(expr, LiteFrameMultiColumnSubExpr) and isinstance(
        expr.multi_expr, LiteFrameUDFExpr
    ):
        return expr.multi_expr
    return None


def _udf_input_data(udf_expr: LiteFrameUDFExpr, df: pd.DataFrame):
    """Select the UDF's declared input columns, converted to the requested
    frame type (pandas or polars)."""
    input_columns = [op.name for op in udf_expr.operands]
    subset = df[input_columns] if input_columns else df.iloc[:, :0]
    if udf_expr.input_frame_type == "polars":
        import polars as pl

        return pl.from_pandas(subset)
    return subset


def _normalize_rowwise_result(
    result: pd.Series, output_names: list, dtypes_dict: Optional[dict] = None
) -> pd.DataFrame:
    """Normalize a 1-D ``pd.Series`` of per-row UDF results into a DataFrame.

    Each element may be a scalar, ``dict``, or ``pd.Series``.  Dict / named
    Series results keep their **natural labels** so a wrong-key result reaches
    ``validate_and_align_output`` and raises under the default ``"raises"``
    policy instead of being silently reindexed to the declared columns (which
    would discard unexpected labels and NaN-fill the declared ones).  A plain
    RangeIndex Series is treated as positional.  List / tuple results are
    mapped positionally to ``output_names``.  Mismatched dict/Series keys are
    NaN-filled (pandas apply(axis=1) semantics).
    """
    out_cols = list(output_names) if output_names else None
    if len(result) == 0:
        if dtypes_dict and out_cols:
            return pd.DataFrame(
                {
                    col: pd.Series(
                        dtype=pd.api.types.pandas_dtype(dtypes_dict.get(col))
                    )
                    for col in out_cols
                }
            )
        return pd.DataFrame(columns=out_cols) if out_cols else pd.DataFrame()

    rows = []
    for val in result:
        if isinstance(val, dict):
            rows.append(val)
        elif isinstance(val, pd.Series):
            rows.append(dict(val))
        elif isinstance(val, (list, tuple)):
            rows.append(
                {out_cols[i]: val[i] for i in range(len(val))}
                if out_cols
                else {i: val[i] for i in range(len(val))}
            )
        else:
            # scalar
            if out_cols and len(out_cols) == 1:
                rows.append({out_cols[0]: val})
            else:
                rows.append({0: val})
    return pd.DataFrame(rows)


def _udf_expected_dtypes(udf_expr: LiteFrameUDFExpr) -> Optional[pd.Series]:
    """Build the declared output schema as a ``pd.Series`` for validation.

    Returns ``None`` when no dtypes are declared (skips validation), matching
    ``validate_and_align_output``'s ``expected_dtypes=None`` pass-through.
    """
    dtypes_dict = getattr(udf_expr, "dtypes", None)
    output_names = list(udf_expr.output_names or [])
    if not dtypes_dict or not output_names:
        return None
    return pd.Series(
        {n: pd.api.types.pandas_dtype(dtypes_dict[n]) for n in output_names}
    )


def _validate_udf_output(udf_expr: LiteFrameUDFExpr, result) -> pd.DataFrame:
    """Validate/align a UDF result against the declared output schema.

    Routes the result through the shared ``validate_and_align_output`` so a
    UDF that returns wrong column names or incompatible dtypes raises (under
    the default ``"raises"`` policy) instead of being silently reindexed to
    all-null or miscast -- mirroring the DPE ``ProjectionUDFRunner`` lifecycle
    and the Output Alignment contract in
    ``docs/designs/liteframe/operators/functional.md``.
    """
    expected = _udf_expected_dtypes(udf_expr)
    policy = getattr(udf_expr, "check_output_dtypes", "raises") or "raises"
    if not isinstance(result, pd.DataFrame):
        result = result.to_frame(
            name=(udf_expr.output_names[0] if udf_expr.output_names else 0)
        )
    if expected is None:
        return result
    return validate_and_align_output(
        result, expected, check_output_dtypes=policy, normalize_str_columns=True
    )


def _run_udf(udf_expr: LiteFrameUDFExpr, df: pd.DataFrame) -> pd.DataFrame:
    """Execute a single projection UDF and return its output as a DataFrame.

    Mirrors the DPE ``ProjectionUDFRunner`` dispatch by ``mode`` but uses
    pandas directly (SPE's execution backend).  Every mode normalizes its
    result through ``_validate_udf_output`` so missing/extra columns and
    incompatible dtypes are rejected under the declared ``check_output_dtypes``
    policy instead of being silently reindexed or miscast.
    """
    func = udf_expr.func
    args = udf_expr.args or ()
    kwargs = udf_expr.kwargs or {}
    mode = udf_expr.mode
    output_names = list(udf_expr.output_names or [])
    input_data = _udf_input_data(udf_expr, df)

    if mode == "map":
        # Element-wise UDF on the whole frame.  Use ``make_callable_wrapper``
        # so a callable class is instantiated once, called per element, and
        # ``close()`` is called at chunk end (mirrors DPE's
        # ``MapUDFRunner``).  Plain functions are called directly.
        pdf = input_data
        with make_callable_wrapper(func, args, kwargs) as call:
            if hasattr(pdf, "map"):
                result = pdf.map(call)
            else:  # pragma: no cover - older pandas
                result = pdf.applymap(call)
        return _validate_udf_output(udf_expr, result)

    if mode == "transform_chunk":
        with make_callable_wrapper(func, args, kwargs) as call:
            result = call(input_data)
        if isinstance(result, pd.Series):
            col = output_names[0] if output_names else (result.name or 0)
            result = result.to_frame(name=col)
        if not isinstance(result, pd.DataFrame):
            raise TypeError(
                f"transform_chunk UDF must return a DataFrame/Series, got "
                f"{type(result).__name__}"
            )
        in_rows = len(input_data)
        if len(result) != in_rows:
            raise ValueError(
                f"transform_chunk requires output row count to match input "
                f"({in_rows} rows), but the UDF returned {len(result)} rows. "
                f"Use apply_chunk() instead if you intend to change the row "
                f"count."
            )
        return _validate_udf_output(udf_expr, result)

    if mode == "transform_row":
        with make_callable_wrapper(func, args, kwargs) as call:
            if udf_expr.input_frame_type == "polars":
                # Polars row-mode: pass each row as a single-row polars
                # DataFrame.  ``input_data`` is already a polars DataFrame
                # from _udf_input_data.
                results = []
                for i in range(len(input_data)):
                    results.append(call(input_data.slice(i, 1)))
                raw = pd.Series(results)
            else:
                raw = input_data.apply(call, axis=1)
        if isinstance(raw, pd.DataFrame):
            return _validate_udf_output(udf_expr, raw)
        # Scalar / dict / Series per-row results: normalize to a frame first,
        # then validate against the declared schema.
        normalized = _normalize_rowwise_result(
            raw, output_names, getattr(udf_expr, "dtypes", None)
        )
        return _validate_udf_output(udf_expr, normalized)

    raise ValueError(f"Unknown UDF mode: {mode!r}")


def _resolve_range_state(op, in_range_columns):
    """Mirror the DPE ``_resolve_range_columns`` logic.

    Returns ``(virtual_range_names, need_materialize)``:

    * ``virtual_range_names`` — projection output names that stay virtual
      (affine range transforms or pass-through range columns). Their
      projections are skipped because the column is not materialized in the
      physical frame.
    * ``need_materialize`` — ``True`` when any range-referencing projection
      is non-affine, forcing all virtual range columns to be materialized
      into physical Int64 columns before evaluation.
    """
    virtual_range_names = set()
    need_materialize = False

    if not in_range_columns:
        return virtual_range_names, need_materialize

    range_analysis = op.range_analysis or {}
    for name, analysis in range_analysis.items():
        if analysis.is_affine and analysis.references_range:
            virtual_range_names.add(name)
        elif analysis.references_range:
            need_materialize = True

    # Range columns not mentioned in range_analysis survive only when they
    # appear in the output projections as a pass-through (same name).
    output_names = {_projection_name(p) for p in op.projections}
    for rc_name in in_range_columns:
        if rc_name not in range_analysis and rc_name in output_names:
            virtual_range_names.add(rc_name)

    if need_materialize:
        virtual_range_names = set()

    return virtual_range_names, need_materialize


@register_op_adapter(LiteFrameProjection)
class LiteFrameProjectionAdapter(SPEOperatorAdapter):
    @staticmethod
    def _run_projection(
        df: pd.DataFrame,
        projections: list,
        virtual_range_names: set,
        need_materialize: bool,
        in_range_columns: dict = None,
    ) -> pd.DataFrame:
        work_df = df

        # When a non-affine range transform is present, consolidate all
        # virtual range columns into physical Int64 columns so every
        # projection expression can reference them.
        if need_materialize and in_range_columns:
            work_df = _materialize_range_columns(work_df, in_range_columns)
            virtual_range_names = set()

        # First pass: run every UDF projection and collect its output column.
        # A UDF may produce multiple output columns shared by several
        # projections (transform_chunk / transform_row), so cache results by
        # UDF expr identity to avoid re-running the same function.
        udf_outputs = {}  # proj_name -> pd.Series
        udf_results = {}  # id(udf_expr) -> pd.DataFrame
        input_cols = set(work_df.columns)
        for proj in projections:
            proj_name = _projection_name(proj)
            if proj_name in virtual_range_names:
                continue
            udf_expr = _get_udf_expr(proj)
            if udf_expr is None:
                continue
            udf_id = id(udf_expr)
            if udf_id not in udf_results:
                udf_results[udf_id] = _run_udf(udf_expr, work_df)
            udf_df = udf_results[udf_id]
            # Extract the specific output column for this projection.
            out_name = proj_name
            if out_name in udf_df.columns:
                udf_outputs[proj_name] = udf_df[out_name]
            else:
                udf_outputs[proj_name] = udf_df.iloc[:, 0]
            # Augment work_df with non-input-colliding UDF outputs so
            # subsequent UDFs / expressions can reference intermediates.
            if proj_name not in input_cols:
                work_df = work_df.assign(**{proj_name: udf_outputs[proj_name]})

        # Second pass: assemble output columns in projection order.
        evaluator = LiteFrameExprEvaluator(work_df)
        result_cols = {}

        for proj in projections:
            proj_name = _projection_name(proj)
            if proj_name in virtual_range_names:
                # Stays virtual — not part of the physical output frame.
                continue
            if proj_name in udf_outputs:
                result_cols[proj_name] = udf_outputs[proj_name]
                continue
            if isinstance(proj, NamedExpr):
                val = evaluator.eval(proj.expr)
                result_cols[proj.name] = val
            elif isinstance(proj, LiteFrameColumn):
                result_cols[proj.name] = work_df[proj.name]
            elif isinstance(proj, LiteFrameMultiColumnExpr):
                result_df = evaluator.eval(proj)
                for name in proj.output_names:
                    result_cols[name] = result_df[name]
            elif isinstance(proj, LiteFrameExpr):
                name = getattr(proj, "name", None)
                if name is None:
                    raise ValueError(
                        f"Cannot extract name from projection: {type(proj)}"
                    )
                result_cols[name] = evaluator.eval(proj)
            else:
                raise ValueError(f"Unsupported projection type: {type(proj)}")

        return pd.DataFrame(result_cols)

    def generate_code(self, op, context: SPECodeContext) -> List[str]:
        cls_name = type(self).__name__
        context.register_import(__name__, from_item=cls_name)
        input_var = context.get_input_tileable_variable(op.inputs[0])
        res_var = context.get_output_tileable_variable(op.outputs[0])

        in_range_columns = _extract_input_range_columns(op.inputs[0])
        virtual_range_names, need_materialize = _resolve_range_state(
            op, in_range_columns
        )

        proj_args = self._translate_call_args(context, op.projections)
        virtual_var = context.register_operator_constants(virtual_range_names)
        range_cols_var = context.register_operator_constants(in_range_columns)
        args = proj_args + [virtual_var, repr(need_materialize), range_cols_var]
        return [
            f"{res_var} = {cls_name}._run_projection({input_var}, {', '.join(args)})"
        ]
