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

import pandas as pd

from maxframe import opcodes
from maxframe.core import OutputType
from maxframe.liteframe.arithmetic.range_affine import (
    RangeAnalysisResult,
    analyze_range_affine,
    transform_range_info,
)
from maxframe.liteframe.core import FrameMetadata
from maxframe.liteframe.expressions import (
    LiteFrameBaseExpr,
    LiteFrameColumn,
    LiteFrameExpr,
    LiteFrameMultiColumnSubExpr,
    LiteFrameUDFExpr,
    NamedExpr,
)
from maxframe.liteframe.operators.core import LiteFrameOperator, LiteFrameOperatorMixin
from maxframe.serialization.serializables import DictField, FieldTypes, ListField


def _projection_name(proj):
    """Extract the output column name from a projection expression."""
    if isinstance(proj, NamedExpr):
        return proj.name
    if isinstance(proj, LiteFrameColumn):
        return proj.name
    # Fallback: any LiteFrameExpr with a .name attribute (e.g. renamed expressions)
    if isinstance(proj, LiteFrameExpr) and hasattr(proj, "name"):
        return proj.name
    raise ValueError(f"Cannot extract name from projection: {type(proj)}")


def _projection_dtype(proj):
    """Extract the output dtype from a projection expression."""
    if isinstance(proj, NamedExpr):
        # NamedExpr inherits dtype from LiteFrameExpr.  Two cases:
        #   1. UDF projections: NamedExpr.dtype is explicitly set to the
        #      inferred output dtype (the inner expr may be a
        #      LiteFrameMultiColumnSubExpr or a UDF expr without .dtype).
        #   2. Arithmetic projections: NamedExpr.dtype is None (not passed
        #      at construction), so we must fall through to expr.dtype.
        # After .bind(), NamedExpr.dtype always equals expr.dtype, so
        # the None check cleanly discriminates between the two cases.
        own_dtype = getattr(proj, "dtype", None)
        if own_dtype is not None:
            return own_dtype
        return proj.expr.dtype
    if isinstance(proj, LiteFrameExpr):
        return proj.dtype
    raise ValueError(f"Cannot extract dtype from projection: {type(proj)}")


def _append_hidden_projections(lf, projections):
    """Append hidden column pass-through expressions to projections.

    Use this when building a LiteFrameProjection that should preserve hidden columns.
    Hidden columns already referenced in projections (e.g. explicitly selected)
    are skipped to avoid duplicates.
    """
    if lf._hidden_columns:
        existing_names = {_projection_name(p) for p in projections}
        for hc in sorted(lf._hidden_columns):
            if hc not in existing_names:
                projections.append(
                    LiteFrameColumn(name=hc, dtype=lf._physical_dtypes[hc])
                )


def iter_projection_udf_funcs(projections):
    """Yield the user callable from each embedded UDF expression.

    Projection UDFs (map / transform_chunk / transform_row) are embedded as
    ``LiteFrameUDFExpr`` nodes inside the projection list, either directly
    wrapped in ``NamedExpr`` or via ``LiteFrameMultiColumnSubExpr``.  Yields
    each UDF's ``func`` so its ``@with_running_options`` / ``@with_resources``
    / ``@with_python_requirements`` decorators can be propagated.
    """
    for proj in projections:
        expr = proj.expr if isinstance(proj, NamedExpr) else proj
        if isinstance(expr, LiteFrameUDFExpr):
            yield expr.func
        elif isinstance(expr, LiteFrameMultiColumnSubExpr) and isinstance(
            expr.multi_expr, LiteFrameUDFExpr
        ):
            yield expr.multi_expr.func


class LiteFrameProjection(LiteFrameOperatorMixin, LiteFrameOperator):
    _op_type_ = opcodes.LITEFRAME_PROJECTION
    _output_type_ = OutputType.liteframe

    projections = ListField("projections")
    # Per-projection result of range-affine analysis. Keyed by projection
    # output name. None means no input range columns (no analysis needed).
    # When any entry has is_affine=False, references_range=True, the
    # executor materializes all range columns.
    range_analysis = DictField(
        "range_analysis",
        key_type=FieldTypes.string,
        value_type=FieldTypes.reference(RangeAnalysisResult),
        default=None,
    )

    def has_custom_code(self) -> bool:
        return any(
            p.has_custom_code
            for p in self.projections
            if isinstance(p, LiteFrameBaseExpr)
        )

    def __call__(self, liteframe):
        input_metadata = liteframe.frame_metadata

        # Bind any projections whose dtype is still None (unbound) against
        # the input schema, so expression dtypes resolve bottom-up. This is
        # idempotent for already-bound projections (arithmetic fused-projection
        # path, UDF path). It matters for the agg post-projection path
        # (compile._build_projection_exprs), which builds expressions with
        # dtype=None leaves referencing intermediate agg columns; without
        # binding the output dtype stays None and the ODPS round-trip loses
        # the correct type (e.g. mean=sum/count planned as int not float).
        in_dtypes = getattr(liteframe, "_physical_dtypes", None)
        if in_dtypes is None:
            in_dtypes = getattr(
                getattr(liteframe, "_data", None), "_physical_dtypes", None
            )
        if any(_projection_dtype(p) is None for p in self.projections):
            bound = [
                p.bind(in_dtypes) if getattr(p, "dtype", None) is None else p
                for p in self.projections
            ]
            self.projections = bound

        # Derive output dtypes from projections (single source of truth)
        out_dtypes = pd.Series(
            {_projection_name(p): _projection_dtype(p) for p in self.projections}
        )

        out_range_cols = None

        input_range_cols = (
            input_metadata.range_columns
            if input_metadata and input_metadata.range_columns
            else None
        )

        if input_range_cols:
            # Run analyzer on each projection
            self.range_analysis = {}
            for proj in self.projections:
                proj_name = _projection_name(proj)
                inner_expr = proj.expr if isinstance(proj, NamedExpr) else proj
                result = analyze_range_affine(inner_expr, input_range_cols)
                self.range_analysis[proj_name] = result

            # Check contiguous prefix: affine range columns must occupy
            # positions 0..k-1. If any range-referencing projection at a
            # non-prefix position is affine, or any is non-affine, mark all
            # for materialization.
            out_col_names = list(out_dtypes.index)
            need_materialize = False
            prefix_affine = True
            for i, name in enumerate(out_col_names):
                result = self.range_analysis[name]
                if result.references_range:
                    if not result.is_affine:
                        need_materialize = True
                        break
                    if not prefix_affine:
                        need_materialize = True
                        break
                else:
                    prefix_affine = False

            if need_materialize:
                for name in out_col_names:
                    if self.range_analysis[name].references_range:
                        self.range_analysis[name] = RangeAnalysisResult(
                            is_affine=False, references_range=True
                        )
            else:
                for name in out_col_names:
                    result = self.range_analysis[name]
                    if result.is_affine and result.references_range:
                        src_info = input_range_cols[result.col_name]
                        transformed = transform_range_info(
                            src_info, result.scale, result.offset
                        )
                        out_range_cols = out_range_cols or {}
                        out_range_cols[name] = transformed

                if out_range_cols:
                    out_dtypes = pd.Series(
                        {
                            col: out_dtypes[col]
                            for col in out_col_names
                            if col not in out_range_cols
                        },
                    )
        else:
            self.range_analysis = None

        # Hidden columns that pass through to the output remain hidden.
        remaining_hidden = (
            set(input_metadata.hidden_columns) & set(out_dtypes.index)
            if input_metadata and input_metadata.hidden_columns
            else set()
        )

        # Build output frame_metadata
        out_frame_metadata = None
        if out_range_cols or remaining_hidden:
            out_frame_metadata = FrameMetadata(
                range_columns=out_range_cols,
                hidden_columns=list(remaining_hidden) if remaining_hidden else None,
            )

        n_range = (
            len(out_frame_metadata.range_columns)
            if out_frame_metadata and out_frame_metadata.range_columns
            else 0
        )
        shape = (liteframe.shape[0], len(out_dtypes) + n_range)

        return self.new_liteframe(
            [liteframe],
            shape=shape,
            physical_dtypes=out_dtypes,
            frame_metadata=out_frame_metadata,
        )
