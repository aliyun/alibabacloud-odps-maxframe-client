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
import pyarrow as pa

from maxframe import opcodes
from maxframe.liteframe.core import FrameMetadata
from maxframe.liteframe.expressions import LiteFrameColumn, LiteFrameExpr, NamedExpr
from maxframe.liteframe.operators.core import LiteFrameOperator, LiteFrameOperatorMixin
from maxframe.serialization.serializables import BoolField, ListField
from maxframe.utils import wrap_arrow_dtype


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
        return proj.expr.dtype
    if isinstance(proj, LiteFrameExpr):
        return proj.dtype
    raise ValueError(f"Cannot extract dtype from projection: {type(proj)}")


def _append_hidden_projections(lf, projections):
    """Append hidden column pass-through expressions to projections.

    Use this when building a LiteFrameProjection that should preserve hidden columns.
    """
    if lf._hidden_columns:
        for hc in sorted(lf._hidden_columns):
            projections.append(LiteFrameColumn(name=hc, dtype=lf._physical_dtypes[hc]))


class LiteFrameProjection(LiteFrameOperatorMixin, LiteFrameOperator):
    _op_type_ = opcodes.LITEFRAME_PROJECTION

    projections = ListField("projections")
    # When True, the executor materializes all virtual range columns into
    # physical Int64 columns before executing the projection expressions.
    # This is required whenever a projection expression references a range
    # column (e.g. cast, isna, arithmetic) — the generated pl.col(RANGE_COL_NAME)
    # needs a physical column to reference.
    # When False, range columns are pass-through only: they remain virtual in
    # the output and the executor skips their projections.
    materialize_range_cols = BoolField("materialize_range_cols", default=False)

    def __call__(self, liteframe):
        input_metadata = liteframe.frame_metadata

        # Derive output dtypes from projections (single source of truth)
        out_dtypes = pd.Series(
            {_projection_name(p): _projection_dtype(p) for p in self.projections}
        )

        out_range_cols = None
        materialized_dtypes = {}

        if (
            input_metadata is not None
            and input_metadata.range_columns
            and not self.materialize_range_cols
        ):
            out_col_names = list(out_dtypes.index)

            for rc_name, rc_info in input_metadata.range_columns.items():
                if rc_name in out_col_names:
                    if out_col_names.index(rc_name) == 0:
                        out_range_cols = out_range_cols or {}
                        out_range_cols[rc_name] = rc_info
                    else:
                        materialized_dtypes[rc_name] = wrap_arrow_dtype(pa.int64())

            # Range columns that remain virtual must be excluded from
            # _physical_dtypes (they are accounted for separately via
            # frame_metadata.range_columns and n_range in shape).
            if out_range_cols or materialized_dtypes:
                out_dtypes = pd.Series(
                    {
                        col: materialized_dtypes.get(col, out_dtypes[col])
                        for col in out_col_names
                        if col not in (out_range_cols or {})
                    },
                )

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
