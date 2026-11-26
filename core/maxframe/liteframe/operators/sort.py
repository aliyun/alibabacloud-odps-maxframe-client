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

from maxframe import opcodes
from maxframe.liteframe.core import FrameMetadata
from maxframe.liteframe.operators.core import LiteFrameOperator, LiteFrameOperatorMixin
from maxframe.serialization.serializables import (
    AnyField,
    BoolField,
    FieldTypes,
    Int64Field,
    ListField,
    StringField,
)
from maxframe.utils import wrap_arrow_dtype

_VALID_KEEP_KINDS = {"head", "first", "last", "all"}


class LiteFrameSort(LiteFrameOperatorMixin, LiteFrameOperator):
    _op_type_ = opcodes.SORT_VALUES

    by = ListField("by", default=None)
    ascending = AnyField("ascending", default=True)
    kind = StringField("kind", default="quicksort")
    na_position = StringField("na_position", default="last")
    parallel_kind = StringField("parallel_kind", default="PSRS")
    psrs_kinds = ListField("psrs_kinds", FieldTypes.string, default=None)
    nrows = Int64Field("nrows", default=None)
    keep_kind = StringField("keep_kind", default="head")

    def __call__(self, liteframe):
        if self.keep_kind not in _VALID_KEEP_KINDS:
            raise ValueError(
                f"Invalid keep_kind '{self.keep_kind}', "
                f"must be one of {sorted(_VALID_KEEP_KINDS)}"
            )
        in_dtypes = liteframe._data._physical_dtypes
        in_metadata = liteframe.frame_metadata

        # If input has virtual range_columns, materialize them into dtypes.
        # Range columns go first to match the tiler's materialization order
        # (range columns first, then physical columns).
        if in_metadata is not None and in_metadata.range_columns:
            rc_dtypes = {
                rc_name: wrap_arrow_dtype(pa.int64())
                for rc_name in in_metadata.range_columns
            }
            new_dtypes = pd.concat([pd.Series(rc_dtypes), in_dtypes])
        else:
            new_dtypes = in_dtypes.copy()

        # Preserve hidden columns
        hidden_columns = None
        if in_metadata is not None and in_metadata.hidden_columns:
            hidden_columns = in_metadata.hidden_columns

        # After sort, no virtual range_columns (row order changed)
        new_metadata = FrameMetadata(
            range_columns={},
            hidden_columns=hidden_columns,
        )
        n_rows = liteframe.shape[0]
        # Apply nrows limit if set by head pushdown optimization.
        # For keep_kind='all', the output may exceed nrows (all tied rows
        # are included), so the exact row count is unknown until execution.
        # For multi-column keep='first'/'last', pandas SelectNFrame can
        # return more or fewer than nrows depending on composite key ties,
        # so the row count is also unknown. Use np.nan (the project's
        # unknown-size sentinel) to avoid publishing incorrect shape
        # metadata to downstream tiling.
        if self.nrows is not None and (
            self.keep_kind == "all"
            or (self.keep_kind in ("first", "last") and len(self.by) > 1)
        ):
            n_rows = np.nan
        elif (
            self.nrows is not None
            and n_rows is not None
            and not (isinstance(n_rows, float) and n_rows != n_rows)
        ):
            n_rows = min(int(n_rows), self.nrows)
        shape = (n_rows, len(new_dtypes))

        return self.new_liteframe(
            [liteframe],
            shape=shape,
            physical_dtypes=new_dtypes,
            frame_metadata=new_metadata,
        )


class LiteFrameRank(LiteFrameOperatorMixin, LiteFrameOperator):
    _op_type_ = opcodes.RANK

    method = StringField("method", default="average")
    numeric_only = BoolField("numeric_only", default=False)
    na_option = StringField("na_option", default="keep")
    ascending = AnyField("ascending", default=True)
    pct = BoolField("pct", default=False)
    hidden_columns = ListField("hidden_columns", FieldTypes.string, default=None)

    def _inherit_alignment_id(self, inputs):
        # Rank preserves the input row layout (it replaces values in place
        # and does not reorder rows); its PSRS tiling always ends with a
        # place_back step restoring the source chunk layout, so the output
        # stays row-aligned with its input.
        return self._inherit_single_input_alignment_id(inputs)

    def __call__(self, liteframe):
        in_dtypes = liteframe._data._physical_dtypes
        in_metadata = liteframe.frame_metadata

        # Target columns are resolved in the API (honoring ``numeric_only``)
        # and carried in extra_params; default to all physical columns.
        # Use ``in`` check (not ``or``) so an explicit empty list (all-string
        # frame with numeric_only=True) is preserved — ``[] or fallback``
        # would wrongly restore every physical column as a rank target.
        if "by" in self.extra_params:
            by = self.extra_params["by"]
        else:
            by = list(in_dtypes.index)

        # Rank preserves the input row layout (it replaces values in place,
        # it does not reorder rows). Virtual range columns are therefore
        # preserved unless they are rank targets (materialized by the tiler).
        #
        # Hidden columns: the rank value itself does not carry hidden columns
        # through the multi-chunk PSRS path. The PSRS pipeline (map →
        # local_rank → place_back → projection) reconstructs frames containing
        # only routing columns and rank values. Hidden columns remain in the
        # source frame; they will be attached back to the rank result once
        # place_back is enhanced to reconstruct the full original frame (not
        # just routing + rank values). The single-chunk path does pass hidden
        # columns through (see LiteFrameRankExecutor). The hidden_columns
        # field is declared in metadata so the physical schema matches, but
        # the PSRS executor does not yet materialize them in multi-chunk.
        rank_dtype = wrap_arrow_dtype(pa.float64())
        new_dtypes = pd.Series([rank_dtype] * len(by), index=list(by), dtype=object)

        hidden_columns = None
        if in_metadata is not None and in_metadata.hidden_columns:
            hidden_columns = in_metadata.hidden_columns
            self.hidden_columns = hidden_columns
            hidden_dtypes = in_dtypes[
                [c for c in hidden_columns if c in in_dtypes.index]
            ]
            new_dtypes = pd.concat([new_dtypes, hidden_dtypes])

        # Range columns that are not rank targets stay virtual; targets are
        # materialized by the tiler. Standalone rank preserves row order, so
        # non-target range columns remain valid.
        input_range_columns = (
            in_metadata.range_columns
            if in_metadata and in_metadata.range_columns
            else {}
        )
        new_range_columns = {
            k: v for k, v in input_range_columns.items() if k not in by
        }
        new_metadata = FrameMetadata(
            range_columns=new_range_columns,
            hidden_columns=hidden_columns,
        )
        n_rows = liteframe.shape[0]
        shape = (n_rows, len(new_dtypes) + len(new_range_columns))
        return self.new_liteframe(
            [liteframe],
            shape=shape,
            physical_dtypes=new_dtypes,
            frame_metadata=new_metadata,
        )
