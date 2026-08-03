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

from collections import OrderedDict

import pandas as pd
import pyarrow as pa

from maxframe import opcodes
from maxframe.core import OutputType
from maxframe.core.operator import OperatorStage
from maxframe.liteframe.core import FrameMetadata
from maxframe.liteframe.expressions import LiteFrameColumn
from maxframe.liteframe.operators.core import LiteFrameOperator, LiteFrameOperatorMixin
from maxframe.liteframe.operators.project import (
    LiteFrameProjection,
    _append_hidden_projections,
)
from maxframe.serialization.serializables import (
    AnyField,
    BoolField,
    DictField,
    Int32Field,
    Int64Field,
    ListField,
    ReferenceField,
    StringField,
)
from maxframe.utils import wrap_arrow_dtype


class LiteFrameAgg(LiteFrameOperatorMixin, LiteFrameOperator):
    _op_type_ = opcodes.AGGREGATE  # reuse opcode 714
    _output_type_ = OutputType.liteframe

    # Stage for tiling (None = tileable-level entry)
    stage = ReferenceField("stage", OperatorStage, default=None)

    # --- User-facing function spec ---
    raw_func = AnyField("raw_func")
    raw_func_kw = DictField("raw_func_kw", default=None)

    # --- Grouping ---
    groupby_params = DictField("groupby_params", default=None)
    # None = whole-frame agg
    # Dict with keys: by (str or list), sort (bool), dropna (bool)

    # --- Compiled agg functions ---
    # Each entry is a dict: {
    #   "raw_func_name": str,         # e.g. "sum", "mean"
    #   "map_func_name": str,         # e.g. "sum", "count", "sum_sq"
    #   "agg_func_name": str,         # e.g. "sum" (count->sum in combine)
    #   "cols": list[str] or None,    # None=all columns
    #   "output_key": str,            # unique key for this agg output
    #   "kwds": dict,
    #   "pre_agg_col": str,           # optional: projected column for power-sum
    # }
    agg_funcs = ListField("agg_funcs", default=None)

    # --- Config ---
    combine_size = Int32Field("combine_size", default=None)
    method = StringField("method", default="auto")
    numeric_only = BoolField("numeric_only", default=None)
    use_inf_as_na = BoolField("use_inf_as_na", default=False)
    chunk_store_limit = Int64Field("chunk_store_limit", default=None)
    size_recorder_name = StringField("size_recorder_name", default=None)

    # --- Pre/post projection info (set by __call__, used by tiler) ---
    # List of column names produced by agg, before post-projection rename
    agg_output_columns = ListField("agg_output_columns", default=None)
    # Final output column names (after suffix/named-agg rename)
    output_column_names = ListField("output_column_names", default=None)
    # dtypes for each agg output column (before post-projection)
    agg_output_dtypes = AnyField("agg_output_dtypes", default=None)
    # Pre-agg projection exprs for power-sum primitives (sum_sq, sum_cube, etc.)
    pre_agg_projection_exprs = ListField("pre_agg_projection_exprs", default=None)

    def __call__(self, liteframe):
        from maxframe.liteframe.reduction.compile import compile_agg

        # Compile agg: normalize func, build agg_funcs, compute output info
        compiled = compile_agg(self, liteframe)

        self.agg_funcs = compiled.agg_funcs
        self.agg_output_columns = compiled.agg_output_columns
        self.agg_output_dtypes = compiled.agg_output_dtypes
        self.pre_agg_projection_exprs = compiled.pre_agg_projection_exprs

        # If pre-agg projection is needed, insert a LiteFrameProjection
        # before the aggregation to compute power-sum input columns.
        if compiled.pre_agg_projection_exprs:
            # Build pass-through projections for all original columns
            # plus the pre-agg power-sum projections
            pass_through = []
            for col_name in liteframe._physical_dtypes.index:
                col_dtype = liteframe._physical_dtypes[col_name]
                pass_through.append(
                    LiteFrameColumn(name=col_name, dtype=col_dtype).rename(col_name)
                )
            all_pre_agg_projections = pass_through + compiled.pre_agg_projection_exprs
            _append_hidden_projections(liteframe, all_pre_agg_projections)

            pre_proj_op = LiteFrameProjection(projections=all_pre_agg_projections)
            liteframe = pre_proj_op(liteframe)

        if compiled.projection_exprs:
            # Recipe-based decomposition: agg produces intermediates,
            # then a LiteFrameProjection reconstructs final values.
            # The agg output_column_names are the intermediate names
            # (no renaming needed at agg level).
            self.output_column_names = None

            # Build intermediate dtypes for the agg output
            inter_dtypes_dict = OrderedDict()
            groupby_params = self.groupby_params
            if groupby_params:
                by = groupby_params["by"]
                group_keys = [by] if isinstance(by, str) else list(by)
                for gk in group_keys:
                    inter_dtypes_dict[gk] = liteframe.dtypes.get(
                        gk, wrap_arrow_dtype(pa.string())
                    )
            for desc in compiled.agg_funcs:
                col = desc["cols"][0]
                prim = desc["map_func_name"]
                inter_col_name = f"{desc['output_key']}__{col}_{prim}"
                if prim == "count":
                    inter_dtypes_dict[inter_col_name] = wrap_arrow_dtype(pa.int64())
                elif prim in ("sum_sq", "sum_cube", "sum_fourth"):
                    # Power-sum intermediates: the pre-agg projection already
                    # produced float64 columns, so sum produces float64.
                    inter_dtypes_dict[inter_col_name] = wrap_arrow_dtype(pa.float64())
                else:
                    # Preserve input column dtype for sum, min, max, prod, etc.
                    # to avoid precision loss (e.g. int64 > 2^53 loses
                    # precision in float64).
                    inter_dtypes_dict[inter_col_name] = liteframe.dtypes.get(
                        col, wrap_arrow_dtype(pa.float64())
                    )

            inter_dtypes = pd.Series(inter_dtypes_dict)
            inter_shape = (compiled.out_shape[0], len(inter_dtypes))

            agg_result = self.new_liteframe(
                [liteframe],
                shape=inter_shape,
                physical_dtypes=inter_dtypes,
                frame_metadata=FrameMetadata(),
            )

            # Chain post-agg projection
            proj_op = LiteFrameProjection(projections=compiled.projection_exprs)
            return proj_op(agg_result)
        else:
            # All directly decomposable or passthrough: no projection needed
            self.output_column_names = compiled.output_column_names
            return self.new_liteframe(
                [liteframe],
                shape=compiled.out_shape,
                physical_dtypes=compiled.out_dtypes,
                frame_metadata=FrameMetadata(),
            )
