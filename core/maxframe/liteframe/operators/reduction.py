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
from maxframe.udf import BuiltinFunction, copy_func_scheduling_hints
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
    # Frame type for custom agg functions (pandas/polars)
    input_frame_type = StringField("input_frame_type", default="pandas")

    # --- Pre/post projection info (set by __call__, used by tiler) ---
    # List of column names produced by agg, before post-projection rename
    agg_output_columns = ListField("agg_output_columns", default=None)
    # Final output column names (after suffix/named-agg rename)
    output_column_names = ListField("output_column_names", default=None)
    # dtypes for each agg output column (before post-projection)
    agg_output_dtypes = AnyField("agg_output_dtypes", default=None)
    # Pre-agg projection exprs for power-sum primitives (sum_sq, sum_cube, etc.)
    pre_agg_projection_exprs = ListField("pre_agg_projection_exprs", default=None)

    def has_custom_code(self) -> bool:
        """Return True when a non-builtin custom agg function is present.

        ``@builtin_function``-decorated custom aggs return False so the
        operator does not trigger an isolation sandbox.
        """
        if not self.agg_funcs:
            return False
        for desc in self.agg_funcs:
            custom_func = desc.get("custom_func")
            if custom_func is not None and not isinstance(custom_func, BuiltinFunction):
                return True
        return False

    def __call__(self, liteframe):
        from maxframe.liteframe.reduction.compile import (
            _size_intermediate_name,
            compile_agg,
        )
        from maxframe.liteframe.reduction.custom_agg import _resolve_scheduling_source

        # Compile agg: normalize func, build agg_funcs, compute output info
        compiled = compile_agg(self, liteframe)

        self.agg_funcs = compiled.agg_funcs
        self.agg_output_columns = compiled.agg_output_columns
        self.agg_output_dtypes = compiled.agg_output_dtypes
        self.pre_agg_projection_exprs = compiled.pre_agg_projection_exprs
        # Propagate the agg hook's running options onto the operator node
        # (mirrors LiteFrameFunctional).  Resolution is client-local: the
        # scheduling source is derived from ``custom_func`` here at call time,
        # never persisted in the agg descriptor, so no raw callable or
        # MarkedFunction leaks into serialized operator state.
        # ``_resolve_scheduling_source`` returns the object
        # ``discover_marked_functions`` can traverse to find the
        # ``@with_running_options`` MarkedFunction for each
        # ``@builtin_function`` shape: the underlying MarkedFunction for a
        # standalone builtin callable, the raw class attr for a builtin
        # CustomAgg class, and the bound hook for a plain CustomAgg instance
        # / _SingleFuncAgg.
        for desc in self.agg_funcs or []:
            custom_func = desc.get("custom_func")
            if custom_func is not None:
                copy_func_scheduling_hints(
                    _resolve_scheduling_source(custom_func, "agg"), self
                )
                break  # one marked function per operator (validated inside)

        # Custom agg pre-projection: runs as transform_chunk before the agg
        if getattr(compiled, "custom_pre_projection", None):
            pre_proj_op = LiteFrameProjection(
                projections=compiled.custom_pre_projection
            )
            liteframe = pre_proj_op(liteframe)

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
                prim = desc["map_func_name"]
                if prim == "size":
                    # frame-level: one column regardless of target columns.
                    inter_col_name = _size_intermediate_name(f"{desc['output_key']}__")
                    inter_dtypes_dict[inter_col_name] = wrap_arrow_dtype(pa.int64())
                    continue
                col = desc["cols"][0]
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
            # All directly decomposable, passthrough, or custom agg.
            self.output_column_names = compiled.output_column_names
            has_post = bool(getattr(compiled, "custom_post_projection", None))

            # With a custom post, the agg node emits intermediate columns
            # (group keys + agg output), not the final post output, so the
            # post projection can define the final schema.
            if has_post:
                inter_dtypes_dict = OrderedDict()
                groupby_params = self.groupby_params
                if groupby_params:
                    by = groupby_params["by"]
                    group_keys = [by] if isinstance(by, str) else list(by)
                    for gk in group_keys:
                        inter_dtypes_dict[gk] = liteframe.dtypes.get(
                            gk, wrap_arrow_dtype(pa.string())
                        )
                for col, dt in compiled.agg_output_dtypes.items():
                    inter_dtypes_dict[col] = dt
                node_dtypes = pd.Series(inter_dtypes_dict)
                node_shape = (compiled.out_shape[0], len(node_dtypes))
            else:
                node_dtypes = compiled.out_dtypes
                node_shape = compiled.out_shape

            agg_result = self.new_liteframe(
                [liteframe],
                shape=node_shape,
                physical_dtypes=node_dtypes,
                frame_metadata=FrameMetadata(),
            )

            if has_post:
                return LiteFrameProjection(projections=compiled.custom_post_projection)(
                    agg_result
                )
            return agg_result
