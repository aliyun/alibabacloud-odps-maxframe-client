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

from typing import List

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
from maxframe.liteframe.expressions import LiteFrameExpr
from maxframe.liteframe.operators.filter import LiteFrameFilter


@register_op_adapter(LiteFrameFilter)
class LiteFrameFilterAdapter(SPEOperatorAdapter):
    @staticmethod
    def _run_filter(
        df: pd.DataFrame,
        predicate,
        mask_df: pd.DataFrame = None,
        range_columns: dict = None,
    ) -> pd.DataFrame:
        work_df = df

        # Always materialize range columns before filtering
        work_df = _materialize_range_columns(work_df, range_columns)

        if isinstance(predicate, LiteFrameExpr):
            evaluator = LiteFrameExprEvaluator(work_df)
            mask = evaluator.eval(predicate)
            return work_df[mask]
        else:
            if mask_df is None:
                raise ValueError(
                    "mask_df must be provided when predicate is not a LiteFrameExpr"
                )
            mask_col = mask_df.columns[0]
            return work_df[mask_df[mask_col]]

    def generate_code(self, op, context: SPECodeContext) -> List[str]:
        cls_name = type(self).__name__
        context.register_import(__name__, from_item=cls_name)
        input_var = context.get_input_tileable_variable(op.inputs[0])
        res_var = context.get_output_tileable_variable(op.outputs[0])

        # Extract range_columns from input tileable's frame_metadata
        range_columns = None
        in_tileable = op.inputs[0]
        if hasattr(in_tileable, "frame_metadata") and in_tileable.frame_metadata:
            if in_tileable.frame_metadata.range_columns:
                range_columns = {}
                for name, info in in_tileable.frame_metadata.range_columns.items():
                    range_columns[name] = (info.start, info.stop, info.step)

        args = self._translate_call_args(
            context,
            op.predicate,
            range_columns=range_columns,
        )

        if not isinstance(op.predicate, LiteFrameExpr):
            mask_var = context.get_input_tileable_variable(op.inputs[1])
            return [
                f"{res_var} = {cls_name}._run_filter({input_var}, {', '.join(args)}, "
                f"mask_df={mask_var})"
            ]
        else:
            return [
                f"{res_var} = {cls_name}._run_filter({input_var}, {', '.join(args)})"
            ]
