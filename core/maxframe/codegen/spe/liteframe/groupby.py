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
from maxframe.liteframe.operators.groupby import LiteFrameGroupByOp


@register_op_adapter(LiteFrameGroupByOp)
class LiteFrameGroupByOpAdapter(SPEOperatorAdapter):
    """SPE adapter for LiteFrameGroupByOp.

    In single-process SPE execution there is no shuffle: the operator is a
    pass-through that returns the input frame unchanged.  The downstream
    LiteFrameAgg operator (stage=agg) performs the actual groupby+aggregation.
    """

    @staticmethod
    def _run_groupby(df: pd.DataFrame) -> pd.DataFrame:
        return df

    def generate_code(self, op, context: SPECodeContext) -> List[str]:
        cls_name = type(self).__name__
        context.register_import(__name__, from_item=cls_name)
        input_var = context.get_input_tileable_variable(op.inputs[0])
        res_var = context.get_output_tileable_variable(op.outputs[0])
        return [f"{res_var} = {cls_name}._run_groupby({input_var})"]
