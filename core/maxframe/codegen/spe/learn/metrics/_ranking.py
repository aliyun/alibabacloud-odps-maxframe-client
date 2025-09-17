# Copyright 1999-2025 Alibaba Group Holding Ltd.
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

from .....learn.metrics._ranking import AucOp, RocAucScore, RocCurve
from ...core import SPECodeContext, SPEOperatorAdapter, register_op_adapter


@register_op_adapter(AucOp)
class AucOpAdapter(SPEOperatorAdapter):
    def generate_code(self, op: AucOp, context: SPECodeContext) -> List[str]:
        context.register_import(
            "sklearn.metrics",
            from_item="auc",
            alias="sk_auc",
        )
        call_args = self.generate_call_args_with_attributes(
            op, context, "x", "y", skip_none=True
        )
        out_vars = [context.get_output_tileable_variable(out) for out in op.outputs]
        out_vars_str = ", ".join(out_vars)
        return [f"{out_vars_str} = sk_auc({call_args})"]


@register_op_adapter(RocAucScore)
class RocAucScoreAdapter(SPEOperatorAdapter):
    def generate_code(self, op: RocAucScore, context: SPECodeContext) -> List[str]:
        context.register_import(
            "sklearn.metrics",
            from_item="roc_auc_score",
            alias="sk_roc_auc_score",
        )
        call_args = self.generate_call_args_with_attributes(
            op,
            context,
            "y_true",
            "y_score",
            skip_none=True,
            kw_keys=["average", "sample_weight", "max_fpr", "multi_class", "labels"],
        )
        out_vars = [context.get_output_tileable_variable(out) for out in op.outputs]
        out_vars_str = ", ".join(out_vars)
        return [f"{out_vars_str} = sk_roc_auc_score({call_args})"]


@register_op_adapter(RocCurve)
class RocCurveAdapter(SPEOperatorAdapter):
    def generate_code(self, op: RocCurve, context: SPECodeContext) -> List[str]:
        context.register_import(
            "sklearn.metrics",
            from_item="roc_curve",
            alias="sk_roc_curve",
        )
        call_args = self.generate_call_args_with_attributes(
            op,
            context,
            "y_true",
            "y_score",
            skip_none=True,
            kw_keys=["sample_weight", "drop_intermediate", "pos_label"],
        )
        out_vars = [context.get_output_tileable_variable(out) for out in op.outputs]
        out_vars_str = ", ".join(out_vars)
        return [f"{out_vars_str} = sk_roc_curve({call_args})"]
