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

from .....learn.metrics._check_targets import CheckTargets
from .....learn.metrics._classification import (
    AccuracyScore,
    MultiLabelConfusionMatrix,
    PrecisionRecallFScoreSupport,
)
from ...core import SPECodeContext, SPEOperatorAdapter, register_op_adapter


@register_op_adapter(CheckTargets)
class CheckTargetsAdapter(SPEOperatorAdapter):
    def generate_code(self, op: CheckTargets, context: SPECodeContext) -> List[str]:
        context.register_import(
            "sklearn.metrics._classification",
            from_item="_check_targets",
            alias="sk_check_targets",
        )
        call_args = self.generate_call_args_with_attributes(
            op,
            context,
            "y_true",
            "y_pred",
            skip_none=True,
            kw_keys=["normalize", "sample_weight"],
        )
        out_vars = [context.get_output_tileable_variable(out) for out in op.outputs]
        out_vars_str = ", ".join(out_vars)
        return [f"{out_vars_str} = sk_check_targets({call_args})"]


@register_op_adapter(AccuracyScore)
class AccuracyScoreAdapter(SPEOperatorAdapter):
    def generate_code(self, op: AccuracyScore, context: SPECodeContext) -> List[str]:
        context.register_import(
            "sklearn.metrics", from_item="accuracy_score", alias="sk_accuracy_score"
        )

        call_args = self.generate_call_args_with_attributes(
            op,
            context,
            "y_true",
            "y_pred",
            skip_none=True,
            kw_keys=["normalize", "sample_weight"],
        )
        res_var_name = context.get_output_tileable_variable(op.outputs[0])
        return [f"{res_var_name} = sk_accuracy_score({call_args})"]


@register_op_adapter(MultiLabelConfusionMatrix)
class MultiLabelConfusionMatrixAdapter(SPEOperatorAdapter):
    def generate_code(
        self, op: MultiLabelConfusionMatrix, context: SPECodeContext
    ) -> List[str]:
        context.register_import(
            "sklearn.metrics",
            from_item="multilabel_confusion_matrix",
            alias="sk_multilabel_confusion_matrix",
        )

        call_args = self.generate_call_args_with_attributes(
            op,
            context,
            "y_true",
            "y_pred",
            skip_none=True,
            kw_keys=["sample_weight", "labels", "samplewise"],
        )
        res_var_name = context.get_output_tileable_variable(op.outputs[0])
        return [f"{res_var_name} = sk_multilabel_confusion_matrix({call_args})"]


@register_op_adapter(PrecisionRecallFScoreSupport)
class PrecisionRecallFScoreSupportAdapter(SPEOperatorAdapter):
    def generate_code(
        self, op: PrecisionRecallFScoreSupport, context: SPECodeContext
    ) -> List[str]:
        context.register_import(
            "sklearn.metrics",
            from_item="precision_recall_fscore_support",
            alias="sk_precision_recall_fscore_support",
        )

        call_args = self.generate_call_args_with_attributes(
            op,
            context,
            "y_true",
            "y_pred",
            skip_none=True,
            kw_keys=[
                "beta",
                "labels",
                "pos_label",
                "average",
                "warn_for",
                "sample_weight",
                "zero_division",
            ],
        )
        out_vars = [context.get_output_tileable_variable(out) for out in op.outputs]
        if op.output_limit < 4:
            out_vars.append("_na")
        out_vars_str = ", ".join(out_vars)
        return [f"{out_vars_str} = sk_precision_recall_fscore_support({call_args})"]
