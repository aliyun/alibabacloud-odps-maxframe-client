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

from .....learn.utils.multiclass import IsMultilabel, TypeOfTarget, UniqueLabels
from ...core import SPECodeContext, SPEOperatorAdapter, register_op_adapter


@register_op_adapter(UniqueLabels)
class UniqueLabelsAdapter(SPEOperatorAdapter):
    def generate_code(self, op: UniqueLabels, context: SPECodeContext) -> List[str]:
        context.register_import(
            "sklearn.utils.multiclass",
            from_item="unique_labels",
            alias="sk_unique_labels",
        )

        call_args = [self.translate_var(context, v) for v in op.ys]
        res_var_name = context.get_output_tileable_variable(op.outputs[0])
        return [f"{res_var_name} = sk_unique_labels({', '.join(call_args)})"]


@register_op_adapter(IsMultilabel)
class IsMultilabelAdapter(SPEOperatorAdapter):
    def generate_code(self, op: IsMultilabel, context: SPECodeContext) -> List[str]:
        context.register_import(
            "sklearn.utils.multiclass",
            from_item="is_multilabel",
            alias="sk_is_multilabel",
        )

        call_args = self.generate_call_args_with_attributes(op, context, "y")
        res_var_name = context.get_output_tileable_variable(op.outputs[0])
        return [f"{res_var_name} = sk_is_multilabel({call_args})"]


@register_op_adapter(TypeOfTarget)
class TypeOfTargetAdapter(SPEOperatorAdapter):
    def generate_code(self, op: TypeOfTarget, context: SPECodeContext) -> List[str]:
        context.register_import(
            "sklearn.utils.multiclass",
            from_item="type_of_target",
            alias="sk_type_of_target",
        )

        call_args = self.generate_call_args_with_attributes(op, context, "y")
        res_var_name = context.get_output_tileable_variable(op.outputs[0])
        return [f"{res_var_name} = sk_type_of_target({call_args})"]
