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

from .....learn.preprocessing._label._label_binarizer import LabelBinarizeOp
from ...core import SPECodeContext, SPEOperatorAdapter, register_op_adapter


@register_op_adapter(LabelBinarizeOp)
class LabelBinarizeOpAdapter(SPEOperatorAdapter):
    def generate_code(self, op: LabelBinarizeOp, context: SPECodeContext) -> List[str]:
        context.register_import("numpy", "np")
        context.register_import(
            "sklearn.preprocessing",
            from_item="label_binarize",
            alias="sk_label_binarize",
        )
        if op.n_classes is not None:
            classes_arg = f"np.arange({op.n_classes})"
        else:
            classes_arg = self.translate_var(context, op.classes)
        call_args = self.generate_call_args_with_attributes(
            op,
            context,
            skip_none=True,
            kw_keys=["neg_label", "pos_label", "sparse_output"],
        )
        if call_args:
            call_args = ", " + call_args
        inp_var_name = context.get_input_tileable_variable(op.inputs[0])
        res_var_name = context.get_output_tileable_variable(op.outputs[0])
        return [
            f"{res_var_name} = sk_label_binarize({inp_var_name}, "
            f"classes={classes_arg}{call_args})"
        ]
