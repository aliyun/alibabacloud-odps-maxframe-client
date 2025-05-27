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

from ....tensor.linalg import (
    TensorInv,
    TensorLU,
    TensorMatmul,
    TensorQR,
    TensorSolveTriangular,
    TensorSVD,
    TensorTensorDot,
)
from ..core import SPECodeContext, SPEOperatorAdapter, register_op_adapter
from ..utils import build_method_call_adapter

TensorInvAdapter = build_method_call_adapter(
    TensorInv, "inv", "input", source_module="npl"
)
TensorLUAdapter = build_method_call_adapter(
    TensorLU, "lu", "input", source_module="spl"
)
TensorQRAdapter = build_method_call_adapter(TensorQR, "qr", 0, source_module="spl")
TensorSVDAdapter = build_method_call_adapter(TensorSVD, "svd", 0, source_module="spl")
TensorMatmulAdapter = build_method_call_adapter(
    TensorMatmul, "matmul", "a", "b", source_module="np", kw_keys=["casting", "order"]
)
TensorSolveTriangularAdapter = build_method_call_adapter(
    TensorSolveTriangular,
    "solve_triangular",
    "a",
    "b",
    source_module="spl",
    kw_keys=["lower", "strict"],
)


@register_op_adapter(TensorTensorDot)
class TensorTensorDotAdapter(SPEOperatorAdapter):
    def generate_code(self, op: TensorTensorDot, context: SPECodeContext) -> List[str]:
        context.register_import("numpy.linalg", "npl")

        lhs_var_name = context.get_input_tileable_variable(op.inputs[0])
        rhs_var_name = context.get_input_tileable_variable(op.inputs[1])
        res_var_name = context.get_output_tileable_variable(op.outputs[0])
        axes_str = self.generate_call_args_with_attributes(
            op, context, "a_axes", "b_axes"
        )
        return [
            f"{res_var_name} = npl.tensordot({lhs_var_name}, {rhs_var_name}, ({axes_str}))"
        ]
