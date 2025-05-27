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

from ....tensor.arithmetic.angle import TensorAngle
from ....tensor.arithmetic.around import TensorAround
from ....tensor.arithmetic.core import TensorBinOp, TensorUnaryOp
from ....tensor.arithmetic.isclose import TensorIsclose
from ....tensor.arithmetic.setimag import TensorSetImag
from ....tensor.arithmetic.setreal import TensorSetReal
from ..core import SPECodeContext, SPEOperatorAdapter, register_op_adapter
from .core import TensorUnaryCommonAdapter


@register_op_adapter(TensorUnaryOp)
class TensorUnaryOpAdapter(TensorUnaryCommonAdapter):
    def _get_op_kw_args(self, op: TensorUnaryOp) -> List[str]:
        return ["casting", "order", "where"]


@register_op_adapter(TensorAngle)
class TensorAngleAdapter(TensorUnaryOpAdapter):
    def _get_op_kw_args(self, op: TensorAngle) -> List[str]:
        return ["deg"]


@register_op_adapter(TensorAround)
class TensorAroundAdapter(TensorUnaryOpAdapter):
    def _get_op_kw_args(self, op: TensorAround) -> List[str]:
        return ["decimals"]


@register_op_adapter(TensorBinOp)
class TensorBinOpAdapter(SPEOperatorAdapter):
    def generate_code(self, op: TensorBinOp, context: SPECodeContext) -> List[str]:
        context.register_import("numpy", "np")
        call_args = self.generate_call_args_with_attributes(
            op,
            context,
            "lhs",
            "rhs",
            skip_none=True,
            kw_keys=self._get_op_kw_args(op),
        )
        res_var_name = context.get_output_tileable_variable(op.outputs[0])
        return [f"{res_var_name} = np.{op._func_name}({call_args})"]

    def _get_op_kw_args(self, op: TensorBinOp) -> List[str]:
        return ["casting", "order", "where"]


@register_op_adapter(TensorIsclose)
class TensorIscloseAdapter(TensorBinOpAdapter):
    def _get_op_kw_args(self, op: TensorIsclose) -> List[str]:
        return ["equal_nan", "rtol", "atol"]


@register_op_adapter(TensorSetImag)
class TensorSetImagAdapter(SPEOperatorAdapter):
    def generate_code(self, op: TensorSetImag, context: SPECodeContext) -> List[str]:
        context.register_import("numpy", "np")

        source_var_name = context.get_input_tileable_variable(op.inputs[0])
        imag_var_name = context.get_input_tileable_variable(op.inputs[1])
        res_var_name = context.get_output_tileable_variable(op.outputs[0])
        return [
            f"{source_var_name}.imag = {imag_var_name}",
            f"{res_var_name} = {source_var_name}",
        ]


@register_op_adapter(TensorSetReal)
class TensorSetRealAdapter(SPEOperatorAdapter):
    def generate_code(self, op: TensorSetReal, context: SPECodeContext) -> List[str]:
        context.register_import("numpy", "np")

        source_var_name = context.get_input_tileable_variable(op.inputs[0])
        imag_var_name = context.get_input_tileable_variable(op.inputs[1])
        res_var_name = context.get_output_tileable_variable(op.outputs[0])
        return [
            f"{source_var_name}.real = {imag_var_name}",
            f"{res_var_name} = {source_var_name}",
        ]
