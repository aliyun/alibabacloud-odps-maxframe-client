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

from typing import Dict, List, Type

from ....core import ENTITY_TYPE
from ....dataframe.arithmetic import (
    DataFrameAnd,
    DataFrameDot,
    DataFrameOr,
    DataFrameRound,
    DataFrameXor,
)
from ....dataframe.arithmetic.core import DataFrameBinOp, DataFrameUnaryOp
from ....dataframe.core import SERIES_TYPE
from ..core import SPECodeContext, SPEOperatorAdapter, register_op_adapter
from ..utils import build_method_call_adapter


@register_op_adapter(DataFrameUnaryOp)
class DataFrameUnaryFuncAdapter(SPEOperatorAdapter):
    def generate_code(self, op: DataFrameUnaryOp, context: SPECodeContext) -> List[str]:
        context.register_import("numpy", "np")

        inp_var = context.get_input_tileable_variable(op.inputs[0])
        res_var_name = context.get_output_tileable_variable(op.outputs[0])
        func = getattr(op, "_func_name")
        if isinstance(op, DataFrameRound):
            decimals_var = self.translate_var(context, op.decimals)
            return [f"{res_var_name} = {inp_var}.round({decimals_var})"]
        else:
            return [f"{res_var_name} = np.{func}({inp_var})"]


@register_op_adapter(DataFrameBinOp)
class DataFrameBinOpAdapter(SPEOperatorAdapter):
    def generate_code(self, op: DataFrameBinOp, context: SPECodeContext) -> List[str]:
        use_reversed = not isinstance(op.lhs, ENTITY_TYPE)
        kw = {}
        for arg in ("axis", "level", "fill_value"):
            if getattr(op, arg, None) is not None:
                kw[arg] = getattr(op, arg)
        if isinstance(op.inputs[0], SERIES_TYPE) and "axis" in kw:
            kw.pop("axis")

        if use_reversed:
            func = getattr(op, "_rfunc_name")
            inst, arg = op.rhs, op.lhs
        else:
            func = getattr(op, "_func_name")
            inst, arg = op.lhs, op.rhs

        inst_var_name = context.get_input_tileable_variable(inst)
        args_list = self._translate_call_args(context, arg, **kw)
        args = ", ".join(args_list)
        res_var_name = context.get_output_tileable_variable(op.outputs[0])
        return [f"{res_var_name} = {inst_var_name}.{func}({args})"]


@register_op_adapter([DataFrameAnd, DataFrameOr, DataFrameXor])
class DataFrameBitwiseBinOpAdapter(SPEOperatorAdapter):
    _python_op: Dict[Type[DataFrameBinOp], str] = {
        DataFrameAnd: "&",
        DataFrameOr: "|",
        DataFrameXor: "^",
    }

    def generate_code(self, op: DataFrameBinOp, context: SPECodeContext) -> List[str]:
        lhs, rhs = (
            (op.lhs, op.rhs) if isinstance(op.lhs, ENTITY_TYPE) else (op.rhs, op.lhs)
        )
        left_var = context.get_input_tileable_variable(lhs)
        res_var = context.get_output_tileable_variable(op.outputs[0])
        right_var = self._translate_call_args(context, rhs)[0]
        return [f"{res_var} = {left_var} {self._python_op[type(op)]} {right_var}"]


DataFrameDotAdapter = build_method_call_adapter(DataFrameDot, "dot", "rhs")
