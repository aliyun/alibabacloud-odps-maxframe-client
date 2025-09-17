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

from typing import List, Union

from ....tensor.fft import TensorFFTFreq, TensorRFFTFreq
from ....tensor.fft.core import TensorFFTMixin, TensorFFTNMixin, TensorFFTShiftMixin
from ..core import SPECodeContext, SPEOperatorAdapter, register_op_adapter


@register_op_adapter(TensorFFTMixin)
class TensorFFTAdapter(SPEOperatorAdapter):
    def generate_code(self, op: TensorFFTMixin, context: SPECodeContext) -> List[str]:
        context.register_import("numpy", "np")
        func_name = type(op).__name__.lower()[6:]
        input_var_name = context.get_input_tileable_variable(op.inputs[0])
        res_var_name = context.get_output_tileable_variable(op.outputs[0])
        kw_str = self.generate_call_args_with_attributes(
            op, context, kw_keys=["n", "axis", "norm"], skip_none=True
        )
        return [f"{res_var_name} = np.fft.{func_name}({input_var_name}, {kw_str})"]


@register_op_adapter(TensorFFTNMixin)
class TensorFFTNAdapter(SPEOperatorAdapter):
    def generate_code(self, op: TensorFFTNMixin, context: SPECodeContext) -> List[str]:
        context.register_import("numpy", "np")
        func_name = type(op).__name__.lower()[6:]
        input_var_name = context.get_input_tileable_variable(op.inputs[0])
        res_var_name = context.get_output_tileable_variable(op.outputs[0])
        kw_str = self.generate_call_args_with_attributes(
            op, context, "shape", kw_keys=["axes", "norm"], skip_none=True
        )
        return [f"{res_var_name} = np.fft.{func_name}({input_var_name}, {kw_str})"]


@register_op_adapter(TensorFFTShiftMixin)
class TensorFFTShiftAdapter(SPEOperatorAdapter):
    def generate_code(
        self, op: TensorFFTShiftMixin, context: SPECodeContext
    ) -> List[str]:
        context.register_import("numpy", "np")
        func_name = type(op).__name__.lower()[6:]
        input_var_name = context.get_input_tileable_variable(op.inputs[0])
        res_var_name = context.get_output_tileable_variable(op.outputs[0])
        kw_str = self.generate_call_args_with_attributes(
            op, context, kw_keys=["axes"], skip_none=True
        )
        return [f"{res_var_name} = np.fft.{func_name}({input_var_name}, {kw_str})"]


@register_op_adapter([TensorFFTFreq, TensorRFFTFreq])
class TensorFFTFreqAdapter(SPEOperatorAdapter):
    def generate_code(
        self, op: Union[TensorFFTFreq, TensorRFFTFreq], context: SPECodeContext
    ) -> List[str]:
        context.register_import("numpy", "np")
        func_name = type(op).__name__.lower()[6:]
        res_var_name = context.get_output_tileable_variable(op.outputs[0])
        kw_str = self.generate_call_args_with_attributes(
            op, context, "n", kw_keys=["d"]
        )
        return [f"{res_var_name} = np.fft.{func_name}({kw_str})"]
