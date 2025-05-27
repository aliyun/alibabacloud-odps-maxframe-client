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

import numpy as np

from ....tensor.misc import (
    TensorDiff,
    TensorIsIn,
    TensorRepeat,
    TensorSearchsorted,
    TensorSqueeze,
    TensorSwapAxes,
    TensorTranspose,
    TensorTrapezoid,
    TensorUnique,
    TensorWhere,
)
from ..core import SPECodeContext, SPEOperatorAdapter, register_op_adapter
from ..utils import build_method_call_adapter

_trapz_func = "trapezoid" if hasattr(np, "trapezoid") else "trapz"


TensorDiffAdapter = build_method_call_adapter(
    TensorDiff, "diff", 0, skip_none=True, kw_keys=["n", "axis"], source_module="np"
)
TensorIsInAdapter = build_method_call_adapter(
    TensorIsIn,
    "isin",
    0,
    1,
    skip_none=True,
    kw_keys=["assume_unique", "invert"],
    source_module="np",
)
TensorRepeatAdapter = build_method_call_adapter(
    TensorRepeat,
    "repeat",
    0,
    skip_none=True,
    kw_keys=["repeats", "axis"],
    source_module="np",
)
TensorSearchsortedAdapter = build_method_call_adapter(
    TensorSearchsorted,
    "searchsorted",
    "a",
    "v",
    kw_keys=["side", "sorter"],
    source_module="np",
)


@register_op_adapter(TensorTranspose)
class TensorTransposeAdapter(SPEOperatorAdapter):
    def generate_code(self, op: TensorTranspose, context: SPECodeContext) -> List[str]:
        context.register_import("numpy", "np")
        input_var_name = context.get_input_tileable_variable(op.inputs[0])
        res_var_name = context.get_output_tileable_variable(op.outputs[0])
        axes_var = self.translate_var(context, op.axes)
        return [f"{res_var_name} = np.transpose({input_var_name}, axes={axes_var})"]


TensorTrapezoidAdapter = build_method_call_adapter(
    TensorTrapezoid,
    _trapz_func,
    skip_none=True,
    kw_keys=["y", "x", "dx", "axis"],
    source_module="np",
)


@register_op_adapter(TensorSqueeze)
class TensorSqueezeAdapter(SPEOperatorAdapter):
    def generate_code(self, op: TensorSqueeze, context: SPECodeContext) -> List[str]:
        context.register_import("numpy", "np")
        input_var_name = context.get_input_tileable_variable(op.inputs[0])
        res_var_name = context.get_output_tileable_variable(op.outputs[0])
        axis_var = self.translate_var(context, op.axis)
        return [f"{res_var_name} = np.squeeze({input_var_name}, axis={axis_var})"]


@register_op_adapter(TensorUnique)
class TensorUniqueAdapter(SPEOperatorAdapter):
    def generate_code(self, op, context):
        context.register_import("numpy", "np")
        input_var_name = context.get_input_tileable_variable(op.inputs[0])
        output_var_names = [
            context.get_output_tileable_variable(out) for out in op.outputs
        ]
        return_index_var = self.translate_var(context, op.return_index)
        return_inverse_var = self.translate_var(context, op.return_inverse)
        return_counts_var = self.translate_var(context, op.return_counts)
        axis_var = self.translate_var(context, op.axis)
        return [
            f"{', '.join(output_var_names)} = "
            f"np.unique({input_var_name}, return_index={return_index_var}, "
            f"return_inverse={return_inverse_var}, return_counts={return_counts_var}, "
            f"axis={axis_var})"
        ]


TensorSwapAxesAdapter = build_method_call_adapter(
    TensorSwapAxes, "swapaxes", "axis1", "axis2", source_module="np"
)
TensorWhereAdapter = build_method_call_adapter(
    TensorWhere, "where", "condition", "x", "y", source_module="np"
)
