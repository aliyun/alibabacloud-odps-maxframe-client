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

from ....tensor.indexing import TensorFillDiagonal
from ....tensor.misc import (
    TensorArgwhere,
    TensorCopyTo,
    TensorDelete,
    TensorDiff,
    TensorGetShape,
    TensorIsIn,
    TensorRepeat,
    TensorSearchsorted,
    TensorSplit,
    TensorSqueeze,
    TensorSwapAxes,
    TensorTranspose,
    TensorTrapezoid,
    TensorUnique,
    TensorWhere,
)
from ....tensor.misc.insert import TensorInsert
from ..core import SPECodeContext, SPEOperatorAdapter, register_op_adapter
from ..utils import build_method_call_adapter

_trapz_func = "trapezoid" if hasattr(np, "trapezoid") else "trapz"


TensorArgwhereAdapter = build_method_call_adapter(
    TensorArgwhere, "argwhere", 0, source_module="np"
)
TensorDeleteAdapter = build_method_call_adapter(
    TensorDelete, "delete", 0, "index_obj", kw_keys=["axis"], source_module="np"
)
TensorDiffAdapter = build_method_call_adapter(
    TensorDiff, "diff", 0, skip_none=True, kw_keys=["n", "axis"], source_module="np"
)
TensorFillDiagonalAdapter = build_method_call_adapter(
    TensorFillDiagonal,
    "fill_diagonal",
    0,
    "val",
    skip_none=True,
    kw_keys=["wrap"],
    source_module="np",
)
TensorInsertAdapter = build_method_call_adapter(
    TensorInsert,
    "insert",
    0,
    "index_obj",
    kw_keys=["values", "axis"],
    source_module="np",
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
TensorTrapezoidAdapter = build_method_call_adapter(
    TensorTrapezoid,
    _trapz_func,
    skip_none=True,
    kw_keys=["y", "x", "dx", "axis"],
    source_module="np",
)
TensorSqueezeAdapter = build_method_call_adapter(
    TensorSqueeze, "squeeze", 0, kw_keys=["axis"], source_module="np"
)
TensorSwapAxesAdapter = build_method_call_adapter(
    TensorSwapAxes, "swapaxes", "axis1", "axis2", source_module="np"
)
TensorTransposeAdapter = build_method_call_adapter(
    TensorTranspose, "transpose", 0, kw_keys=["axes"], source_module="np"
)
TensorWhereAdapter = build_method_call_adapter(
    TensorWhere, "where", "condition", "x", "y", source_module="np"
)


@register_op_adapter(TensorCopyTo)
class TensorCopyToAdapter(SPEOperatorAdapter):
    def generate_code(self, op: TensorCopyTo, context: SPECodeContext) -> List[str]:
        context.register_import("numpy", "np")
        dst_var_name = context.get_input_tileable_variable(op.inputs[1])
        src_var_name = context.get_input_tileable_variable(op.inputs[0])
        res_var_name = context.get_output_tileable_variable(op.outputs[0])
        args_str = self.generate_call_args_with_attributes(
            op, context, kw_keys=["casting", "where"], skip_none=True
        )
        return [
            f"{res_var_name} = {dst_var_name}.copy()",
            f"np.copyto({res_var_name}, {src_var_name}, {args_str})",
        ]


@register_op_adapter(TensorGetShape)
class TensorGetShapeAdapter(SPEOperatorAdapter):
    def generate_code(self, op: TensorGetShape, context: SPECodeContext) -> List[str]:
        context.register_import("numpy", "np")
        input_var_name = context.get_input_tileable_variable(op.inputs[0])
        res_var_names = ", ".join(
            context.get_output_tileable_variable(out) for out in op.outputs
        )
        return [f"{res_var_names} = np.shape({input_var_name})"]


@register_op_adapter(TensorSplit)
class TensorSplitAdapter(SPEOperatorAdapter):
    def generate_code(self, op: TensorSplit, context: SPECodeContext) -> List[str]:
        context.register_import("numpy", "np")
        input_var_name = context.get_input_tileable_variable(op.inputs[0])
        res_var_names = ", ".join(
            context.get_output_tileable_variable(out) for out in op.outputs
        )
        idx_var = self.translate_var(context, op.indices_or_sections)
        axis_var = self.translate_var(context, op.axis)
        return [
            f"{res_var_names} = np.split({input_var_name}, {idx_var} axis={axis_var})"
        ]


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
