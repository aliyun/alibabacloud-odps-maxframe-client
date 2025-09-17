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

from typing import List, Type

from ...typing_ import OperatorType
from .core import SPECodeContext, SPEOperatorAdapter, register_op_adapter

_import_aliases = {
    "np": "numpy",
    "npl": "numpy.linalg",
    "pd": "pandas",
    "spl": "scipy.linalg",
}


def build_method_call_adapter(
    op_cls: Type[OperatorType],
    method_name: str,
    *call_args,
    skip_none: bool = True,
    source_module: str = None,
    **call_kwargs,
) -> Type[SPEOperatorAdapter]:
    class MethodCallAdapter(SPEOperatorAdapter):
        def generate_code(self, op: OperatorType, context: SPECodeContext) -> List[str]:
            if source_module in _import_aliases:
                context.register_import(_import_aliases[source_module], source_module)
            elif source_module:
                context.register_import(source_module)

            input_var_name = source_module or context.get_input_tileable_variable(
                op.inputs[0]
            )
            res_var_names = [
                context.get_output_tileable_variable(out) for out in op.outputs
            ]
            args_str = self.generate_call_args_with_attributes(
                op, context, *call_args, skip_none=skip_none, **call_kwargs
            )
            res_var_names_str = ", ".join(res_var_names)
            return [f"{res_var_names_str} = {input_var_name}.{method_name}({args_str})"]

    MethodCallAdapter.__name__ = op_cls.__name__ + "Adapter"
    return register_op_adapter(op_cls)(MethodCallAdapter)
