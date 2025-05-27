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

from abc import ABC, abstractmethod
from typing import List

from ....tensor.operators import TensorOperator
from ..core import SPECodeContext, SPEOperatorAdapter


class TensorUnaryCommonAdapter(SPEOperatorAdapter, ABC):
    @classmethod
    def _gen_func_name(cls, op: TensorOperator):
        return f"np.{op._func_name}"

    def generate_code(self, op: TensorOperator, context: SPECodeContext) -> List[str]:
        context.register_import("numpy", "np")
        call_args = self.generate_call_args_with_attributes(
            op,
            context,
            "_input",
            skip_none=True,
            kw_keys=self._get_op_kw_args(op),
        )
        res_var_name = context.get_output_tileable_variable(op.outputs[0])
        return [f"{res_var_name} = {self._gen_func_name(op)}({call_args})"]

    @abstractmethod
    def _get_op_kw_args(self, op: TensorOperator) -> List[str]:
        raise NotImplementedError
