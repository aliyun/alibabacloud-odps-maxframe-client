# Copyright 1999-2026 Alibaba Group Holding Ltd.
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

from maxframe.codegen.spe.core import SPECodeContext
from maxframe.codegen.spe.tensor.core import TensorUnaryCommonAdapter
from maxframe.tensor.special.core import TensorSpecialUnaryOp


class TensorSpecialUnaryOpAdapter(TensorUnaryCommonAdapter):
    @classmethod
    def _gen_func_name(cls, op: TensorSpecialUnaryOp):
        return f"spspecial.{op._func_name}"

    @classmethod
    def _get_op_kw_args(self, op: TensorSpecialUnaryOp) -> List[str]:
        return []

    def generate_code(
        self, op: TensorSpecialUnaryOp, context: SPECodeContext
    ) -> List[str]:
        context.register_import("scipy.special", "spspecial")
        return super().generate_code(op, context)
