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

from ....tensor.reduction.core import TensorCumReduction, TensorReduction
from ....tensor.reduction.nanvar import TensorNanVar
from ....tensor.reduction.var import TensorVar
from ..core import register_op_adapter
from .core import TensorUnaryCommonAdapter


@register_op_adapter(TensorReduction)
class TensorReductionAdapter(TensorUnaryCommonAdapter):
    def _get_op_kw_args(self, op: TensorReduction) -> List[str]:
        return ["axis", "keepdims"]


@register_op_adapter([TensorVar, TensorNanVar])
class TensorVarAdapter(TensorReductionAdapter):
    def _get_op_kw_args(self, op: TensorReduction) -> List[str]:
        return super()._get_op_kw_args(op) + ["ddof"]


@register_op_adapter(TensorCumReduction)
class TensorCumReductionAdapter(TensorUnaryCommonAdapter):
    def _get_op_kw_args(self, op: TensorCumReduction) -> List[str]:
        return ["axis"]
