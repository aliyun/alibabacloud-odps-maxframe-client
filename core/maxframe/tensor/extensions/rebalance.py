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

from maxframe import opcodes
from maxframe.serialization.serializables import BoolField, Float64Field, Int64Field
from maxframe.tensor.operators import TensorOperator, TensorOperatorMixin


class RebalanceMixin:
    """Shared ``__call__`` for legacy ``*Rebalance`` ops (kept for
    deserialization of already-serialized ``REBALANCE`` op nodes)."""

    def __call__(self, in_obj):
        self._output_types = in_obj.op.output_types
        return self.new_tileable([in_obj], kws=[in_obj.params])


class TensorRebalance(RebalanceMixin, TensorOperatorMixin, TensorOperator):
    """Legacy rebalance op, kept for deserialization compatibility.

    New code constructs ``TensorRechunk`` directly (see
    ``maxframe.tensor.rechunk.rechunk.rebalance``). This op exists solely so
    that already-serialized ``REBALANCE`` graph nodes can be deserialized; its
    tiler redirects to ``TensorRechunk`` via ``recursive_tile``.
    """

    _op_type_ = opcodes.REBALANCE

    axis = Int64Field("axis", default=None)
    factor = Float64Field("factor", default=None)
    num_partitions = Int64Field("num_partitions", default=None)
    rough = BoolField("rough", default=False)

    def __init__(self, output_types=None, **kw):
        super().__init__(_output_types=output_types, **kw)
