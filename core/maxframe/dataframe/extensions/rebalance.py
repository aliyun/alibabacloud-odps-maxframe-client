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
from maxframe.dataframe.operators import DataFrameOperator, DataFrameOperatorMixin
from maxframe.serialization.serializables import BoolField, Float64Field, Int64Field
from maxframe.tensor.extensions.rebalance import RebalanceMixin


class DataFrameRebalance(RebalanceMixin, DataFrameOperatorMixin, DataFrameOperator):
    """Legacy rebalance op, kept for deserialization compatibility.

    New code constructs ``DataFrameRechunk`` directly (see
    ``maxframe.dataframe.misc.rechunk.rebalance``). This op exists solely so
    that already-serialized ``REBALANCE`` graph nodes can be deserialized; its
    tiler redirects to ``DataFrameRechunk`` via ``recursive_tile``.
    """

    _op_type_ = opcodes.REBALANCE

    axis = Int64Field("axis")
    factor = Float64Field("factor", default=None)
    num_partitions = Int64Field("num_partitions")
    rough = BoolField("rough", default=False)

    def __init__(self, output_types=None, **kw):
        super().__init__(_output_types=output_types, **kw)
