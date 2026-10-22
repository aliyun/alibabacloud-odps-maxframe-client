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
from maxframe.liteframe.operators.core import LiteFrameOperator, LiteFrameOperatorMixin
from maxframe.serialization.serializables import (
    AnyField,
    BoolField,
    Float64Field,
    Int64Field,
)
from maxframe.tensor.rechunk.rechunk import TensorRechunk


class LiteFrameRechunk(LiteFrameOperatorMixin, LiteFrameOperator):
    _op_type_ = opcodes.RECHUNK

    chunk_size = AnyField("chunk_size", default=None)
    axis = Int64Field("axis", default=None)
    factor = Float64Field("factor", default=None)
    num_partitions = Int64Field("num_partitions", default=None)
    rough = BoolField("rough", default=False)

    validate_rebalance_args = staticmethod(TensorRechunk.validate_rebalance_args)
    normalize_rechunk_axis = staticmethod(TensorRechunk.normalize_rechunk_axis)

    def __call__(self, x):
        return self.new_liteframe(
            [x],
            shape=x.shape,
            physical_dtypes=x._physical_dtypes,
            frame_metadata=x.frame_metadata,
        )
