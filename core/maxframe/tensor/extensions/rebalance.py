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

from ... import opcodes
from ...serialization.serializables import Float64Field, Int64Field
from ..operators import TensorOperator, TensorOperatorMixin


class RebalanceMixin:
    def __call__(self, in_obj):
        self._output_types = in_obj.op.output_types
        return self.new_tileable([in_obj], kws=[in_obj.params])


class TensorRebalance(RebalanceMixin, TensorOperatorMixin, TensorOperator):
    _op_type_ = opcodes.REBALANCE

    axis = Int64Field("axis", default=None)
    factor = Float64Field("factor", default=None)
    num_partitions = Int64Field("num_partitions", default=None)

    def __init__(self, output_types=None, **kw):
        super().__init__(_output_types=output_types, **kw)


def rebalance(tensor, axis=0, factor=None, num_partitions=None):
    """
    Make data more balanced across entire cluster.

    Parameters
    ----------
    axis : int
        The axis to rebalance.
    factor : float
        Specified so that number of chunks after balance is
        total number of input chunks * factor.
    num_partitions : int
        Specified so the number of chunks are at most
        num_partitions.

    Returns
    -------
    Tensor
        Result of tensor after rebalanced.
    """
    if num_partitions is None and factor is None:
        raise ValueError("Need to specify num_partitions or factor")
    if num_partitions is not None and factor is not None:
        raise ValueError(
            "num_partitions and factor cannot be specified at the same time"
        )

    op = TensorRebalance(axis=axis, factor=factor, num_partitions=num_partitions)
    return op(tensor)
