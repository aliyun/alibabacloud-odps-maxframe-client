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


from ..core import OutputType
from ..core.operator import (
    HasInput,
    MapReduceOperator,
    Operator,
    ShuffleProxy,
    TileableOperatorMixin,
)
from ..serialization.serializables import DataTypeField


class TensorOperatorMixin(TileableOperatorMixin):
    __slots__ = ()
    _op_module_ = "tensor"
    _output_type_ = OutputType.tensor

    def new_tensors(
        self,
        inputs,
        shape=None,
        dtype=None,
        order=None,
        chunks=None,
        nsplits=None,
        output_limit=None,
        kws=None,
        **kw
    ):
        return self.new_tileables(
            inputs,
            shape=shape,
            chunks=chunks,
            nsplits=nsplits,
            output_limit=output_limit,
            kws=kws,
            dtype=dtype,
            order=order,
            **kw
        )

    def new_tensor(self, inputs, shape, dtype=None, order=None, **kw):
        if getattr(self, "output_limit") != 1:
            raise TypeError("cannot new tensor with more than 1 outputs")
        return self.new_tensors(inputs, shape=shape, dtype=dtype, order=order, **kw)[0]


class TensorOperator(Operator):
    _output_type_ = OutputType.tensor

    dtype = DataTypeField("dtype", default=None)


class TensorHasInput(HasInput):
    _output_type_ = OutputType.tensor

    dtype = DataTypeField("dtype", default=None)


class TensorShuffleProxy(ShuffleProxy, TensorOperatorMixin):
    _output_type_ = OutputType.tensor

    dtype = DataTypeField("dtype", default=None)


class TensorMapReduceOperator(MapReduceOperator):
    _output_type_ = OutputType.tensor

    dtype = DataTypeField("dtype", default=None)
