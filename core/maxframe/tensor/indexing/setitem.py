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

from numbers import Integral
from typing import List

import numpy as np

from ... import opcodes
from ...core import ENTITY_TYPE, EntityData
from ...core.operator import OperatorStage
from ...serialization.serializables import AnyField, BoolField, KeyField, TupleField
from ...tensor import tensor as astensor
from ..core import TENSOR_TYPE
from ..operators import TensorMapReduceOperator, TensorOperatorMixin
from ..utils import filter_inputs
from .core import process_index


class TensorIndexSetValue(TensorMapReduceOperator, TensorOperatorMixin):
    _op_type_ = opcodes.INDEXSETVALUE

    input = KeyField("input")
    indexes = TupleField("indexes")
    value = AnyField("value")
    is_fancy_index = BoolField("is_fancy_index")
    input_nsplits = TupleField("input_nsplits")
    chunk_offsets = TupleField("chunk_offsets")
    shuffle_axes = TupleField("shuffle_axes")

    def __init__(
        self,
        indexes=None,
        value=None,
        is_fancy_index=None,
        input_nsplits=None,
        chunk_offsets=None,
        shuffle_axes=None,
        **kw,
    ):
        super().__init__(
            indexes=indexes,
            value=value,
            is_fancy_index=is_fancy_index,
            input_nsplits=input_nsplits,
            chunk_offsets=chunk_offsets,
            shuffle_axes=shuffle_axes,
            **kw,
        )

    @classmethod
    def _set_inputs(cls, op: "TensorIndexSetValue", inputs: List[EntityData]):
        super()._set_inputs(op, inputs)
        op.input = op._inputs[0]
        if op.stage == OperatorStage.reduce:
            op.input = op._inputs[0]
            return
        elif op.stage == OperatorStage.map:
            inputs_iter = iter(op._inputs)
        else:
            op.input = op._inputs[0]
            inputs_iter = iter(op._inputs[1:])
        new_indexes = [
            next(inputs_iter) if isinstance(index, ENTITY_TYPE) else index
            for index in op.indexes
        ]
        op.indexes = tuple(new_indexes)
        if isinstance(op.value, ENTITY_TYPE):
            op.value = next(inputs_iter)

    def __call__(self, a, index, value):
        inputs = filter_inputs([a] + list(index) + [value])
        self.indexes = tuple(index)
        self.value = value
        return self.new_tensor(inputs, a.shape, order=a.order)

    def on_output_modify(self, new_output):
        return new_output

    def on_input_modify(self, new_input):
        new_op = self.copy().reset_key()
        new_inputs = [new_input] + self.inputs[1:]
        return new_op.new_tensor(new_inputs, shape=self.outputs[0].shape)


def _check_support(indexes):
    if all(
        (
            isinstance(ix, (TENSOR_TYPE, np.ndarray))
            and ix.dtype != np.bool_
            or isinstance(ix, slice)
            and ix == slice(None)
        )
        for ix in indexes
    ):
        if any(isinstance(ix, (TENSOR_TYPE, np.ndarray)) for ix in indexes):
            return True
    for index in indexes:
        if isinstance(index, (slice, Integral)):
            pass
        elif isinstance(index, (np.ndarray, TENSOR_TYPE)) and index.dtype == np.bool_:
            pass
        else:  # pragma: no cover
            raise NotImplementedError(
                "Only slice, int, or bool indexing "
                f"supported by now, got {type(index)}"
            )
    return False


def _setitem(a, item, value):
    index = process_index(a.ndim, item, convert_bool_to_fancy=False)
    if not (np.isscalar(value) or (isinstance(value, tuple) and a.dtype.fields)):
        # do not convert for tuple when dtype is record type.
        value = astensor(value)

    is_fancy_index = _check_support(index)
    if is_fancy_index:
        index = [astensor(ind) if isinstance(ind, np.ndarray) else ind for ind in index]

    # __setitem__ on a view should be still a view, see GH #732.
    op = TensorIndexSetValue(
        dtype=a.dtype,
        sparse=a.issparse(),
        is_fancy_index=is_fancy_index,
        indexes=tuple(index),
        value=value,
        create_view=a.op.create_view,
    )
    ret = op(a, index, value)
    a.data = ret.data
