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
from ...serialization.serializables import KeyField, ListField
from ..core import TENSOR_TYPE, TensorOrder
from ..operators import TensorHasInput, TensorOperatorMixin
from ..utils import filter_inputs
from .core import calc_shape, process_index

FANCY_INDEX_TYPES = TENSOR_TYPE + (np.ndarray,)


class TensorIndex(TensorHasInput, TensorOperatorMixin):
    _op_type_ = opcodes.INDEX

    _input = KeyField("input")
    indexes = ListField("indexes", default=None)

    @classmethod
    def _set_inputs(cls, op: "TensorIndex", inputs: List[EntityData]):
        super()._set_inputs(op, inputs)
        inputs_iter = iter(op._inputs[1:])
        new_indexes = [
            next(inputs_iter) if isinstance(index, ENTITY_TYPE) else index
            for index in op.indexes
        ]
        op.indexes = new_indexes

    def on_output_modify(self, new_output):
        from .setitem import TensorIndexSetValue

        if self.create_view:
            a = self.input
            op = TensorIndexSetValue(
                dtype=a.dtype,
                sparse=a.issparse(),
                indexes=tuple(self.indexes),
                value=new_output,
            )
            return op(a, self.indexes, new_output)

    def on_input_modify(self, new_input):
        if self.create_view:
            new_op = self.copy().reset_key()
            new_inputs = [new_input] + self.inputs[1:]
            return new_op.new_tensor(new_inputs, shape=self.outputs[0].shape)

    def __call__(self, a, index, shape, order):
        self.indexes = list(index)
        return self.new_tensor(filter_inputs([a] + list(index)), shape, order=order)


def _is_bool_index(index_obj):
    return isinstance(index_obj, TENSOR_TYPE) and index_obj.dtype == np.bool_


def _is_fancy_index(index_obj):
    return isinstance(index_obj, FANCY_INDEX_TYPES) and index_obj.dtype != np.bool_


def _is_create_view(index):
    # is view if all of index is slice, int or newaxis
    return all(isinstance(ind, (slice, Integral)) or ind is None for ind in index)


def _calc_order(a, index):
    if a.order == TensorOrder.C_ORDER:
        return TensorOrder.C_ORDER

    in_axis = 0
    for ind in index:
        if _is_bool_index(ind):
            in_axis += ind.ndim
            return TensorOrder.C_ORDER
        elif _is_fancy_index(ind):
            in_axis += 1
            return TensorOrder.C_ORDER
        elif ind is None:
            continue
        elif isinstance(ind, slice):
            shape = a.shape[in_axis]
            if shape is np.nan:
                return TensorOrder.C_ORDER
            slc = ind.indices(shape)
            if slc[0] == 0 and slc[1] == shape and slc[2] == 1:
                continue
            else:
                return TensorOrder.C_ORDER
        else:
            assert isinstance(ind, Integral)
            in_axis += 1
            return TensorOrder.C_ORDER

    return TensorOrder.F_ORDER


def _getitem_nocheck(a, item, convert_bool_to_fancy=None):
    index = process_index(a.ndim, item, convert_bool_to_fancy=convert_bool_to_fancy)
    if convert_bool_to_fancy is False:
        # come from __setitem__, the bool index is not converted to fancy index
        # if multiple bool indexes or bool + fancy indexes exist,
        # thus the shape will be wrong,
        # here we just convert when calculating shape,
        # refer to issue #1282.
        shape = calc_shape(a.shape, process_index(a.ndim, index))
    else:
        shape = calc_shape(a.shape, index)
    tensor_order = _calc_order(a, index)
    op = TensorIndex(
        dtype=a.dtype,
        sparse=a.issparse(),
        indexes=list(index),
        create_view=_is_create_view(index),
    )
    return op(a, index, tuple(shape), order=tensor_order)


def _getitem(a, item):
    if isinstance(item, (list, tuple)) and all(
        isinstance(it, slice) and it == slice(None) for it in item
    ):
        # nothing to do
        return a

    # TODO(jisheng): field access, e.g. t['a'], t[['a', 'b']]
    return _getitem_nocheck(a, item)
