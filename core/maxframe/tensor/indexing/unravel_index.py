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

from collections.abc import Iterable
from typing import List

import numpy as np

from ... import opcodes
from ...core import EntityData, ExecutableTuple
from ...serialization.serializables import FieldTypes, KeyField, StringField, TupleField
from ..core import TensorOrder
from ..datasource import tensor as astensor
from ..operators import TensorHasInput, TensorOperatorMixin


class TensorUnravelIndex(TensorHasInput, TensorOperatorMixin):
    _op_type_ = opcodes.UNRAVEL_INDEX

    _input = KeyField("input")
    dims = TupleField("dims", FieldTypes.int32, default=None)
    order = StringField("order", default=None)

    def __init__(self, order=None, **kw):
        super().__init__(order=order, **kw)
        if self.order is None:
            self.order = "C"

    @property
    def output_limit(self):
        return float("inf")

    @classmethod
    def _set_inputs(cls, op: "TensorUnravelIndex", inputs: List[EntityData]):
        super()._set_inputs(op, inputs)
        op._input = op._inputs[0]

    def __call__(self, indices):
        order = TensorOrder.C_ORDER if self.order == "C" else TensorOrder.F_ORDER
        kws = [{"pos": i, "order": order} for i in range(len(self.dims))]
        return ExecutableTuple(
            self.new_tensors([indices], indices.shape, kws=kws, output_limit=len(kws))
        )


def unravel_index(indices, dims, order="C"):
    """
    Converts a flat index or tensor of flat indices into a tuple
    of coordinate tensors.

    Parameters
    ----------
    indices : array_like
        An integer tensor whose elements are indices into the flattened
        version of a tensor of dimensions ``dims``.
    dims : tuple of ints
        The shape of the tensor to use for unraveling ``indices``.
    order : {'C', 'F'}, optional
        Determines whether the indices should be viewed as indexing in
        row-major (C-style) or column-major (Fortran-style) order.

    Returns
    -------
    unraveled_coords : tuple of Tensor
        Each tensor in the tuple has the same shape as the ``indices``
        tensor.

    See Also
    --------
    ravel_multi_index

    Examples
    --------
    >>> import maxframe.tensor as mt

    >>> mt.unravel_index([22, 41, 37], (7,6)).execute()
    (array([3, 6, 6]), array([4, 5, 1]))

    >>> mt.unravel_index(1621, (6,7,8,9)).execute()
    (3, 1, 4, 1)
    """
    indices = astensor(indices)
    if isinstance(dims, Iterable):
        dims = tuple(dims)
    else:
        dims = (dims,)

    if order not in "CF":
        raise TypeError("only 'C' or 'F' order is permitted")

    op = TensorUnravelIndex(dims=dims, dtype=np.dtype(np.intp), order=order)
    return op(indices)
