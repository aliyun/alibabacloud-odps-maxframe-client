#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 1999-2024 Alibaba Group Holding Ltd.
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

import numpy as np

from ... import opcodes
from ...serialization.serializables import FieldTypes, KeyField, ListField
from ..core import TensorOrder
from ..datasource import tensor as astensor
from ..operators import TensorHasInput, TensorOperatorMixin
from ..utils import reverse_order


def _reorder(x, axes):
    if x is None:
        return
    return type(x)(x[ax] for ax in axes)


class TensorTranspose(TensorHasInput, TensorOperatorMixin):
    _op_type_ = opcodes.TRANSPOSE

    _input = KeyField("input")
    axes = ListField("axes", FieldTypes.int32, default=None)

    def __init__(self, axes=None, **kw):
        # transpose will create a view
        super().__init__(axes=axes, create_view=True, **kw)

    def __call__(self, a):
        shape = tuple(
            s if np.isnan(s) else int(s) for s in _reorder(a.shape, self.axes)
        )
        if self.axes == list(reversed(range(a.ndim))):
            # order reversed
            tensor_order = reverse_order(a.order)
        else:
            tensor_order = TensorOrder.C_ORDER
        return self.new_tensor([a], shape, order=tensor_order)

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        self._input = self._inputs[0]

    def on_output_modify(self, new_output):
        op = self.copy().reset_key()
        return op(new_output)

    def on_input_modify(self, new_input):
        op = self.copy().reset_key()
        return op(new_input)


def transpose(a, axes=None):
    """
    Permute the dimensions of a tensor.

    Parameters
    ----------
    a : array_like
        Input tensor.
    axes : list of ints, optional
        By default, reverse the dimensions, otherwise permute the axes
        according to the values given.

    Returns
    -------
    p : Tensor
        `a` with its axes permuted.  A view is returned whenever
        possible.

    See Also
    --------
    moveaxis
    argsort

    Notes
    -----
    Use `transpose(a, argsort(axes))` to invert the transposition of tensors
    when using the `axes` keyword argument.

    Transposing a 1-D array returns an unchanged view of the original tensor.

    Examples
    --------
    >>> import maxframe.tensor as mt

    >>> x = mt.arange(4).reshape((2,2))
    >>> x.execute()
    array([[0, 1],
           [2, 3]])

    >>> mt.transpose(x).execute()
    array([[0, 2],
           [1, 3]])

    >>> x = mt.ones((1, 2, 3))
    >>> mt.transpose(x, (1, 0, 2)).shape
    (2, 1, 3)

    """
    a = astensor(a)
    if axes:
        if len(axes) != a.ndim:
            raise ValueError("axes don't match tensor")

    if not axes:
        axes = list(range(a.ndim))[::-1]
    else:
        axes = list(axes)
    op = TensorTranspose(axes, dtype=a.dtype, sparse=a.issparse())
    return op(a)
