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
from ...core import EntityData
from ...serialization.serializables import Int32Field, KeyField
from ..datasource import tensor as astensor
from ..operators import TensorOperatorMixin
from ..utils import gen_random_seeds, validate_axis
from .core import TensorRandomMapReduceOperator


def _permutation_on_axis(ar, axis, rs, xp):
    try:
        return rs.permutation(ar, axis=axis)
    except TypeError:
        # numpy starts to support axis from 1.18
        if axis == 0:
            return rs.permutation(ar)
        indices = xp.arange(ar.shape[axis])
        rs.shuffle(indices)
        slc = (slice(None),) * axis + (indices,)
        return ar[slc]


class TensorPermutation(TensorRandomMapReduceOperator, TensorOperatorMixin):
    _op_type_ = opcodes.PERMUTATION

    input = KeyField("input")
    axis = Int32Field("axis")

    reduce_size = Int32Field("reduce_size")

    @classmethod
    def _set_inputs(cls, op: "TensorPermutation", inputs: List[EntityData]):
        super()._set_inputs(op, inputs)
        op.input = op._inputs[0]

    def __call__(self, x):
        return self.new_tensor([x], x.shape, order=x.order)


def permutation(random_state, x, axis=0, chunk_size=None):
    r"""
    Randomly permute a sequence, or return a permuted range.

    Parameters
    ----------
    x : int or array_like
        If `x` is an integer, randomly permute ``mt.arange(x)``.
        If `x` is an array, make a copy and shuffle the elements
        randomly.
    axis : int, optional
        The axis which `x` is shuffled along. Default is 0.
    chunk_size : : int or tuple of int or tuple of ints, optional
        Desired chunk size on each dimension
    Returns
    -------
    out : Tensor
        Permuted sequence or tensor range.
    Examples
    --------
    >>> import maxframe.tensor as mt
    >>> rng = mt.random.RandomState()
    >>> rng.permutation(10).execute()
    array([1, 2, 3, 7, 9, 8, 0, 6, 4, 5]) # random
    >>> rng.permutation([1, 4, 9, 12, 15]).execute()
    array([ 9,  4, 12,  1, 15]) # random
    >>> arr = mt.arange(9).reshape((3, 3))
    >>> rng.permutation(arr).execute()
    array([[3, 4, 5], # random
           [6, 7, 8],
           [0, 1, 2]])
    >>> rng.permutation("abc")
    Traceback (most recent call last):
        ...
    numpy.AxisError: x must be an integer or at least 1-dimensional
    """
    if isinstance(x, (Integral, np.integer)):
        from ..datasource import arange

        x = arange(x, chunk_size=chunk_size)
    else:
        x = astensor(x, chunk_size=chunk_size)
        if x.ndim < 1:
            raise np.AxisError("x must be an integer or at least 1-dimensional")

    axis = validate_axis(x.ndim, axis)
    seed = gen_random_seeds(1, random_state.to_numpy())[0]
    op = TensorPermutation(seed=seed, axis=axis, dtype=x.dtype, gpu=x.op.gpu)
    return op(x)
