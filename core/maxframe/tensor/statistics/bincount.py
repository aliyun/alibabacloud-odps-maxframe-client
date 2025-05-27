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

from typing import List, Optional

import numpy as np

from ... import opcodes
from ...serialization.serializables import Int64Field, ReferenceField
from ...typing_ import EntityType
from ..datasource import tensor as astensor
from ..operators import TensorMapReduceOperator, TensorOperatorMixin

_DEFAULT_CHUNK_SIZE_LIMIT = 1e8


class TensorBinCount(TensorMapReduceOperator, TensorOperatorMixin):
    _op_type_ = opcodes.BINCOUNT

    weights = ReferenceField("weights", default=None)
    minlength: Optional[int] = Int64Field("minlength", default=0)
    chunk_size_limit: int = Int64Field("chunk_size_limit")

    chunk_count: Optional[int] = Int64Field("chunk_count")
    tileable_right_bound: Optional[int] = Int64Field("tileable_right_bound")

    def __call__(self, x, weights=None):
        inputs = [x]
        self.weights = weights
        dtype = np.dtype(np.int_)
        if weights is not None:
            inputs.append(weights)
            dtype = weights.dtype
        return self.new_tensor(inputs, dtype=dtype, shape=(np.nan,))

    @classmethod
    def _set_inputs(cls, op: "TensorBinCount", inputs: List[EntityType]):
        super()._set_inputs(op, inputs)
        if len(inputs) > 1:
            op.weights = inputs[1]


def bincount(x, weights=None, minlength=0, chunk_size_limit=None):
    """
    Count number of occurrences of each value in array of non-negative ints.

    The number of bins (of size 1) is one larger than the largest value in
    `x`. If `minlength` is specified, there will be at least this number
    of bins in the output array (though it will be longer if necessary,
    depending on the contents of `x`).
    Each bin gives the number of occurrences of its index value in `x`.
    If `weights` is specified the input array is weighted by it, i.e. if a
    value ``n`` is found at position ``i``, ``out[n] += weight[i]`` instead
    of ``out[n] += 1``.

    Parameters
    ----------
    x : tensor or array_like, 1 dimension, nonnegative ints
        Input array.
    weights : tensor or array_like, optional
        Weights, array of the same shape as `x`.
    minlength : int, optional
        A minimum number of bins for the output array.

    Returns
    -------
    out : tensor of ints
        The result of binning the input array.
        The length of `out` is equal to ``np.amax(x)+1``.

    Raises
    ------
    ValueError
        If the input is not 1-dimensional, or contains elements with negative
        values, or if `minlength` is negative.
    TypeError
        If the type of the input is float or complex.

    See Also
    --------
    histogram, digitize, unique

    Examples
    --------
    >>> import maxframe.tensor as mt
    >>> mt.bincount(mt.arange(5)).execute()
    array([1, 1, 1, 1, 1])
    >>> mt.bincount(mt.tensor([0, 1, 1, 3, 2, 1, 7])).execute()
    array([1, 3, 1, 1, 0, 0, 0, 1])

    The input array needs to be of integer dtype, otherwise a
    TypeError is raised:

    >>> mt.bincount(mt.arange(5, dtype=float)).execute()
    Traceback (most recent call last):
      ....execute()
    TypeError: Cannot cast array data from dtype('float64') to dtype('int64')
    according to the rule 'safe'

    A possible use of ``bincount`` is to perform sums over
    variable-size chunks of an array, using the ``weights`` keyword.

    >>> w = mt.array([0.3, 0.5, 0.2, 0.7, 1., -0.6]) # weights
    >>> x = mt.array([0, 1, 1, 2, 2, 2])
    >>> mt.bincount(x, weights=w).execute()
    array([ 0.3,  0.7,  1.1])
    """
    x = astensor(x)
    weights = astensor(weights) if weights is not None else None

    if not np.issubdtype(x.dtype, np.int_):
        raise TypeError(f"Cannot cast array data from {x.dtype} to {np.dtype(np.int_)}")
    if x.ndim != 1:
        raise ValueError("'x' must be 1 dimension")
    if minlength < 0:
        raise ValueError("'minlength' must not be negative")

    chunk_size_limit = (
        chunk_size_limit if chunk_size_limit is not None else _DEFAULT_CHUNK_SIZE_LIMIT
    )
    op = TensorBinCount(minlength=minlength, chunk_size_limit=chunk_size_limit)
    return op(x, weights=weights)
