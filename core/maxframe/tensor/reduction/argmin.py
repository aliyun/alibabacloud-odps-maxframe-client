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

import numpy as np

from ... import opcodes
from ...serialization.serializables import AnyField, TupleField
from ..utils import validate_axis
from .core import TensorArgReductionMixin, TensorReduction


class TensorArgmin(TensorReduction, TensorArgReductionMixin):
    _op_type_ = opcodes.ARGMIN
    _func_name = "argmin"
    _agg_func_name = "min"

    offset = AnyField("offset", default=None)
    total_shape = TupleField("total_shape", default=None)

    def __init__(self, dtype=None, **kw):
        if dtype is None:
            dtype = np.dtype(int)
        super().__init__(dtype=dtype, **kw)


def argmin(a, axis=None, out=None):
    """
    Returns the indices of the minimum values along an axis.

    Parameters
    ----------
    a : array_like
        Input tensor.
    axis : int, optional
        By default, the index is into the flattened tensor, otherwise
        along the specified axis.
    out : Tensor, optional
        If provided, the result will be inserted into this tensor. It should
        be of the appropriate shape and dtype.

    Returns
    -------
    index_array : Tensor of ints
        Tensor of indices into the tensor. It has the same shape as `a.shape`
        with the dimension along `axis` removed.

    See Also
    --------
    Tensor.argmin, argmax
    amin : The minimum value along a given axis.
    unravel_index : Convert a flat index into an index tuple.

    Notes
    -----
    In case of multiple occurrences of the minimum values, the indices
    corresponding to the first occurrence are returned.

    Examples
    --------
    >>> import maxframe.tensor as mt

    >>> a = mt.arange(6).reshape(2,3)
    >>> a.execute()
    array([[0, 1, 2],
           [3, 4, 5]])
    >>> mt.argmin(a).execute()
    0
    >>> mt.argmin(a, axis=0).execute()
    array([0, 0, 0])
    >>> mt.argmin(a, axis=1).execute()
    array([0, 0])

    Indices of the minimum elements of a N-dimensional tensor:

    >>> ind = mt.unravel_index(mt.argmin(a, axis=None), a.shape)
    >>> ind.execute()
    (0, 0)
    >>> a[ind]

    >>> b = mt.arange(6)
    >>> b[4] = 0
    >>> b.execute()
    array([0, 1, 2, 3, 0, 5])
    >>> mt.argmin(b).execute()  # Only the first occurrence is returned.
    0

    """
    axis = validate_axis(a.ndim, axis) if axis is not None else None
    op = TensorArgmin(axis=axis, dtype=np.dtype(int))
    return op(a, out=out)
