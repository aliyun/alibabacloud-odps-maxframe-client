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
from ..datasource import tensor as astensor
from ..operators import TensorHasInput, TensorOperatorMixin


class TensorArgwhere(TensorHasInput, TensorOperatorMixin):
    _op_type_ = opcodes.ARGWHERE

    def __call__(self, a):
        shape = (np.nan, a.ndim)
        return self.new_tensor([a], shape)


def argwhere(a):
    """
    Find the indices of tensor elements that are non-zero, grouped by element.

    Parameters
    ----------
    a : array_like
        Input data.

    Returns
    -------
    index_tensor : Tensor
        Indices of elements that are non-zero. Indices are grouped by element.

    See Also
    --------
    where, nonzero

    Notes
    -----
    ``mt.argwhere(a)`` is the same as ``mt.transpose(mt.nonzero(a))``.

    The output of ``argwhere`` is not suitable for indexing tensors.
    For this purpose use ``nonzero(a)`` instead.

    Examples
    --------
    >>> import maxframe.tensor as mt

    >>> x = mt.arange(6).reshape(2,3)
    >>> x.execute()
    array([[0, 1, 2],
           [3, 4, 5]])
    >>> mt.argwhere(x>1).execute()
    array([[0, 2],
           [1, 0],
           [1, 1],
           [1, 2]])

    """
    a = astensor(a).astype(bool, order="A")
    op = TensorArgwhere(dtype=np.dtype(np.intp))
    return op(a)
