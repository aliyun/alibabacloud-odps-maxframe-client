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
from ...serialization.serializables import KeyField, TupleField
from ..datasource import tensor as astensor
from ..operators import TensorHasInput, TensorOperatorMixin


class TensorBroadcastTo(TensorHasInput, TensorOperatorMixin):
    _op_type_ = opcodes.BROADCAST_TO

    _input = KeyField("input")
    shape = TupleField("shape", default=None)

    def __call__(self, tensor, shape):
        return self.new_tensor([tensor], shape)


def broadcast_to(tensor, shape):
    """Broadcast a tensor to a new shape.

    Parameters
    ----------
    tensor : array_like
        The tensor to broadcast.
    shape : tuple
        The shape of the desired array.

    Returns
    -------
    broadcast : Tensor

    Raises
    ------
    ValueError
        If the tensor is not compatible with the new shape according to MaxFrame's
        broadcasting rules.

    Examples
    --------
    >>> import maxframe.tensor as mt

    >>> x = mt.array([1, 2, 3])
    >>> mt.broadcast_to(x, (3, 3)).execute()
    array([[1, 2, 3],
           [1, 2, 3],
           [1, 2, 3]])
    """
    from ..core import Tensor

    tensor = tensor if isinstance(tensor, Tensor) else astensor(tensor)
    shape = tuple(shape) if isinstance(shape, (list, tuple)) else (shape,)

    if any(np.isnan(s) for s in tensor.shape):
        raise ValueError(
            "input tensor has unknown shape, need to call `.execute()` first"
        )

    if tensor.shape == shape:
        return tensor

    new_ndim = len(shape) - tensor.ndim
    if new_ndim < 0:
        raise ValueError(
            "input operator has more dimensions than allowed by the axis remapping"
        )
    if any(o != n for o, n in zip(tensor.shape, shape[new_ndim:]) if o != 1):
        raise ValueError(
            "operators could not be broadcast together "
            f"with remapped shapes [original->remapped]: {tensor.shape} "
            f"and requested shape {shape}"
        )

    op = TensorBroadcastTo(shape=shape, dtype=tensor.dtype, sparse=tensor.issparse())
    return op(tensor, shape)
