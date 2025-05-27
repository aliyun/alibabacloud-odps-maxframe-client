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

from typing import List

import numpy as np

from ... import opcodes
from ...core import EntityData
from ...serialization.serializables import KeyField, StringField
from ...utils import make_dtype
from ..operators import TensorHasInput, TensorOperatorMixin
from ..utils import get_order


class TensorAstype(TensorHasInput, TensorOperatorMixin):
    _op_type_ = opcodes.ASTYPE

    _input = KeyField("input")
    order = StringField("order", default=None)
    casting = StringField("casting", default=None)

    def __init__(self, dtype=None, sparse=False, **kw):
        super().__init__(dtype=dtype, sparse=sparse, **kw)

    @classmethod
    def _set_inputs(cls, op: "TensorAstype", inputs: List[EntityData]):
        super()._set_inputs(op, inputs)
        op._input = op._inputs[0]

    def __call__(self, tensor, order=None):
        return self.new_tensor([tensor], tensor.shape, order=order)


def astype(tensor, dtype, order="K", casting="unsafe", copy=True):
    """
    Copy of the tensor, cast to a specified type.

    Parameters
    ----------
    dtype : str or dtype
        Typecode or data-type to which the array is cast.
    casting : {'no', 'equiv', 'safe', 'same_kind', 'unsafe'}, optional
        Controls what kind of data casting may occur. Defaults to 'unsafe'
        for backwards compatibility.
          * 'no' means the data types should not be cast at all.
          * 'equiv' means only byte-order changes are allowed.
          * 'safe' means only casts which can preserve values are allowed.
          * 'same_kind' means only safe casts or casts within a kind,
            like float64 to float32, are allowed.
          * 'unsafe' means any data conversions may be done.
    order : {'C', 'F', 'A', 'K'}, optional
        Controls the memory layout order of the result.
        'C' means C order, 'F' means Fortran order, 'A'
        means 'F' order if all the arrays are Fortran contiguous,
        'C' order otherwise, and 'K' means as close to the
        order the array elements appear in memory as possible.
        Default is 'K'.
    copy : bool, optional
        By default, astype always returns a newly allocated array. If this
        is set to false, and the `dtype`, `order`, and `subok`
        requirements are satisfied, the input array is returned instead
        of a copy.

    Returns
    -------
    arr_t : Tensor
        Unless `copy` is False and the other conditions for returning the input
        array are satisfied (see description for `copy` input parameter), `arr_t`
        is a new tensor of the same shape as the input array, with dtype, order
        given by `dtype`, `order`.

    Notes
    -----
    astype method returns an error if the string
    dtype to cast to is not long enough in 'safe' casting mode to hold the max
    value of integer/float array that is being casted. Previously the casting
    was allowed even if the result was truncated.

    Raises
    ------
    ComplexWarning
        When casting from complex to float or int. To avoid this,
        one should use ``a.real.astype(t)``.

    Examples
    --------
    >>> import maxframe.tensor as mt
    >>> x = mt.array([1, 2, 2.5])
    >>> x.execute()
    array([ 1. ,  2. ,  2.5])

    >>> x.astype(int).execute()
    array([1, 2, 2])
    """
    dtype = make_dtype(dtype)
    tensor_order = get_order(order, tensor.order)

    if tensor.dtype == dtype and tensor.order == tensor_order:
        return tensor if not copy else tensor.copy(order=order)
    elif not np.can_cast(tensor.dtype, dtype, casting=casting):
        raise TypeError(
            f"Cannot cast array from {tensor.dtype!r} to {dtype!r} "
            f"according to the rule {casting}"
        )

    op = TensorAstype(
        dtype=dtype, order=order, casting=casting, sparse=tensor.issparse()
    )
    return op(tensor, order=tensor_order)
