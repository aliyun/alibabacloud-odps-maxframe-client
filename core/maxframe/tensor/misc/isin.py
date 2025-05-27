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

from typing import Union

import numpy as np

from ... import opcodes
from ...serialization.serializables import BoolField
from ...typing_ import TileableType
from ..core import TensorOrder
from ..datasource import tensor as astensor
from ..operators import TensorOperator, TensorOperatorMixin


class TensorIsIn(TensorOperator, TensorOperatorMixin):
    _op_type_ = opcodes.ISIN

    assume_unique = BoolField("assume_unique")
    invert = BoolField("invert")

    def __call__(self, element, test_elements):
        self.dtype = np.dtype(bool)
        return self.new_tensor(
            [element, test_elements], shape=element.shape, order=TensorOrder.C_ORDER
        )


def isin(
    element: Union[TileableType, np.ndarray],
    test_elements: Union[TileableType, np.ndarray, list],
    assume_unique: bool = False,
    invert: bool = False,
):
    """
    Calculates `element in test_elements`, broadcasting over `element` only.
    Returns a boolean array of the same shape as `element` that is True
    where an element of `element` is in `test_elements` and False otherwise.

    Parameters
    ----------
    element : array_like
        Input tensor.
    test_elements : array_like
        The values against which to test each value of `element`.
        This argument is flattened if it is a tensor or array_like.
        See notes for behavior with non-array-like parameters.
    assume_unique : bool, optional
        If True, the input tensors are both assumed to be unique, which
        can speed up the calculation.  Default is False.
    invert : bool, optional
        If True, the values in the returned tensor are inverted, as if
        calculating `element not in test_elements`. Default is False.
        ``mt.isin(a, b, invert=True)`` is equivalent to (but faster
        than) ``mt.invert(mt.isin(a, b))``.

    Returns
    -------
    isin : Tensor, bool
        Has the same shape as `element`. The values `element[isin]`
        are in `test_elements`.

    See Also
    --------
    in1d : Flattened version of this function.

    Notes
    -----

    `isin` is an element-wise function version of the python keyword `in`.
    ``isin(a, b)`` is roughly equivalent to
    ``mt.array([item in b for item in a])`` if `a` and `b` are 1-D sequences.

    `element` and `test_elements` are converted to tensors if they are not
    already. If `test_elements` is a set (or other non-sequence collection)
    it will be converted to an object tensor with one element, rather than a
    tensor of the values contained in `test_elements`. This is a consequence
    of the `tensor` constructor's way of handling non-sequence collections.
    Converting the set to a list usually gives the desired behavior.

    Examples
    --------
    >>> import maxframe.tensor as mt

    >>> element = 2*mt.arange(4).reshape((2, 2))
    >>> element.execute()
    array([[0, 2],
           [4, 6]])
    >>> test_elements = [1, 2, 4, 8]
    >>> mask = mt.isin(element, test_elements)
    >>> mask.execute()
    array([[ False,  True],
           [ True,  False]])
    >>> element[mask].execute()
    array([2, 4])
    >>> mask = mt.isin(element, test_elements, invert=True)
    >>> mask.execute()
    array([[ True, False],
           [ False, True]])
    >>> element[mask]
    array([0, 6])

    Because of how `array` handles sets, the following does not
    work as expected:

    >>> test_set = {1, 2, 4, 8}
    >>> mt.isin(element, test_set).execute()
    array([[ False, False],
           [ False, False]])

    Casting the set to a list gives the expected result:

    >>> mt.isin(element, list(test_set)).execute()
    array([[ False,  True],
           [ True,  False]])
    """
    element, test_elements = astensor(element), astensor(test_elements).ravel()
    op = TensorIsIn(assume_unique=assume_unique, invert=invert)
    return op(element, test_elements)
