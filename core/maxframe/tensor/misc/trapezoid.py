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
from ...serialization.serializables import Float64Field, Int8Field, KeyField
from ...typing_ import EntityType
from ..core import TensorOrder
from ..datasource import tensor as astensor
from ..operators import TensorOperator, TensorOperatorMixin
from ..utils import validate_axis


class TensorTrapezoid(TensorOperator, TensorOperatorMixin):
    _op_type_ = opcodes.TRAPEZOID

    y = KeyField("y")
    x = KeyField("x")
    dx = Float64Field("dx")
    axis = Int8Field("axis")

    @classmethod
    def _set_inputs(cls, op: "TensorTrapezoid", inputs: List[EntityType]):
        super()._set_inputs(op, inputs)
        op.y = op._inputs[0]
        if op.x is not None:
            op.x = op._inputs[-1]

    def __call__(self, y, x=None):
        inputs = [y]
        order = y.order
        if x is not None:
            x = astensor(x)
            inputs.append(x)
            if x.order == TensorOrder.C_ORDER:
                order = TensorOrder.C_ORDER

        shape = tuple(s for ax, s in enumerate(y.shape) if ax != self.axis)
        dtype = np.trapz(np.empty(1, dtype=y.dtype)).dtype
        return self.new_tensor(inputs, shape=shape, dtype=dtype, order=order)


def trapezoid(y, x=None, dx=1.0, axis=-1):
    """
    Integrate along the given axis using the composite trapezoidal rule.

    Integrate `y` (`x`) along given axis.

    Parameters
    ----------
    y : array_like
        Input tensor to integrate.
    x : array_like, optional
        The sample points corresponding to the `y` values. If `x` is None,
        the sample points are assumed to be evenly spaced `dx` apart. The
        default is None.
    dx : scalar, optional
        The spacing between sample points when `x` is None. The default is 1.
    axis : int, optional
        The axis along which to integrate.

    Returns
    -------
    trapezoid : float
        Definite integral as approximated by trapezoidal rule.

    See Also
    --------
    sum, cumsum

    Notes
    -----
    Image [2]_ illustrates trapezoidal rule -- y-axis locations of points
    will be taken from `y` tensor, by default x-axis distances between
    points will be 1.0, alternatively they can be provided with `x` tensor
    or with `dx` scalar.  Return value will be equal to combined area under
    the red lines.


    References
    ----------
    .. [1] Wikipedia page: https://en.wikipedia.org/wiki/Trapezoidal_rule

    .. [2] Illustration image:
           https://en.wikipedia.org/wiki/File:Composite_trapezoidal_rule_illustration.png

    Examples
    --------
    >>> import maxframe.tensor as mt
    >>> mt.trapezoid([1,2,3]).execute()
    4.0
    >>> mt.trapezoid([1,2,3], x=[4,6,8]).execute()
    8.0
    >>> mt.trapezoid([1,2,3], dx=2).execute()
    8.0
    >>> a = mt.arange(6).reshape(2, 3)
    >>> a.execute()
    array([[0, 1, 2],
           [3, 4, 5]])
    >>> mt.trapezoid(a, axis=0).execute()
    array([1.5, 2.5, 3.5])
    >>> mt.trapezoid(a, axis=1).execute()
    array([2.,  8.])

    """
    y = astensor(y)
    axis = validate_axis(y.ndim, axis)
    op = TensorTrapezoid(y=y, x=x, dx=dx, axis=axis)
    return op(y, x=x)
