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
from ...core import ExecutableTuple
from ...serialization.serializables import Int32Field
from ..core import TensorOrder
from ..datasource import tensor as astensor
from ..operators import TensorOperator, TensorOperatorMixin


class TensorGetShape(TensorOperator, TensorOperatorMixin):
    _op_type_ = opcodes.GET_SHAPE

    ndim = Int32Field("ndim")

    def __init__(self, pure_depends=None, **kw):
        super().__init__(_pure_depends=pure_depends, **kw)

    @property
    def output_limit(self):
        return self.ndim

    def __call__(self, a):
        if not np.isnan(a.size):
            return ExecutableTuple([astensor(s) for s in a.shape])

        kws = []
        for i in range(self.output_limit):
            kws.append(
                {
                    "shape": (),
                    "dtype": np.dtype(np.intc),
                    "order": TensorOrder.C_ORDER,
                    "i": i,
                }
            )
        return ExecutableTuple(self.new_tensors([a], kws=kws))


def shape(a):
    """
    Return the shape of a tensor.

    Parameters
    ----------
    a : array_like
        Input tensor.

    Returns
    -------
    shape : ExecutableTuple of tensors
        The elements of the shape tuple give the lengths of the
        corresponding array dimensions.

    Examples
    --------
    >>> import maxframe.tensor as mt

    >>> mt.shape(mt.eye(3)).execute()
    (3, 3)
    >>> mt.shape([[1, 2]]).execute()
    (1, 2)
    >>> mt.shape([0]).execute()
    (1,)
    >>> mt.shape(0).execute()
    ()

    >>> a = mt.array([(1, 2), (3, 4)], dtype=[('x', 'i4'), ('y', 'i4')])
    >>> mt.shape(a).execute()
    (2,)

    """
    a = astensor(a)
    op = TensorGetShape(ndim=a.ndim)
    return op(a)
