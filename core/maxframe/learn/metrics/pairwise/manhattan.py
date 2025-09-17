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

from .... import opcodes
from ....core import EntityData
from ....serialization.serializables import KeyField
from ....tensor.core import TensorOrder
from .core import PairwiseDistances


class ManhattanDistances(PairwiseDistances):
    _op_type_ = opcodes.PAIRWISE_MANHATTAN_DISTANCES

    x = KeyField("x")
    y = KeyField("y")

    @classmethod
    def _set_inputs(cls, op: "ManhattanDistances", inputs: List[EntityData]):
        super()._set_inputs(op, inputs)
        op.x, op.y = inputs[:2]

    def __call__(self, X, Y=None):
        X, Y = self.check_pairwise_arrays(X, Y)
        if self.y is None:
            self.y = Y
        shape = (X.shape[0], Y.shape[0])

        return self.new_tensor([X, Y], shape=shape, order=TensorOrder.C_ORDER)


def manhattan_distances(X, Y=None):
    """ Compute the L1 distances between the vectors in X and Y.

    Read more in the :ref:`User Guide <metrics>`.

    Parameters
    ----------
    X : array_like
        A tensor with shape (n_samples_X, n_features).

    Y : array_like, optional
        A tensor with shape (n_samples_Y, n_features).

    Returns
    -------
    D : Tensor
        Shape is (n_samples_X, n_samples_Y) and D contains
        the pairwise L1 distances.

    Examples
    --------
    >>> from maxframe.learn.metrics.pairwise import manhattan_distances
    >>> manhattan_distances([[3]], [[3]]).execute() #doctest:+ELLIPSIS
    array([[0.]])
    >>> manhattan_distances([[3]], [[2]]).execute() #doctest:+ELLIPSIS
    array([[1.]])
    >>> manhattan_distances([[2]], [[3]]).execute() #doctest:+ELLIPSIS
    array([[1.]])
    >>> manhattan_distances([[1, 2], [3, 4]],\
         [[1, 2], [0, 3]]).execute() #doctest:+ELLIPSIS
    array([[0., 2.],
           [4., 4.]])
    """
    op = ManhattanDistances(x=X, y=Y, dtype=np.dtype(np.float64))
    return op(X, Y=Y)
