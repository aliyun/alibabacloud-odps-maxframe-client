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
from ....serialization.serializables import BoolField, KeyField
from ....tensor.core import TensorOrder
from .core import PairwiseDistances


class HaversineDistances(PairwiseDistances):
    _op_type_ = opcodes.PAIRWISE_HAVERSINE_DISTANCES

    x = KeyField("x")
    y = KeyField("y")
    # for test purpose
    use_sklearn = BoolField("use_sklearn", default=True)

    @classmethod
    def _set_inputs(cls, op: "HaversineDistances", inputs: List[EntityData]):
        super()._set_inputs(op, inputs)
        op.x, op.y = op._inputs[:2]

    def __call__(self, X, Y=None):
        X, Y = self.check_pairwise_arrays(X, Y)
        if self.y is None:
            self.y = Y

        if X.shape[1] != 2 or Y.shape[1] != 2:
            raise ValueError("Haversine distance only valid in 2 dimensions")
        if X.issparse() or Y.issparse():
            raise TypeError("Haversine distance requires inputs dense")

        return self.new_tensor(
            [X, Y], shape=(X.shape[0], Y.shape[0]), order=TensorOrder.C_ORDER
        )


def haversine_distances(X, Y=None):
    """Compute the Haversine distance between samples in X and Y

    The Haversine (or great circle) distance is the angular distance between
    two points on the surface of a sphere. The first distance of each point is
    assumed to be the latitude, the second is the longitude, given in radians.
    The dimension of the data must be 2.

    .. math::
       D(x, y) = 2\\arcsin[\\sqrt{\\sin^2((x1 - y1) / 2)
                                + \\cos(x1)\\cos(y1)\\sin^2((x2 - y2) / 2)}]

    Parameters
    ----------
    X : array_like, shape (n_samples_1, 2)

    Y : array_like, shape (n_samples_2, 2), optional

    Returns
    -------
    distance : {Tensor}, shape (n_samples_1, n_samples_2)

    Notes
    -----
    As the Earth is nearly spherical, the haversine formula provides a good
    approximation of the distance between two points of the Earth surface, with
    a less than 1% error on average.

    Examples
    --------
    We want to calculate the distance between the Ezeiza Airport
    (Buenos Aires, Argentina) and the Charles de Gaulle Airport (Paris, France)

    >>> from maxframe.learn.metrics.pairwise import haversine_distances
    >>> bsas = [-34.83333, -58.5166646]
    >>> paris = [49.0083899664, 2.53844117956]
    >>> result = haversine_distances([bsas, paris])
    >>> (result * 6371000/1000).execute()  # multiply by Earth radius to get kilometers
    array([[    0.        , 11279.45379464],
           [11279.45379464,     0.        ]])
    """
    op = HaversineDistances(x=X, y=Y, dtype=np.dtype(np.float64))
    return op(X, Y=Y)
