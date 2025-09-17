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
from .... import tensor as mt
from ....config import options
from ....core import EntityData
from ....serialization.serializables import BoolField, KeyField
from ....tensor.core import TensorOrder
from ...utils.validation import check_array
from .core import PairwiseDistances


class EuclideanDistances(PairwiseDistances):
    _op_type_ = opcodes.PAIRWISE_EUCLIDEAN_DISTANCES

    x = KeyField("X")
    y = KeyField("Y", default=None)
    x_norm_squared = KeyField("X_norm_squared", default=None)
    y_norm_squared = KeyField("Y_norm_squared", default=None)
    squared = BoolField("squared", default=None)

    @classmethod
    def _set_inputs(cls, op: "EuclideanDistances", inputs: List[EntityData]):
        super()._set_inputs(op, inputs)
        input_iter = iter(inputs)
        op.x = next(input_iter)
        if op.y is not None:
            op.y = next(input_iter)
        if op.x_norm_squared is not None:
            op.x_norm_squared = next(input_iter)
        if op.y_norm_squared is not None:
            op.y_norm_squared = next(input_iter)

    def __call__(self, X, Y=None, Y_norm_squared=None, X_norm_squared=None):
        # If norms are passed as float32, they are unused. If arrays are passed as
        # float32, norms needs to be recomputed on upcast chunks.
        # TODO: use a float64 accumulator in row_norms to avoid the latter.
        if X_norm_squared is not None:
            XX = check_array(X_norm_squared)
            if XX.shape == (1, X.shape[0]):
                XX = XX.T
            elif XX.shape != (X.shape[0], 1):
                raise ValueError("Incompatible dimensions for X and X_norm_squared")
            if XX.dtype == np.float32:
                XX = self.x_norm_squared = None
        else:
            XX = None

        if X is Y and XX is not None:
            # shortcut in the common case euclidean_distances(X, X)
            YY = XX.T
        elif Y_norm_squared is not None:
            YY = mt.atleast_2d(Y_norm_squared)

            if YY.shape != (1, Y.shape[0]):
                raise ValueError("Incompatible dimensions for Y and Y_norm_squared")
            if YY.dtype == np.float32:
                YY = self.y_norm_squared = None
        else:
            YY = None

        inputs = [X, Y]
        if XX is not None:
            inputs.append(XX)
        if YY is not None:
            inputs.append(YY)
        return self.new_tensor(
            inputs, shape=(X.shape[0], Y.shape[0]), order=TensorOrder.C_ORDER
        )


def euclidean_distances(
    X, Y=None, Y_norm_squared=None, squared=False, X_norm_squared=None
):
    """
    Considering the rows of X (and Y=X) as vectors, compute the
    distance matrix between each pair of vectors.

    For efficiency reasons, the euclidean distance between a pair of row
    vector x and y is computed as::

        dist(x, y) = sqrt(dot(x, x) - 2 * dot(x, y) + dot(y, y))

    This formulation has two advantages over other ways of computing distances.
    First, it is computationally efficient when dealing with sparse data.
    Second, if one argument varies but the other remains unchanged, then
    `dot(x, x)` and/or `dot(y, y)` can be pre-computed.

    However, this is not the most precise way of doing this computation, and
    the distance matrix returned by this function may not be exactly
    symmetric as required by, e.g., ``scipy.spatial.distance`` functions.

    Read more in the :ref:`User Guide <metrics>`.

    Parameters
    ----------
    X : {array-like, sparse matrix}, shape (n_samples_1, n_features)

    Y : {array-like, sparse matrix}, shape (n_samples_2, n_features)

    Y_norm_squared : array-like, shape (n_samples_2, ), optional
        Pre-computed dot-products of vectors in Y (e.g.,
        ``(Y**2).sum(axis=1)``)
        May be ignored in some cases, see the note below.

    squared : boolean, optional
        Return squared Euclidean distances.

    X_norm_squared : array-like, shape = [n_samples_1], optional
        Pre-computed dot-products of vectors in X (e.g.,
        ``(X**2).sum(axis=1)``)
        May be ignored in some cases, see the note below.

    Notes
    -----
    To achieve better accuracy, `X_norm_squared` and `Y_norm_squared` may be
    unused if they are passed as ``float32``.

    Returns
    -------
    distances : tensor, shape (n_samples_1, n_samples_2)

    Examples
    --------
    >>> from maxframe.learn.metrics.pairwise import euclidean_distances
    >>> X = [[0, 1], [1, 1]]
    >>> # distance between rows of X
    >>> euclidean_distances(X, X).execute()
    array([[0., 1.],
           [1., 0.]])
    >>> # get distance to origin
    >>> euclidean_distances(X, [[0, 0]]).execute()
    array([[1.        ],
           [1.41421356]])

    See also
    --------
    paired_distances : distances betweens pairs of elements of X and Y.
    """
    if X.dtype == np.float32:
        if Y is None:
            dtype = X.dtype
        elif Y.dtype == np.float32:
            dtype = np.float32
        else:
            dtype = np.float64
    else:
        dtype = np.float64

    X, Y = EuclideanDistances.check_pairwise_arrays(X, Y)
    op = EuclideanDistances(
        x=X,
        y=Y,
        x_norm_squared=X_norm_squared,
        y_norm_squared=Y_norm_squared,
        squared=squared,
        dtype=np.dtype(dtype),
        chunk_store_limit=options.chunk_store_limit,
    )
    return op(X, Y=Y, Y_norm_squared=Y_norm_squared, X_norm_squared=X_norm_squared)
