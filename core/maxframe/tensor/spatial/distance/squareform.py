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
from ....serialization.serializables import BoolField, FieldTypes, KeyField, TupleField
from ...core import TensorOrder
from ...datasource import array, ascontiguousarray, zeros
from ...operators import TensorMapReduceOperator, TensorOperatorMixin


class TensorSquareForm(TensorMapReduceOperator, TensorOperatorMixin):
    _op_type_ = opcodes.SQUAREFORM

    checks = BoolField("checks", default=None)

    checks_input = KeyField("checks_input", default=None)
    x_shape = TupleField("x_shape", FieldTypes.int32, default=None)
    reduce_sizes = TupleField("reduce_sizes", FieldTypes.tuple, default=None)
    start_positions = TupleField("start_positions", FieldTypes.int32, default=None)

    @classmethod
    def _set_inputs(cls, op: "TensorSquareForm", inputs: List[EntityData]):
        super()._set_inputs(op, inputs)
        if op.checks_input is not None:
            op._checks_input = inputs[-1]

    def __call__(self, X, force="no", chunk_size=None):
        s = X.shape

        if force.lower() == "tomatrix":
            if len(s) != 1:
                raise ValueError(
                    "Forcing 'tomatrix' but input X is not a distance vector."
                )
        elif force.lower() == "tovector":
            if len(s) != 2:
                raise ValueError(
                    "Forcing 'tovector' but input X is not a distance matrix."
                )

        # X = squareform(v)
        if len(s) == 1:
            if s[0] == 0:
                return zeros((1, 1), dtype=X.dtype)

            # Grab the closest value to the square root of the number
            # of elements times 2 to see if the number of elements
            # is indeed a binomial coefficient.
            d = int(np.ceil(np.sqrt(s[0] * 2)))

            # Check that v is of valid dimensions.
            if d * (d - 1) != s[0] * 2:
                raise ValueError(
                    "Incompatible vector size. It must be a binomial "
                    "coefficient n choose 2 for some integer n >= 2."
                )

            shape = (d, d)
        elif len(s) == 2:
            if s[0] != s[1]:
                raise ValueError("The matrix argument must be square.")

            # One-side of the dimensions is set here.
            d = s[0]

            if d <= 1:
                return array([], dtype=X.dtype)

            shape = ((d * (d - 1)) // 2,)
        else:
            raise ValueError(
                (
                    "The first argument must be one or two dimensional "
                    "tensor. A %d-dimensional tensor is not "
                    "permitted"
                )
                % len(s)
            )

        return self.new_tensor(
            [X], shape=shape, order=TensorOrder.C_ORDER, raw_chunk_size=chunk_size
        )


def squareform(X, force="no", checks=True, chunk_size=None):
    """
    Convert a vector-form distance vector to a square-form distance
    matrix, and vice-versa.

    Parameters
    ----------
    X : Tensor
        Either a condensed or redundant distance matrix.
    force : str, optional
        As with MATLAB(TM), if force is equal to ``'tovector'`` or
        ``'tomatrix'``, the input will be treated as a distance matrix or
        distance vector respectively.
    checks : bool, optional
        If set to False, no checks will be made for matrix
        symmetry nor zero diagonals. This is useful if it is known that
        ``X - X.T1`` is small and ``diag(X)`` is close to zero.
        These values are ignored any way so they do not disrupt the
        squareform transformation.

    Returns
    -------
    Y : Tensor
        If a condensed distance matrix is passed, a redundant one is
        returned, or if a redundant one is passed, a condensed distance
        matrix is returned.

    Notes
    -----
    1. v = squareform(X)

       Given a square d-by-d symmetric distance matrix X,
       ``v = squareform(X)`` returns a ``d * (d-1) / 2`` (or
       :math:`{n \\choose 2}`) sized vector v.

      :math:`v[{n \\choose 2}-{n-i \\choose 2} + (j-i-1)]` is the distance
      between points i and j. If X is non-square or asymmetric, an error
      is returned.

    2. X = squareform(v)

      Given a ``d*(d-1)/2`` sized v for some integer ``d >= 2`` encoding
      distances as described, ``X = squareform(v)`` returns a d by d distance
      matrix X.  The ``X[i, j]`` and ``X[j, i]`` values are set to
      :math:`v[{n \\choose 2}-{n-i \\choose 2} + (j-i-1)]` and all
      diagonal elements are zero.

    """

    X = ascontiguousarray(X)

    op = TensorSquareForm(checks=checks, dtype=X.dtype, gpu=X.op.gpu)
    return op(X, force=force, chunk_size=chunk_size)
