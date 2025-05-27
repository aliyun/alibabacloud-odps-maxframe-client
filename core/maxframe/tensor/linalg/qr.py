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
from numpy.linalg import LinAlgError

from ... import opcodes
from ...core import ExecutableTuple
from ...serialization.serializables import StringField
from ..core import TensorOrder
from ..datasource import tensor as astensor
from ..operators import TensorHasInput, TensorOperatorMixin


class TensorQR(TensorHasInput, TensorOperatorMixin):
    _op_type_ = opcodes.QR

    method = StringField("method")

    @property
    def output_limit(self):
        return 2

    def __call__(self, a):
        a = astensor(a)

        if a.ndim != 2:
            raise LinAlgError(
                f"{a.ndim}-dimensional tensor given. Tensor must be two-dimensional"
            )

        tiny_q, tiny_r = np.linalg.qr(np.ones((1, 1), dtype=a.dtype))

        x, y = a.shape
        q_shape, r_shape = (a.shape, (y, y)) if x > y else ((x, x), a.shape)
        q, r = self.new_tensors(
            [a],
            kws=[
                {
                    "side": "q",
                    "dtype": tiny_q.dtype,
                    "shape": q_shape,
                    "order": TensorOrder.C_ORDER,
                },
                {
                    "side": "r",
                    "dtype": tiny_r.dtype,
                    "shape": r_shape,
                    "order": TensorOrder.C_ORDER,
                },
            ],
        )
        return ExecutableTuple([q, r])


def qr(a, method="tsqr"):
    """
    Compute the qr factorization of a matrix.

    Factor the matrix `a` as *qr*, where `q` is orthonormal and `r` is
    upper-triangular.

    Parameters
    ----------
    a : array_like, shape (M, N)
        Matrix to be factored.
    method: {'tsqr', 'sfqr'}, optional
        method to calculate qr factorization, tsqr as default

        TSQR is presented in:

            A. Benson, D. Gleich, and J. Demmel.
            Direct QR factorizations for tall-and-skinny matrices in
            MapReduce architectures.
            IEEE International Conference on Big Data, 2013.
            http://arxiv.org/abs/1301.1071

        FSQR is a QR decomposition for fat and short matrix:
            A = [A1, A2, A3, ...], A1 may be decomposed as A1 = Q1 * R1,
            for A = Q * R, Q = Q1, R = [R1, R2, R3, ...] where A2 = Q1 * R2, A3 = Q1 * R3, ...

    Returns
    -------
    q : Tensor of float or complex, optional
        A matrix with orthonormal columns. When mode = 'complete' the
        result is an orthogonal/unitary matrix depending on whether or not
        a is real/complex. The determinant may be either +/- 1 in that
        case.
    r : Tensor of float or complex, optional
        The upper-triangular matrix.

    Raises
    ------
    LinAlgError
        If factoring fails.

    Notes
    -----
    For more information on the qr factorization, see for example:
    http://en.wikipedia.org/wiki/QR_factorization

    Examples
    --------
    >>> import maxframe.tensor as mt

    >>> a = mt.random.randn(9, 6)
    >>> q, r = mt.linalg.qr(a)
    >>> mt.allclose(a, mt.dot(q, r)).execute()  # a does equal qr
    True

    """
    op = TensorQR(method=method)
    return op(a)
