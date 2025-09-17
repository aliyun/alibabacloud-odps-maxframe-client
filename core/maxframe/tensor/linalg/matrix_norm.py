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

from .norm import norm


def matrix_norm(x, *, keepdims=False, ord="fro"):
    """
    Computes the matrix norm of a matrix (or a stack of matrices) ``x``.

    This function is Array API compatible.

    Parameters
    ----------
    x : array_like
        Input array having shape (..., M, N) and whose two innermost
        dimensions form ``MxN`` matrices.
    keepdims : bool, optional
        If this is set to True, the axes which are normed over are left in
        the result as dimensions with size one. Default: False.
    ord : {1, -1, 2, -2, inf, -inf, 'fro', 'nuc'}, optional
        The order of the norm. For details see the table under ``Notes``
        in `numpy.linalg.norm`.

    See Also
    --------
    numpy.linalg.norm : Generic norm function

    Examples
    --------
    >>> import maxframe.tensor as mt
    >>> from maxframe.tensor import linalg as LA
    >>> a = mt.arange(9) - 4
    >>> a.execute()
    array([-4, -3, -2, ...,  2,  3,  4])
    >>> b = a.reshape((3, 3))
    >>> b.execute()
    array([[-4, -3, -2],
           [-1,  0,  1],
           [ 2,  3,  4]])

    >>> LA.matrix_norm(b).execute()
    7.745966692414834
    >>> LA.matrix_norm(b, ord='fro').execute()
    7.745966692414834
    >>> LA.matrix_norm(b, ord=np.inf).execute()
    9.0
    >>> LA.matrix_norm(b, ord=-np.inf).execute()
    2.0

    >>> LA.matrix_norm(b, ord=1).execute()
    7.0
    >>> LA.matrix_norm(b, ord=-1).execute()
    6.0
    >>> LA.matrix_norm(b, ord=2).execute()
    7.3484692283495345
    >>> LA.matrix_norm(b, ord=-2).execute()
    1.8570331885190563e-016 # may vary

    """
    from ..datasource.array import asarray

    x = asarray(x)
    return norm(x, axis=(-2, -1), keepdims=keepdims, ord=ord)
