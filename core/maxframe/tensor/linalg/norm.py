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

from collections.abc import Iterable

import numpy as np

from ... import opcodes
from ...serialization.serializables import AnyField, BoolField, FieldTypes, TupleField
from ..arithmetic import sqrt
from ..datasource import tensor as astensor
from ..operators import TensorHasInput, TensorOperatorMixin
from ..utils import validate_axis
from .svd import svd


class TensorNorm(TensorHasInput, TensorOperatorMixin):
    _op_type_ = opcodes.NORM

    ord = AnyField("ord", default=None)
    axis = TupleField("axis", FieldTypes.int32, default=0)
    keepdims = BoolField("keepdims", default=None)

    def __call__(self, x):
        r = x.astype(self.dtype)
        shape = self._norm(r, self.ord, self.axis, self.keepdims).shape
        return self.new_tensor([x], shape)

    @staticmethod
    def _norm(r, ord, axis, keepdims):
        if ord is None:
            return sqrt((abs(r) ** 2).sum(axis=axis, keepdims=keepdims))
        elif ord == "nuc":
            if len(axis) == 1:
                raise ValueError("Invalid norm order for vectors.")
            return svd(r)[1][np.newaxis].sum(keepdims=keepdims)
        elif ord == np.inf:
            if r.ndim > 2:
                raise ValueError("Improper number of dimensions to norm.")
            r = abs(r)
            if len(axis) == 1:
                return r.max(axis=axis, keepdims=keepdims)
            else:
                return r.sum(axis=axis[1], keepdims=keepdims).max(keepdims=keepdims)
        elif ord == -np.inf:
            if r.ndim > 2:
                raise ValueError("Improper number of dimensions to norm.")
            r = abs(r)
            if len(axis) == 1:
                return r.min(axis=axis, keepdims=keepdims)
            else:
                return r.sum(axis=axis[1], keepdims=keepdims).min(keepdims=keepdims)
        elif ord == 0:
            if r.ndim > 2:
                raise ValueError("Improper number of dimensions to norm.")
            if len(axis) == 2:
                raise ValueError("Invalid norm order for matrices.")
            return (r != 0).astype(r.dtype).sum(axis=axis, keepdims=keepdims)
        elif ord == 1:
            if r.ndim > 2:
                raise ValueError("Improper number of dimensions to norm.")
            r = abs(r)
            if len(axis) == 1:
                return r.sum(axis=axis, keepdims=keepdims)
            else:
                return r.sum(axis=axis[0], keepdims=keepdims).max(keepdims=keepdims)
        elif ord == -1 and len(axis) == 2:
            if r.ndim > 2:
                raise ValueError("Improper number of dimensions to norm.")
            return abs(r).sum(axis=axis[0], keepdims=keepdims).min(keepdims=keepdims)
        elif ord == 2 and len(axis) == 2:
            return svd(r)[1][np.newaxis].max(keepdims=keepdims)
        elif ord == -2 and len(axis) == 2:
            return svd(r)[1][np.newaxis].min(keepdims=keepdims)
        else:
            if len(axis) == 2:
                raise ValueError("Invalid norm order for matrices.")

            return (abs(r) ** ord).sum(axis=axis, keepdims=keepdims) ** (1.0 / ord)


def norm(x, ord=None, axis=None, keepdims=False):
    r"""
    Matrix or vector norm.

    This function is able to return one of eight different matrix norms,
    or one of an infinite number of vector norms (described below), depending
    on the value of the ``ord`` parameter.

    Parameters
    ----------
    x : array_like
        Input tensor.  If `axis` is None, `x` must be 1-D or 2-D.
    ord : {non-zero int, inf, -inf, 'fro', 'nuc'}, optional
        Order of the norm (see table under ``Notes``). inf means maxframe tensor's
        `inf` object.
    axis : {int, 2-tuple of ints, None}, optional
        If `axis` is an integer, it specifies the axis of `x` along which to
        compute the vector norms.  If `axis` is a 2-tuple, it specifies the
        axes that hold 2-D matrices, and the matrix norms of these matrices
        are computed.  If `axis` is None then either a vector norm (when `x`
        is 1-D) or a matrix norm (when `x` is 2-D) is returned.
    keepdims : bool, optional
        If this is set to True, the axes which are normed over are left in the
        result as dimensions with size one.  With this option the result will
        broadcast correctly against the original `x`.

    Returns
    -------
    n : float or Tensor
        Norm of the matrix or vector(s).

    Notes
    -----
    For values of ``ord <= 0``, the result is, strictly speaking, not a
    mathematical 'norm', but it may still be useful for various numerical
    purposes.

    The following norms can be calculated:

    =====  ============================  ==========================
    ord    norm for matrices             norm for vectors
    =====  ============================  ==========================
    None   Frobenius norm                2-norm
    'fro'  Frobenius norm                --
    'nuc'  nuclear norm                  --
    inf    max(sum(abs(x), axis=1))      max(abs(x))
    -inf   min(sum(abs(x), axis=1))      min(abs(x))
    0      --                            sum(x != 0)
    1      max(sum(abs(x), axis=0))      as below
    -1     min(sum(abs(x), axis=0))      as below
    2      2-norm (largest sing. value)  as below
    -2     smallest singular value       as below
    other  --                            sum(abs(x)**ord)**(1./ord)
    =====  ============================  ==========================

    The Frobenius norm is given by [1]_:

        :math:`||A||_F = [\\sum_{i,j} abs(a_{i,j})^2]^{1/2}`

    The nuclear norm is the sum of the singular values.

    References
    ----------
    .. [1] G. H. Golub and C. F. Van Loan, *Matrix Computations*,
           Baltimore, MD, Johns Hopkins University Press, 1985, pg. 15

    Examples
    --------
    >>> from maxframe.tensor import linalg as LA
    >>> import maxframe.tensor as mt
    >>> a = mt.arange(9) - 4
    >>> a.execute()
    array([-4, -3, -2, -1,  0,  1,  2,  3,  4])
    >>> b = a.reshape((3, 3))
    >>> b.execute()
    array([[-4, -3, -2],
           [-1,  0,  1],
           [ 2,  3,  4]])

    >>> LA.norm(a).execute()
    7.745966692414834
    >>> LA.norm(b).execute()
    7.745966692414834
    >>> LA.norm(b, 'fro').execute()
    7.745966692414834
    >>> LA.norm(a, mt.inf).execute()
    4.0
    >>> LA.norm(b, mt.inf).execute()
    9.0
    >>> LA.norm(a, -mt.inf).execute()
    0.0
    >>> LA.norm(b, -mt.inf).execute()
    2.0

    >>> LA.norm(a, 1).execute()
    20.0
    >>> LA.norm(b, 1).execute()
    7.0
    >>> LA.norm(a, -1).execute()
    0.0
    >>> LA.norm(b, -1).execute()
    6.0
    >>> LA.norm(a, 2).execute()
    7.745966692414834
    >>> LA.norm(b, 2).execute()
    7.3484692283495345

    >>> LA.norm(a, -2).execute()
    0.0
    >>> LA.norm(b, -2).execute()
    4.351066026358965e-18
    >>> LA.norm(a, 3).execute()
    5.8480354764257312
    >>> LA.norm(a, -3).execute()
    0.0

    Using the `axis` argument to compute vector norms:

    >>> c = mt.array([[ 1, 2, 3],
    ...               [-1, 1, 4]])
    >>> LA.norm(c, axis=0).execute()
    array([ 1.41421356,  2.23606798,  5.        ])
    >>> LA.norm(c, axis=1).execute()
    array([ 3.74165739,  4.24264069])
    >>> LA.norm(c, ord=1, axis=1).execute()
    array([ 6.,  6.])

    Using the `axis` argument to compute matrix norms:

    >>> m = mt.arange(8).reshape(2,2,2)
    >>> LA.norm(m, axis=(1,2)).execute()
    array([  3.74165739,  11.22497216])
    >>> LA.norm(m[0, :, :]).execute(), LA.norm(m[1, :, :]).execute()
    (3.7416573867739413, 11.224972160321824)

    """
    x = astensor(x)
    ndim = x.ndim

    if ord == "fro":
        ord = None
    if axis is not None:
        if isinstance(axis, Iterable):
            axis = tuple(validate_axis(ndim, a) for a in axis)
        else:
            axis = (validate_axis(ndim, axis),)
    else:
        axis = tuple(range(x.ndim))

    op = TensorNorm(
        ord=ord,
        axis=axis,
        keepdims=keepdims,
        dtype=np.result_type(x.dtype, np.float_),
        sparse=x.issparse(),
    )
    return op(x)
