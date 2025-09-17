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

from ..utils import normalize_axis_tuple


def vector_norm(x, *, axis=None, keepdims=False, ord=2):
    """
    Computes the vector norm of a vector (or batch of vectors) ``x``.

    This function is Array API compatible.

    Parameters
    ----------
    x : array_like
        Input array.
    axis : {None, int, 2-tuple of ints}, optional
        If an integer, ``axis`` specifies the axis (dimension) along which
        to compute vector norms. If an n-tuple, ``axis`` specifies the axes
        (dimensions) along which to compute batched vector norms. If ``None``,
        the vector norm must be computed over all array values (i.e.,
        equivalent to computing the vector norm of a flattened array).
        Default: ``None``.
    keepdims : bool, optional
        If this is set to True, the axes which are normed over are left in
        the result as dimensions with size one. Default: False.
    ord : {int, float, inf, -inf}, optional
        The order of the norm. For details see the table under ``Notes``
        in `numpy.linalg.norm`.

    See Also
    --------
    numpy.linalg.norm : Generic norm function

    Examples
    --------
    >>> from numpy import linalg as LA
    >>> a = np.arange(9) + 1
    >>> a
    array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> b = a.reshape((3, 3))
    >>> b
    array([[1, 2, 3],
           [4, 5, 6],
           [7, 8, 9]])

    >>> LA.vector_norm(b)
    16.881943016134134
    >>> LA.vector_norm(b, ord=np.inf)
    9.0
    >>> LA.vector_norm(b, ord=-np.inf)
    1.0

    >>> LA.vector_norm(b, ord=0)
    9.0
    >>> LA.vector_norm(b, ord=1)
    45.0
    >>> LA.vector_norm(b, ord=-1)
    0.3534857623790153
    >>> LA.vector_norm(b, ord=2)
    16.881943016134134
    >>> LA.vector_norm(b, ord=-2)
    0.8058837395885292

    """
    from ..datasource.array import asarray
    from ..misc import transpose
    from ..reduction import prod
    from .norm import norm

    x = asarray(x)
    shape = list(x.shape)
    if axis is None:
        # Note: np.linalg.norm() doesn't handle 0-D arrays
        x = x.ravel()
        _axis = 0
    elif isinstance(axis, tuple):
        # Note: The axis argument supports any number of axes, whereas
        # np.linalg.norm() only supports a single axis for vector norm.
        normalized_axis = normalize_axis_tuple(axis, x.ndim)
        rest = tuple(i for i in range(x.ndim) if i not in normalized_axis)
        newshape = axis + rest
        x = transpose(x, newshape).reshape(
            (prod([x.shape[i] for i in axis], dtype=int), *[x.shape[i] for i in rest])
        )
        _axis = 0
    else:
        _axis = axis

    res = norm(x, axis=_axis, ord=ord)

    if keepdims:
        # We can't reuse np.linalg.norm(keepdims) because of the reshape hacks
        # above to avoid matrix norm logic.
        _axis = normalize_axis_tuple(
            range(len(shape)) if axis is None else axis, len(shape)
        )
        for i in _axis:
            shape[i] = 1
        res = res.reshape(tuple(shape))

    return res
