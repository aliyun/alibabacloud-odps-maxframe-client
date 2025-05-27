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

from math import factorial

import numpy as np

from ... import opcodes
from ...serialization.serializables import Int32Field
from ..datasource import tensor as astensor
from .core import TensorReduction, TensorReductionMixin


def reduce_var_square(var_square, avg_diff, count, op, axis, sum_func):
    moment = op.moment
    dtype = op.dtype
    kw = dict(axis=axis, dtype=dtype, keepdims=bool(op.keepdims))

    reduced_var_square = var_square[..., moment - 2].sum(**kw) + sum_func(
        count * avg_diff**moment, **kw
    )
    for i in range(1, moment - 1):
        coeff = factorial(moment) / float(factorial(i) * factorial(moment - i))
        reduced_var_square += coeff * sum_func(
            var_square[..., moment - i - 2] * avg_diff**moment, **kw
        )
    return reduced_var_square


class TensorMoment(TensorReduction, TensorReductionMixin):
    _op_type_ = opcodes.MOMENT
    _func_name = "var"

    moment = Int32Field("moment", default=2)
    ddof = Int32Field("ddof", default=None)

    def __init__(self, moment=None, **kw):
        if moment is not None:
            kw["moment"] = moment
        super().__init__(**kw)


class TensorVar(TensorReduction, TensorReductionMixin):
    _op_type_ = opcodes.VAR
    _func_name = "var"

    ddof = Int32Field("ddof", default=0)

    def __new__(cls, *args, **kwargs):
        if kwargs.get("stage") is not None:
            return TensorMoment(*args, **kwargs)
        return super().__new__(cls)

    def _get_op_kw(self):
        kw = dict()
        kw["ddof"] = self.ddof
        return kw


def var(a, axis=None, dtype=None, out=None, ddof=0, keepdims=None):
    """
    Compute the variance along the specified axis.

    Returns the variance of the tensor elements, a measure of the spread of a
    distribution.  The variance is computed for the flattened tensor by
    default, otherwise over the specified axis.

    Parameters
    ----------
    a : array_like
        Tensor containing numbers whose variance is desired.  If `a` is not a
        tensor, a conversion is attempted.
    axis : None or int or tuple of ints, optional
        Axis or axes along which the variance is computed.  The default is to
        compute the variance of the flattened array.

        If this is a tuple of ints, a variance is performed over multiple axes,
        instead of a single axis or all the axes as before.
    dtype : data-type, optional
        Type to use in computing the variance.  For arrays of integer type
        the default is `float32`; for tensors of float types it is the same as
        the tensor type.
    out : Tensor, optional
        Alternate output array in which to place the result.  It must have
        the same shape as the expected output, but the type is cast if
        necessary.
    ddof : int, optional
        "Delta Degrees of Freedom": the divisor used in the calculation is
        ``N - ddof``, where ``N`` represents the number of elements. By
        default `ddof` is zero.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the input tensor.

        If the default value is passed, then `keepdims` will not be
        passed through to the `var` method of sub-classes of
        `Tensor`, however any non-default value will be.  If the
        sub-classes `sum` method does not implement `keepdims` any
        exceptions will be raised.

    Returns
    -------
    variance : Tensor, see dtype parameter above
        If ``out=None``, returns a new tensor containing the variance;
        otherwise, a reference to the output tensor is returned.

    See Also
    --------
    std , mean, nanmean, nanstd, nanvar

    Notes
    -----
    The variance is the average of the squared deviations from the mean,
    i.e.,  ``var = mean(abs(x - x.mean())**2)``.

    The mean is normally calculated as ``x.sum() / N``, where ``N = len(x)``.
    If, however, `ddof` is specified, the divisor ``N - ddof`` is used
    instead.  In standard statistical practice, ``ddof=1`` provides an
    unbiased estimator of the variance of a hypothetical infinite population.
    ``ddof=0`` provides a maximum likelihood estimate of the variance for
    normally distributed variables.

    Note that for complex numbers, the absolute value is taken before
    squaring, so that the result is always real and nonnegative.

    For floating-point input, the variance is computed using the same
    precision the input has.  Depending on the input data, this can cause
    the results to be inaccurate, especially for `float32` (see example
    below).  Specifying a higher-accuracy accumulator using the ``dtype``
    keyword can alleviate this issue.

    Examples
    --------
    >>> import maxframe.tensor as mt

    >>> a = mt.array([[1, 2], [3, 4]])
    >>> mt.var(a).execute()
    1.25
    >>> mt.var(a, axis=0).execute()
    array([ 1.,  1.])
    >>> mt.var(a, axis=1).execute()
    array([ 0.25,  0.25])

    In single precision, var() can be inaccurate:

    >>> a = mt.zeros((2, 512*512), dtype=mt.float32)
    >>> a[0, :] = 1.0
    >>> a[1, :] = 0.1
    >>> mt.var(a).execute()
    0.20250003

    Computing the variance in float64 is more accurate:

    >>> mt.var(a, dtype=mt.float64).execute()
    0.20249999932944759
    >>> ((1-0.55)**2 + (0.1-0.55)**2)/2
    0.2025

    """
    a = astensor(a)
    if dtype is None:
        dtype = np.var(np.ones((1,), dtype=a.dtype)).dtype
    op = TensorVar(axis=axis, dtype=dtype, keepdims=keepdims, ddof=ddof)
    return op(a, out=out)
