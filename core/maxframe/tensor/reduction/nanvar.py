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
from ...serialization.serializables import Int32Field
from ..datasource import tensor as astensor
from .core import TensorReduction, TensorReductionMixin


class TensorNanMoment(TensorReduction, TensorReductionMixin):
    _op_type_ = opcodes.NANMOMENT
    _func_name = "nanvar"

    moment = Int32Field("moment", default=2)
    ddof = Int32Field("ddof", default=None)

    def __init__(self, moment=None, **kw):
        if moment is not None:
            kw["moment"] = moment
        super().__init__(**kw)


class TensorNanVar(TensorReduction, TensorReductionMixin):
    _op_type_ = opcodes.NANVAR
    _func_name = "nanvar"

    ddof = Int32Field("ddof", default=0)

    def __new__(cls, *args, **kwargs):
        if kwargs.get("stage") is not None:
            return TensorNanMoment(*args, **kwargs)
        return super().__new__(cls)

    def __init__(self, dtype=None, **kw):
        super().__init__(dtype=dtype, **kw)

    def _get_op_kw(self):
        kw = dict()
        kw["ddof"] = self.ddof
        return kw


def nanvar(a, axis=None, dtype=None, out=None, ddof=0, keepdims=None):
    """
    Compute the variance along the specified axis, while ignoring NaNs.

    Returns the variance of the tensor elements, a measure of the spread of
    a distribution.  The variance is computed for the flattened tensor by
    default, otherwise over the specified axis.

    For all-NaN slices or slices with zero degrees of freedom, NaN is
    returned and a `RuntimeWarning` is raised.

    Parameters
    ----------
    a : array_like
        Tensor containing numbers whose variance is desired.  If `a` is not a
        tensor, a conversion is attempted.
    axis : int, optional
        Axis along which the variance is computed.  The default is to compute
        the variance of the flattened array.
    dtype : data-type, optional
        Type to use in computing the variance.  For tensors of integer type
        the default is `float32`; for tensors of float types it is the same as
        the tensor type.
    out : Tensor, optional
        Alternate output tensor in which to place the result.  It must have
        the same shape as the expected output, but the type is cast if
        necessary.
    ddof : int, optional
        "Delta Degrees of Freedom": the divisor used in the calculation is
        ``N - ddof``, where ``N`` represents the number of non-NaN
        elements. By default `ddof` is zero.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the original `a`.


    Returns
    -------
    variance : Tensor, see dtype parameter above
        If `out` is None, return a new tensor containing the variance,
        otherwise return a reference to the output tensor. If ddof is >= the
        number of non-NaN elements in a slice or the slice contains only
        NaNs, then the result for that slice is NaN.

    See Also
    --------
    std : Standard deviation
    mean : Average
    var : Variance while not ignoring NaNs
    nanstd, nanmean

    Notes
    -----
    The variance is the average of the squared deviations from the mean,
    i.e.,  ``var = mean(abs(x - x.mean())**2)``.

    The mean is normally calculated as ``x.sum() / N``, where ``N = len(x)``.
    If, however, `ddof` is specified, the divisor ``N - ddof`` is used
    instead.  In standard statistical practice, ``ddof=1`` provides an
    unbiased estimator of the variance of a hypothetical infinite
    population.  ``ddof=0`` provides a maximum likelihood estimate of the
    variance for normally distributed variables.

    Note that for complex numbers, the absolute value is taken before
    squaring, so that the result is always real and nonnegative.

    For floating-point input, the variance is computed using the same
    precision the input has.  Depending on the input data, this can cause
    the results to be inaccurate, especially for `float32` (see example
    below).  Specifying a higher-accuracy accumulator using the ``dtype``
    keyword can alleviate this issue.

    For this function to work on sub-classes of Tensor, they must define
    `sum` with the kwarg `keepdims`

    Examples
    --------
    >>> import maxframe.tensor as mt

    >>> a = mt.array([[1, mt.nan], [3, 4]])
    >>> mt.nanvar(a).execute()
    1.5555555555555554
    >>> mt.nanvar(a, axis=0).execute()
    array([ 1.,  0.])
    >>> mt.nanvar(a, axis=1).execute()
    array([ 0.,  0.25])

    """
    a = astensor(a)
    if dtype is None:
        dtype = np.nanvar(np.ones((1,), dtype=a.dtype)).dtype
    op = TensorNanVar(axis=axis, dtype=dtype, keepdims=keepdims, ddof=ddof)
    return op(a, out=out)
