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

from ..utils import implement_scipy, infer_scipy_dtype
from .core import TensorSpecialBinOp, _register_special_op


@_register_special_op
class TensorXLogY(TensorSpecialBinOp):
    _func_name = "xlogy"

    @classmethod
    def _is_sparse(cls, x1, x2):
        if hasattr(x1, "issparse") and x1.issparse():
            return True
        return False


@implement_scipy("scipy.special.xlogy")
@infer_scipy_dtype("scipy.special.xlogy")
def xlogy(x1, x2, out=None, where=None, **kwargs):
    op = TensorXLogY(**kwargs)
    return op(x1, x2, out=out, where=where)


def softmax(x, axis=None):
    r"""
    Compute the softmax function.
    The softmax function transforms each element of a collection by
    computing the exponential of each element divided by the sum of the
    exponentials of all the elements. That is, if `x` is a one-dimensional
    numpy array::

        softmax(x) = np.exp(x)/sum(np.exp(x))

    Parameters
    ----------
    x : array_like
        Input array.
    axis : int or tuple of ints, optional
        Axis to compute values along. Default is None and softmax will be
        computed over the entire array `x`.

    Returns
    -------
    s : ndarray
        An array the same shape as `x`. The result will sum to 1 along the
        specified axis.

    Notes
    -----
    The formula for the softmax function :math:`\sigma(x)` for a vector
    :math:`x = \{x_0, x_1, ..., x_{n-1}\}` is

    .. math:: \sigma(x)_j = \frac{e^{x_j}}{\sum_k e^{x_k}}

    The `softmax` function is the gradient of `logsumexp`.

    The implementation uses shifting to avoid overflow. See [1]_ for more
    details.

    References
    ----------

    .. [1] P. Blanchard, D.J. Higham, N.J. Higham, "Accurately computing the
       log-sum-exp and softmax functions", IMA Journal of Numerical Analysis,
       Vol.41(4), :doi:`10.1093/imanum/draa038`.

    Examples
    --------
    >>> import maxframe.tensor as mt
    >>> from maxframe.tensor.special import softmax

    >>> x = mt.array([[1, 0.5, 0.2, 3],
    ...               [1,  -1,   7, 3],
    ...               [2,  12,  13, 3]])
    ...

    Compute the softmax transformation over the entire array.

    >>> m = softmax(x)
    >>> m.to_numpy()
    array([[  4.48309e-06,   2.71913e-06,   2.01438e-06,   3.31258e-05],
           [  4.48309e-06,   6.06720e-07,   1.80861e-03,   3.31258e-05],
           [  1.21863e-05,   2.68421e-01,   7.29644e-01,   3.31258e-05]])

    >>> m.sum().to_numpy()
    1.0

    Compute the softmax transformation along the first axis (i.e., the
    columns).

    >>> m = softmax(x, axis=0)
    >>> m.to_numpy()
    array([[  2.11942e-01,   1.01300e-05,   2.75394e-06,   3.33333e-01],
           [  2.11942e-01,   2.26030e-06,   2.47262e-03,   3.33333e-01],
           [  5.76117e-01,   9.99988e-01,   9.97525e-01,   3.33333e-01]])
    >>> m.sum(axis=0).to_numpy()
    array([ 1.,  1.,  1.,  1.])

    Compute the softmax transformation along the second axis (i.e., the rows).

    >>> m = softmax(x, axis=1)
    >>> m.to_numpy()
    array([[  1.05877e-01,   6.42177e-02,   4.75736e-02,   7.82332e-01],
           [  2.42746e-03,   3.28521e-04,   9.79307e-01,   1.79366e-02],
           [  1.22094e-05,   2.68929e-01,   7.31025e-01,   3.31885e-05]])
    >>> m.sum(axis=1).to_numpy()
    array([ 1.,  1.,  1.])
    """
    from ... import tensor as mt

    x = mt.tensor(x)
    x_max = mt.amax(x, axis=axis, keepdims=True)
    exp_x_shifted = mt.exp(x - x_max)
    return exp_x_shifted / mt.sum(exp_x_shifted, axis=axis, keepdims=True)


def softplus(x, **kwargs):
    r"""
    Compute the softplus function element-wise.

    The softplus function is defined as: ``softplus(x) = log(1 + exp(x))``.
    It is a smooth approximation of the rectifier function (ReLU).

    Parameters
    ----------
    x : array_like
        Input value.
    **kwargs
        For other keyword-only arguments, see the
        `ufunc docs <https://numpy.org/doc/stable/reference/ufuncs.html>`_.

    Returns
    -------
    softplus : ndarray
        Logarithm of ``exp(0) + exp(x)``.

    Examples
    --------
    >>> from maxframe.tensor import special

    >>> special.softplus(0).to_numpy()
    0.6931471805599453

    >>> special.softplus([-1, 0, 1]).to_numpy()
    array([0.31326169, 0.69314718, 1.31326169])
    """
    # use numpy implementation as softplus is introduced in scipy 1.15.
    from ... import tensor as mt

    return mt.logaddexp(0, x, **kwargs)
