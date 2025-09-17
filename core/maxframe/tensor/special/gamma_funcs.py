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

from ..arithmetic.utils import arithmetic_operator
from ..utils import implement_scipy, infer_scipy_dtype
from .core import (
    TensorSpecialBinOp,
    TensorSpecialMultiOp,
    TensorSpecialUnaryOp,
    _register_special_op,
)


class NoOrderSpecialMixin:
    @classmethod
    def _get_func(cls, xp):
        func = super()._get_func(xp)

        def _wrapped(*args, **kw):
            kw.pop("order", None)
            return func(*args, **kw)

        return _wrapped


@_register_special_op
@arithmetic_operator(sparse_mode="unary")
class TensorGamma(TensorSpecialUnaryOp):
    _func_name = "gamma"


@implement_scipy("scipy.special.gamma")
@infer_scipy_dtype("scipy.special.gamma")
def gamma(x, **kwargs):
    op = TensorGamma(**kwargs)
    return op(x)


@_register_special_op
@arithmetic_operator(sparse_mode="unary")
class TensorGammaln(TensorSpecialUnaryOp):
    _func_name = "gammaln"


@implement_scipy("scipy.special.gammaln")
@infer_scipy_dtype("scipy.special.gammaln")
def gammaln(x, out=None, where=None, **kwargs):
    """
    Logarithm of the absolute value of the Gamma function.

    Parameters
    ----------
    x : array-like
        Values on the real line at which to compute ``gammaln``
    out : Tensor, None, or tuple of Tensor and None, optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or `None`,
        a freshly-allocated tensor is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : array_like, optional
        Values of True indicate to calculate the ufunc at that position, values
        of False indicate to leave the value in the output alone.
    **kwargs

    Returns
    -------
    gammaln : Tensor
        Values of ``gammaln`` at x.

    See Also
    --------
    gammasgn : sign of the gamma function
    loggamma : principal branch of the logarithm of the gamma function

    Notes
    -----
    When used in conjunction with `gammasgn`, this function is useful
    for working in logspace on the real axis without having to deal with
    complex numbers, via the relation ``exp(gammaln(x)) = gammasgn(x)*gamma(x)``.

    For complex-valued log-gamma, use `loggamma` instead of `gammaln`.
    """
    op = TensorGammaln(**kwargs)
    return op(x, out=out, where=where)


@_register_special_op
@arithmetic_operator(sparse_mode="unary")
class TensorLogGamma(TensorSpecialUnaryOp):
    _func_name = "loggamma"


@implement_scipy("scipy.special.loggamma")
@infer_scipy_dtype("scipy.special.loggamma")
def loggamma(x, **kwargs):
    op = TensorLogGamma(**kwargs)
    return op(x)


@_register_special_op
@arithmetic_operator(sparse_mode="unary")
class TensorGammaSgn(TensorSpecialUnaryOp):
    _func_name = "gammasgn"


@implement_scipy("scipy.special.gammasgn")
@infer_scipy_dtype("scipy.special.gammasgn")
def gammasgn(x, **kwargs):
    op = TensorGammaSgn(**kwargs)
    return op(x)


@_register_special_op
@arithmetic_operator(sparse_mode="binary_and")
class TensorGammaInc(TensorSpecialBinOp):
    _func_name = "gammainc"


@implement_scipy("scipy.special.gammainc")
@infer_scipy_dtype("scipy.special.gammainc")
def gammainc(a, b, **kwargs):
    op = TensorGammaInc(**kwargs)
    return op(a, b)


@_register_special_op
@arithmetic_operator(sparse_mode="binary_and")
class TensorGammaIncInv(TensorSpecialBinOp):
    _func_name = "gammaincinv"


@implement_scipy("scipy.special.gammaincinv")
@infer_scipy_dtype("scipy.special.gammaincinv")
def gammaincinv(a, b, **kwargs):
    op = TensorGammaIncInv(**kwargs)
    return op(a, b)


@_register_special_op
@arithmetic_operator(sparse_mode="binary_and")
class TensorGammaIncc(TensorSpecialBinOp):
    _func_name = "gammaincc"


@implement_scipy("scipy.special.gammainc")
@infer_scipy_dtype("scipy.special.gammainc")
def gammaincc(a, b, **kwargs):
    op = TensorGammaIncc(**kwargs)
    return op(a, b)


@_register_special_op
@arithmetic_operator(sparse_mode="binary_and")
class TensorGammaInccInv(TensorSpecialBinOp):
    _func_name = "gammainccinv"


@implement_scipy("scipy.special.gammainccinv")
@infer_scipy_dtype("scipy.special.gammainccinv")
def gammainccinv(a, b, **kwargs):
    op = TensorGammaInccInv(**kwargs)
    return op(a, b)


@_register_special_op
@arithmetic_operator(sparse_mode="binary_and")
class TensorBeta(TensorSpecialBinOp):
    _func_name = "beta"


@implement_scipy("scipy.special.beta")
@infer_scipy_dtype("scipy.special.beta")
def beta(a, b, out=None, **kwargs):
    op = TensorBeta(**kwargs)
    return op(a, b, out=out)


@_register_special_op
@arithmetic_operator(sparse_mode="binary_and")
class TensorBetaLn(TensorSpecialBinOp):
    _func_name = "betaln"


@implement_scipy("scipy.special.betaln")
@infer_scipy_dtype("scipy.special.betaln")
def betaln(a, b, out=None, **kwargs):
    op = TensorBetaLn(**kwargs)
    return op(a, b, out=out)


@_register_special_op
class TensorBetaInc(TensorSpecialMultiOp):
    _ARG_COUNT = 3
    _func_name = "betainc"


@implement_scipy("scipy.special.betainc")
@infer_scipy_dtype("scipy.special.betainc")
def betainc(a, b, x, out=None, **kwargs):
    op = TensorBetaInc(**kwargs)
    return op(a, b, x, out=out)


@_register_special_op
class TensorBetaIncInv(TensorSpecialMultiOp):
    _ARG_COUNT = 3
    _func_name = "betaincinv"


@implement_scipy("scipy.special.betaincinv")
@infer_scipy_dtype("scipy.special.betaincinv")
def betaincinv(a, b, y, out=None, **kwargs):
    op = TensorBetaIncInv(**kwargs)
    return op(a, b, y, out=out)


@_register_special_op
@arithmetic_operator(sparse_mode="unary")
class TensorPsi(TensorSpecialUnaryOp):
    _func_name = "psi"


@implement_scipy("scipy.special.psi")
@infer_scipy_dtype("scipy.special.psi")
def psi(x, out=None, **kwargs):
    op = TensorPsi(**kwargs)
    return op(x, out=out)


@_register_special_op
@arithmetic_operator(sparse_mode="unary")
class TensorRGamma(TensorSpecialUnaryOp):
    _func_name = "rgamma"


@implement_scipy("scipy.special.rgamma")
@infer_scipy_dtype("scipy.special.rgamma")
def rgamma(x, out=None, **kwargs):
    op = TensorRGamma(**kwargs)
    return op(x, out=out)


@_register_special_op
@arithmetic_operator(sparse_mode="binary_and")
class TensorPolyGamma(NoOrderSpecialMixin, TensorSpecialBinOp):
    _func_name = "polygamma"


@implement_scipy("scipy.special.polygamma")
@infer_scipy_dtype("scipy.special.polygamma")
def polygamma(a, b, **kwargs):
    op = TensorPolyGamma(**kwargs)
    return op(a, b)


@_register_special_op
@arithmetic_operator(sparse_mode="binary_and")
class TensorMultiGammaLn(NoOrderSpecialMixin, TensorSpecialBinOp):
    _func_name = "multigammaln"


@implement_scipy("scipy.special.multigammaln")
@infer_scipy_dtype("scipy.special.multigammaln")
def multigammaln(a, b, **kwargs):
    op = TensorMultiGammaLn(**kwargs)
    return op(a, b)


@_register_special_op
@arithmetic_operator(sparse_mode="unary")
class TensorDiGamma(TensorSpecialUnaryOp):
    _func_name = "digamma"


@implement_scipy("scipy.special.digamma")
@infer_scipy_dtype("scipy.special.digamma")
def digamma(x, out=None, **kwargs):
    op = TensorDiGamma(**kwargs)
    return op(x, out=out)


@_register_special_op
@arithmetic_operator(sparse_mode="binary_and")
class TensorPoch(TensorSpecialBinOp):
    _func_name = "poch"


@implement_scipy("scipy.special.poch")
@infer_scipy_dtype("scipy.special.poch")
def poch(a, b, **kwargs):
    op = TensorPoch(**kwargs)
    return op(a, b)
