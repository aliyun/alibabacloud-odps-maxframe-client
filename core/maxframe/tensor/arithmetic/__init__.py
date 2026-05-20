# Copyright 1999-2026 Alibaba Group Holding Ltd.
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

from maxframe.core import is_build_mode
from maxframe.tensor.arithmetic.abs import TensorAbs, abs
from maxframe.tensor.arithmetic.absolute import TensorAbsolute, absolute
from maxframe.tensor.arithmetic.add import TensorAdd, TensorTreeAdd, add, tree_add
from maxframe.tensor.arithmetic.angle import TensorAngle, angle
from maxframe.tensor.arithmetic.arccos import TensorArccos, arccos
from maxframe.tensor.arithmetic.arccosh import TensorArccosh, arccosh
from maxframe.tensor.arithmetic.arcsin import TensorArcsin, arcsin
from maxframe.tensor.arithmetic.arcsinh import TensorArcsinh, arcsinh
from maxframe.tensor.arithmetic.arctan import TensorArctan, arctan
from maxframe.tensor.arithmetic.arctan2 import TensorArctan2, arctan2
from maxframe.tensor.arithmetic.arctanh import TensorArctanh, arctanh
from maxframe.tensor.arithmetic.around import TensorAround
from maxframe.tensor.arithmetic.around import around
from maxframe.tensor.arithmetic.around import around as round_
from maxframe.tensor.arithmetic.bitand import TensorBitand, bitand
from maxframe.tensor.arithmetic.bitor import TensorBitor, bitor
from maxframe.tensor.arithmetic.bitxor import TensorBitxor, bitxor
from maxframe.tensor.arithmetic.cbrt import TensorCbrt, cbrt
from maxframe.tensor.arithmetic.ceil import TensorCeil, ceil
from maxframe.tensor.arithmetic.clip import TensorClip, clip
from maxframe.tensor.arithmetic.conj import TensorConj
from maxframe.tensor.arithmetic.conj import conj
from maxframe.tensor.arithmetic.conj import conj as conjugate
from maxframe.tensor.arithmetic.copysign import TensorCopysign, copysign
from maxframe.tensor.arithmetic.cos import TensorCos, cos
from maxframe.tensor.arithmetic.cosh import TensorCosh, cosh
from maxframe.tensor.arithmetic.deg2rad import TensorDeg2rad, deg2rad
from maxframe.tensor.arithmetic.degrees import TensorDegrees, degrees
from maxframe.tensor.arithmetic.divide import TensorDivide, divide
from maxframe.tensor.arithmetic.equal import TensorEqual, equal
from maxframe.tensor.arithmetic.exp import TensorExp, exp
from maxframe.tensor.arithmetic.exp2 import TensorExp2, exp2
from maxframe.tensor.arithmetic.expm1 import TensorExpm1, expm1
from maxframe.tensor.arithmetic.fabs import TensorFabs, fabs
from maxframe.tensor.arithmetic.fix import TensorFix, fix
from maxframe.tensor.arithmetic.float_power import TensorFloatPower, float_power
from maxframe.tensor.arithmetic.floor import TensorFloor, floor
from maxframe.tensor.arithmetic.floordiv import TensorFloorDiv, floordiv
from maxframe.tensor.arithmetic.fmax import TensorFMax, fmax
from maxframe.tensor.arithmetic.fmin import TensorFMin, fmin
from maxframe.tensor.arithmetic.fmod import TensorFMod, fmod
from maxframe.tensor.arithmetic.frexp import TensorFrexp, frexp
from maxframe.tensor.arithmetic.greater import TensorGreaterThan, greater
from maxframe.tensor.arithmetic.greater_equal import TensorGreaterEqual, greater_equal
from maxframe.tensor.arithmetic.hypot import TensorHypot, hypot
from maxframe.tensor.arithmetic.i0 import TensorI0, i0
from maxframe.tensor.arithmetic.imag import TensorImag, imag
from maxframe.tensor.arithmetic.invert import TensorInvert, invert
from maxframe.tensor.arithmetic.isclose import TensorIsclose, isclose
from maxframe.tensor.arithmetic.iscomplex import TensorIsComplex, iscomplex
from maxframe.tensor.arithmetic.iscomplexobj import iscomplexobj
from maxframe.tensor.arithmetic.isfinite import TensorIsFinite, isfinite
from maxframe.tensor.arithmetic.isinf import TensorIsInf, isinf
from maxframe.tensor.arithmetic.isnan import TensorIsNan, isnan
from maxframe.tensor.arithmetic.isreal import TensorIsReal, isreal
from maxframe.tensor.arithmetic.ldexp import TensorLdexp, ldexp
from maxframe.tensor.arithmetic.less import TensorLessThan, less
from maxframe.tensor.arithmetic.less_equal import TensorLessEqual, less_equal
from maxframe.tensor.arithmetic.log import TensorLog, log
from maxframe.tensor.arithmetic.log1p import TensorLog1p, log1p
from maxframe.tensor.arithmetic.log2 import TensorLog2, log2
from maxframe.tensor.arithmetic.log10 import TensorLog10, log10
from maxframe.tensor.arithmetic.logaddexp import TensorLogAddExp, logaddexp
from maxframe.tensor.arithmetic.logaddexp2 import TensorLogAddExp2, logaddexp2
from maxframe.tensor.arithmetic.logical_and import TensorAnd, logical_and
from maxframe.tensor.arithmetic.logical_not import TensorNot, logical_not
from maxframe.tensor.arithmetic.logical_or import TensorOr, logical_or
from maxframe.tensor.arithmetic.logical_xor import TensorXor, logical_xor
from maxframe.tensor.arithmetic.lshift import TensorLshift, lshift
from maxframe.tensor.arithmetic.maximum import TensorMaximum, maximum
from maxframe.tensor.arithmetic.minimum import TensorMinimum, minimum
from maxframe.tensor.arithmetic.mod import TensorMod
from maxframe.tensor.arithmetic.mod import mod
from maxframe.tensor.arithmetic.mod import mod as remainder
from maxframe.tensor.arithmetic.modf import TensorModf, modf
from maxframe.tensor.arithmetic.multiply import (
    TensorMultiply,
    TensorTreeMultiply,
    multiply,
    tree_multiply,
)
from maxframe.tensor.arithmetic.nan_to_num import TensorNanToNum, nan_to_num
from maxframe.tensor.arithmetic.negative import TensorNegative, negative
from maxframe.tensor.arithmetic.nextafter import TensorNextafter, nextafter
from maxframe.tensor.arithmetic.not_equal import TensorNotEqual, not_equal
from maxframe.tensor.arithmetic.positive import TensorPositive, positive
from maxframe.tensor.arithmetic.power import TensorPower, power
from maxframe.tensor.arithmetic.rad2deg import TensorRad2deg, rad2deg
from maxframe.tensor.arithmetic.radians import TensorRadians, radians
from maxframe.tensor.arithmetic.real import TensorReal, real
from maxframe.tensor.arithmetic.reciprocal import TensorReciprocal, reciprocal
from maxframe.tensor.arithmetic.rint import TensorRint, rint
from maxframe.tensor.arithmetic.rshift import TensorRshift, rshift
from maxframe.tensor.arithmetic.setimag import TensorSetImag
from maxframe.tensor.arithmetic.setreal import TensorSetReal
from maxframe.tensor.arithmetic.sign import TensorSign, sign
from maxframe.tensor.arithmetic.signbit import TensorSignbit, signbit
from maxframe.tensor.arithmetic.sin import TensorSin, sin
from maxframe.tensor.arithmetic.sinc import TensorSinc, sinc
from maxframe.tensor.arithmetic.sinh import TensorSinh, sinh
from maxframe.tensor.arithmetic.spacing import TensorSpacing, spacing
from maxframe.tensor.arithmetic.sqrt import TensorSqrt, sqrt
from maxframe.tensor.arithmetic.square import TensorSquare, square
from maxframe.tensor.arithmetic.subtract import TensorSubtract, subtract
from maxframe.tensor.arithmetic.tan import TensorTan, tan
from maxframe.tensor.arithmetic.tanh import TensorTanh, tanh
from maxframe.tensor.arithmetic.truediv import TensorTrueDiv, truediv
from maxframe.tensor.arithmetic.trunc import TensorTrunc, trunc


def _install():
    from maxframe.tensor.arithmetic.add import add, radd
    from maxframe.tensor.arithmetic.bitand import bitand, rbitand
    from maxframe.tensor.arithmetic.bitor import bitor, rbitor
    from maxframe.tensor.arithmetic.bitxor import bitxor, rbitxor
    from maxframe.tensor.arithmetic.divide import divide, rdivide
    from maxframe.tensor.arithmetic.floordiv import floordiv, rfloordiv
    from maxframe.tensor.arithmetic.lshift import lshift, rlshift
    from maxframe.tensor.arithmetic.mod import mod, rmod
    from maxframe.tensor.arithmetic.multiply import multiply, rmultiply
    from maxframe.tensor.arithmetic.power import power, rpower
    from maxframe.tensor.arithmetic.rshift import rrshift, rshift
    from maxframe.tensor.arithmetic.subtract import rsubtract, subtract
    from maxframe.tensor.arithmetic.truediv import rtruediv, truediv
    from maxframe.tensor.core import TENSOR_TYPE, Tensor, TensorData
    from maxframe.tensor.datasource import tensor as astensor

    def _wrap_equal(func):
        def eq(x1, x2, **kwargs):
            if is_build_mode():
                return astensor(x1)._equals(x2)
            return func(x1, x2, **kwargs)

        return eq

    def _wrap_iop(func):
        def inner(self, *args, **kwargs):
            kwargs["out"] = self
            return func(self, *args, **kwargs)

        return inner

    def _wrap_magic(func):
        def inner(lhs, rhs, **kwargs):
            ret = func(lhs, rhs, **kwargs)
            if isinstance(ret, TENSOR_TYPE):
                ret.op.extra_params["magic"] = True
            return ret

        return inner

    for cls in TENSOR_TYPE:
        setattr(cls, "__add__", add)
        setattr(cls, "__iadd__", _wrap_iop(add))
        setattr(cls, "__radd__", radd)
        setattr(cls, "__sub__", subtract)
        setattr(cls, "__isub__", _wrap_iop(subtract))
        setattr(cls, "__rsub__", rsubtract)
        setattr(cls, "__mul__", multiply)
        setattr(cls, "__imul__", _wrap_iop(multiply))
        setattr(cls, "__rmul__", rmultiply)
        setattr(cls, "__div__", divide)
        setattr(cls, "__idiv__", _wrap_iop(divide))
        setattr(cls, "__rdiv__", rdivide)
        setattr(cls, "__truediv__", truediv)
        setattr(cls, "__itruediv__", _wrap_iop(truediv))
        setattr(cls, "__rtruediv__", rtruediv)
        setattr(cls, "__floordiv__", floordiv)
        setattr(cls, "__ifloordiv__", _wrap_iop(floordiv))
        setattr(cls, "__rfloordiv__", rfloordiv)
        setattr(cls, "__pow__", power)
        setattr(cls, "__ipow__", _wrap_iop(power))
        setattr(cls, "__rpow__", rpower)
        setattr(cls, "__mod__", mod)
        setattr(cls, "__imod__", _wrap_iop(mod))
        setattr(cls, "__rmod__", rmod)
        setattr(cls, "__lshift__", lshift)
        setattr(cls, "__ilshift__", _wrap_iop(lshift))
        setattr(cls, "__rlshift__", rlshift)
        setattr(cls, "__rshift__", rshift)
        setattr(cls, "__irshift__", _wrap_iop(rshift))
        setattr(cls, "__rrshift__", rrshift)

        setattr(cls, "__eq__", _wrap_magic(_wrap_equal(equal)))
        setattr(cls, "__ne__", _wrap_magic(not_equal))
        setattr(cls, "__lt__", less)
        setattr(cls, "__le__", less_equal)
        setattr(cls, "__gt__", greater)
        setattr(cls, "__ge__", greater_equal)
        setattr(cls, "__and__", bitand)
        setattr(cls, "__iand__", _wrap_iop(bitand))
        setattr(cls, "__rand__", rbitand)
        setattr(cls, "__or__", bitor)
        setattr(cls, "__ior__", _wrap_iop(bitor))
        setattr(cls, "__ror__", rbitor)
        setattr(cls, "__xor__", bitxor)
        setattr(cls, "__ixor__", _wrap_iop(bitxor))
        setattr(cls, "__rxor__", rbitxor)

        setattr(cls, "__neg__", negative)
        setattr(cls, "__pos__", positive)
        setattr(cls, "__abs__", abs)
        setattr(cls, "__invert__", invert)

    setattr(Tensor, "round", round_)
    setattr(Tensor, "conj", conj)
    setattr(Tensor, "conjugate", conjugate)
    setattr(TensorData, "round", round_)
    setattr(TensorData, "conj", conj)
    setattr(TensorData, "conjugate", conjugate)


_install()
del _install


BIN_UFUNC = {
    add,
    subtract,
    multiply,
    divide,
    truediv,
    floordiv,
    power,
    mod,
    fmod,
    logaddexp,
    logaddexp2,
    equal,
    not_equal,
    less,
    less_equal,
    greater,
    greater_equal,
    arctan2,
    hypot,
    bitand,
    bitor,
    bitxor,
    lshift,
    rshift,
    logical_and,
    logical_or,
    logical_xor,
    maximum,
    minimum,
    float_power,
    remainder,
    fmax,
    fmin,
    copysign,
    nextafter,
    ldexp,
}

UNARY_UFUNC = {
    square,
    arcsinh,
    rint,
    sign,
    conj,
    tan,
    absolute,
    deg2rad,
    log,
    fabs,
    exp2,
    invert,
    negative,
    sqrt,
    arctan,
    positive,
    cbrt,
    log10,
    sin,
    rad2deg,
    log2,
    arcsin,
    expm1,
    arctanh,
    cosh,
    sinh,
    cos,
    reciprocal,
    tanh,
    log1p,
    exp,
    arccos,
    arccosh,
    around,
    logical_not,
    conjugate,
    isfinite,
    isinf,
    isnan,
    signbit,
    spacing,
    floor,
    ceil,
    trunc,
    degrees,
    radians,
    angle,
    isreal,
    iscomplex,
    real,
    imag,
    fix,
    i0,
    sinc,
    nan_to_num,
}
