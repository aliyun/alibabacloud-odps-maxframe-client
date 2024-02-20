# Copyright 1999-2024 Alibaba Group Holding Ltd.
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

from . import fetch, random
from .arithmetic import absolute
from .arithmetic import absolute as abs
from .arithmetic import (
    add,
    angle,
    arccos,
    arccosh,
    arcsin,
    arcsinh,
    arctan,
    arctan2,
    arctanh,
    around,
)
from .arithmetic import bitand as bitwise_and
from .arithmetic import bitor as bitwise_or
from .arithmetic import bitxor as bitwise_xor
from .arithmetic import (
    cbrt,
    ceil,
    clip,
    conj,
    conjugate,
    copysign,
    cos,
    cosh,
    deg2rad,
    degrees,
    divide,
    equal,
    exp,
    exp2,
    expm1,
    fabs,
    fix,
    float_power,
    floor,
)
from .arithmetic import floordiv as floor_divide
from .arithmetic import fmax, fmin, fmod, frexp, greater, greater_equal, hypot, i0, imag
from .arithmetic import invert
from .arithmetic import invert as bitwise_not
from .arithmetic import (
    isclose,
    iscomplex,
    isfinite,
    isinf,
    isnan,
    isreal,
    ldexp,
    less,
    less_equal,
    log,
    log1p,
    log2,
    log10,
    logaddexp,
    logaddexp2,
    logical_and,
    logical_not,
    logical_or,
    logical_xor,
)
from .arithmetic import lshift as left_shift
from .arithmetic import (
    maximum,
    minimum,
    mod,
    modf,
    multiply,
    nan_to_num,
    negative,
    nextafter,
    not_equal,
    positive,
    power,
    rad2deg,
    radians,
    real,
    reciprocal,
    remainder,
    rint,
)
from .arithmetic import round_
from .arithmetic import round_ as round
from .arithmetic import rshift as right_shift
from .arithmetic import (
    sign,
    signbit,
    sin,
    sinc,
    sinh,
    spacing,
    sqrt,
    square,
    subtract,
    tan,
    tanh,
)
from .arithmetic import truediv as true_divide
from .arithmetic import trunc
from .base import broadcast_to, transpose, where
from .core import Tensor
from .datasource import (
    arange,
    array,
    asarray,
    ascontiguousarray,
    asfortranarray,
    empty,
    empty_like,
    from_dataframe,
    full,
    full_like,
    ones,
    ones_like,
    scalar,
    tensor,
    zeros,
)
from .indexing import (
    choose,
    compress,
    extract,
    fill_diagonal,
    flatnonzero,
    nonzero,
    take,
    unravel_index,
)
from .merge import stack
from .rechunk import rechunk
from .reduction import (
    all,
    allclose,
    any,
    argmax,
    argmin,
    array_equal,
    count_nonzero,
    cumprod,
    cumsum,
)
from .reduction import max
from .reduction import max as amax
from .reduction import mean
from .reduction import min
from .reduction import min as amin
from .reduction import (
    nanargmax,
    nanargmin,
    nancumprod,
    nancumsum,
    nanmax,
    nanmean,
    nanmin,
    nanprod,
    nanstd,
    nansum,
    nanvar,
)
from .reduction import prod
from .reduction import prod as product
from .reduction import std, sum, var
from .reshape import reshape
from .ufunc import ufunc

del fetch, ufunc
