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

from maxframe.tensor import (
    extensions,
    fetch,
    fft,
    linalg,
    random,
    spatial,
    special,
    statistics,
)
from maxframe.tensor.arithmetic import absolute
from maxframe.tensor.arithmetic import absolute as abs
from maxframe.tensor.arithmetic import (
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
from maxframe.tensor.arithmetic import bitand as bitwise_and
from maxframe.tensor.arithmetic import bitor as bitwise_or
from maxframe.tensor.arithmetic import bitxor as bitwise_xor
from maxframe.tensor.arithmetic import (
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
from maxframe.tensor.arithmetic import floordiv as floor_divide
from maxframe.tensor.arithmetic import (
    fmax,
    fmin,
    fmod,
    frexp,
    greater,
    greater_equal,
    hypot,
    i0,
    imag,
)
from maxframe.tensor.arithmetic import invert
from maxframe.tensor.arithmetic import invert as bitwise_not
from maxframe.tensor.arithmetic import (
    isclose,
    iscomplex,
    iscomplexobj,
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
from maxframe.tensor.arithmetic import lshift as left_shift
from maxframe.tensor.arithmetic import (
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
from maxframe.tensor.arithmetic import round_
from maxframe.tensor.arithmetic import round_ as round
from maxframe.tensor.arithmetic import rshift as right_shift
from maxframe.tensor.arithmetic import (
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
from maxframe.tensor.arithmetic import truediv as true_divide
from maxframe.tensor.arithmetic import trunc
from maxframe.tensor.core import Tensor
from maxframe.tensor.datasource import (
    arange,
    array,
    asarray,
    ascontiguousarray,
    asfortranarray,
    diag,
    diagflat,
    empty,
    empty_like,
    eye,
    from_dataframe,
    full,
    full_like,
    identity,
    indices,
    linspace,
    meshgrid,
    ones,
    ones_like,
    scalar,
    tensor,
    tril,
    triu,
    zeros,
    zeros_like,
)
from maxframe.tensor.indexing import (
    choose,
    compress,
    extract,
    fill_diagonal,
    flatnonzero,
    nonzero,
    take,
    unravel_index,
)
from maxframe.tensor.linalg import (
    dot,
    einsum,
    inner,
    innerproduct,
    matmul,
    tensordot,
    vdot,
)
from maxframe.tensor.merge import (
    append,
    column_stack,
    concatenate,
    dstack,
    hstack,
    stack,
    vstack,
)
from maxframe.tensor.misc import (
    argwhere,
    array_split,
    astype,
    atleast_1d,
    atleast_2d,
    atleast_3d,
    broadcast_arrays,
    broadcast_to,
    copy,
    copyto,
    delete,
    diff,
    dsplit,
    ediff1d,
    expand_dims,
    flip,
    fliplr,
    flipud,
    hsplit,
    in1d,
    insert,
    isin,
    moveaxis,
    ndim,
    ravel,
    repeat,
    result_type,
    roll,
    rollaxis,
    searchsorted,
    setdiff1d,
    shape,
    split,
    squeeze,
    swapaxes,
    tile,
    transpose,
    trapezoid,
    unique,
    vsplit,
    where,
)
from maxframe.tensor.rechunk import rechunk
from maxframe.tensor.reduction import (
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
from maxframe.tensor.reduction import max
from maxframe.tensor.reduction import max as amax
from maxframe.tensor.reduction import mean
from maxframe.tensor.reduction import min
from maxframe.tensor.reduction import min as amin
from maxframe.tensor.reduction import (
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
from maxframe.tensor.reduction import prod
from maxframe.tensor.reduction import prod as product
from maxframe.tensor.reduction import std, sum, var
from maxframe.tensor.reshape import reshape
from maxframe.tensor.sort import argpartition, argsort, partition, sort
from maxframe.tensor.statistics import (
    average,
    bincount,
    corrcoef,
    cov,
    digitize,
    histogram,
    histogram_bin_edges,
    median,
    percentile,
    ptp,
    quantile,
)
from maxframe.tensor.ufunc import ufunc
from maxframe.tensor.utils import AxisError

# isort: off
# noinspection PyUnresolvedReferences
from maxframe.tensor.lib.index_tricks import mgrid, ogrid, ndindex, r_, c_

# noinspection PyUnresolvedReferences
from numpy import (
    e,
    errstate,
    geterr,
    inf,
    nan,
    newaxis,
    pi,
    seterr,
)

NAN = nan
NINF = -inf
Inf = inf
NaN = nan

# import numpy types
# noinspection PyUnresolvedReferences
from numpy import (
    bool_ as bool,
    bytes_,
    character,
    complex64,
    complex128,
    complexfloating,
    datetime64,
    double,
    dtype,
    flexible,
    float16,
    float32,
    float64,
    floating,
    generic,
    inexact,
    int8,
    int16,
    int32,
    int64,
    intc,
    intp,
    number,
    integer,
    object_ as object,
    signedinteger,
    timedelta64,
    uint,
    uint8,
    uint16,
    uint32,
    uint64,
    unsignedinteger,
    void,
)

try:
    from numpy import cfloat
except ImportError:
    from numpy import cdouble as cfloat
try:
    from numpy import str_ as unicode_
except ImportError:
    from numpy import unicode_

del fetch, ufunc
