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

from numbers import Number

import numpy as np

from maxframe.liteframe.arithmetic.core import _build_unary_projection
from maxframe.liteframe.core import LITEFRAME_TYPE
from maxframe.liteframe.expressions import LiteFrameAbs, LiteFrameNeg, LiteFrameRound


def neg(lf):
    """Element-wise numerical negation of all column values.

    Bound to the unary ``-`` operator (``__neg__``), so ``-lf`` is
    equivalent to ``lf.neg()``.

    Returns
    -------
    LiteFrame
        Lazy LiteFrame with every numeric value negated.

    Examples
    --------
    >>> import maxframe.liteframe as mlf
    >>> lf = mlf.LiteFrame({'a': [1, -2, 3]})
    >>> result = -lf
    >>> result.execute()
       a
    0 -1
    1  2
    2 -3
    """
    return _build_unary_projection(lf, LiteFrameNeg)


def pos(lf):
    """Element-wise unary positive; returns the LiteFrame unchanged.

    Bound to the unary ``+`` operator (``__pos__``), so ``+lf`` is
    equivalent to ``lf.pos()``.

    Returns
    -------
    LiteFrame
        The same LiteFrame, unchanged.

    Examples
    --------
    >>> import maxframe.liteframe as mlf
    >>> lf = mlf.LiteFrame({'a': [1, -2, 3]})
    >>> result = +lf
    >>> result.execute()
       a
    0  1
    1 -2
    2  3
    """
    return lf


def abs_(lf):
    """Element-wise absolute value of all column values.

    Bound to the builtin ``abs()`` (``__abs__``), so ``abs(lf)`` is
    equivalent to ``lf.abs_()``.

    Returns
    -------
    LiteFrame
        Lazy LiteFrame with absolute values.

    Examples
    --------
    >>> import maxframe.liteframe as mlf
    >>> lf = mlf.LiteFrame({'a': [-1, 2, -3]})
    >>> result = abs(lf)
    >>> result.execute()
       a
    0  1
    1  2
    2  3
    """
    return _build_unary_projection(lf, LiteFrameAbs)


def round_(lf, decimals=0):
    """Element-wise rounding to ``decimals`` decimal places.

    Parameters
    ----------
    decimals : int, default 0
        Number of decimal places to round to.

    Returns
    -------
    LiteFrame
        Lazy LiteFrame with rounded values.

    Examples
    --------
    >>> import maxframe.liteframe as mlf
    >>> lf = mlf.LiteFrame({'a': [1.234, 2.345, 3.456]})
    >>> result = lf.round(2)
    >>> result.execute()
         a
    0 1.23
    1 2.35
    2 3.46
    """
    return _build_unary_projection(lf, LiteFrameRound, decimals=decimals)


_UFUNC_MAP = {
    "absolute": abs_,
    "fabs": abs_,
    "negative": neg,
    "positive": pos,
    "round": round_,
    "around": round_,
}


def __array_ufunc__(lf, ufunc, method, *inputs, **kwargs):
    """NumPy ufunc dispatch for LiteFrame."""
    if method != "__call__":
        return NotImplemented

    for x in inputs:
        if not isinstance(x, (Number,)) and not isinstance(x, LITEFRAME_TYPE):
            try:
                np.asarray(x)
            except Exception:
                return NotImplemented

    handler = _UFUNC_MAP.get(ufunc.__name__)
    if handler is None:
        return NotImplemented

    # Find the LiteFrame input
    lf_input = None
    for x in inputs:
        if isinstance(x, LITEFRAME_TYPE):
            lf_input = x
            break

    if lf_input is None:
        return NotImplemented

    return handler(lf_input, **kwargs)
