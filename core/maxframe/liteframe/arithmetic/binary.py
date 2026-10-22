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

from maxframe.liteframe.arithmetic.core import _build_binary_projection
from maxframe.liteframe.expressions import (
    LiteFrameAdd,
    LiteFrameDiv,
    LiteFrameFloorDiv,
    LiteFrameLShift,
    LiteFrameMod,
    LiteFrameMul,
    LiteFramePow,
    LiteFrameRShift,
    LiteFrameSub,
)


def add(lf, other):
    """Element-wise addition of ``other`` to the LiteFrame.

    Bound to the ``+`` operator (``__add__``), so ``lf + other`` is
    equivalent to ``lf.add(other)``.

    Parameters
    ----------
    other : scalar, LiteFrame, or column expression
        Value to add to each element of ``lf``.

    Returns
    -------
    LiteFrame
        Lazy frame with element-wise sum of ``lf`` and ``other``.

    Examples
    --------
    >>> import maxframe.liteframe as mlf
    >>> lf = mlf.LiteFrame({'a': [1, 2]})
    >>> result = lf.add(10)
    >>> result.execute()
        a
    0  11
    1  12

    Using the ``+`` operator:

    >>> result = lf + 10
    >>> result.execute()
        a
    0  11
    1  12
    """
    return _build_binary_projection(lf, LiteFrameAdd, other)


def radd(lf, other):
    """Reverse element-wise addition (``other + lf``).

    Bound to ``__radd__``, so ``other + lf`` is equivalent to
    ``lf.radd(other)``.

    Parameters
    ----------
    other : scalar, LiteFrame, or column expression
        Value added to each element of ``lf`` from the left.

    Returns
    -------
    LiteFrame
        Lazy frame with element-wise sum of ``other`` and ``lf``.

    Examples
    --------
    >>> import maxframe.liteframe as mlf
    >>> lf = mlf.LiteFrame({'a': [1, 2]})
    >>> result = lf.radd(10)
    >>> result.execute()
        a
    0  11
    1  12

    Using reverse ``+``:

    >>> result = 10 + lf
    >>> result.execute()
        a
    0  11
    1  12
    """
    return _build_binary_projection(lf, LiteFrameAdd, other)


def sub(lf, other):
    """Element-wise subtraction (``lf - other``).

    Bound to the ``-`` operator (``__sub__``), so ``lf - other`` is
    equivalent to ``lf.sub(other)``.

    Parameters
    ----------
    other : scalar, LiteFrame, or column expression
        Value to subtract from each element of ``lf``.

    Returns
    -------
    LiteFrame
        Lazy frame with element-wise difference of ``lf`` and ``other``.

    Examples
    --------
    >>> import maxframe.liteframe as mlf
    >>> lf = mlf.LiteFrame({'a': [5, 8]})
    >>> result = lf.sub(2)
    >>> result.execute()
       a
    0  3
    1  6

    Using the ``-`` operator:

    >>> result = lf - 2
    >>> result.execute()
       a
    0  3
    1  6
    """
    return _build_binary_projection(lf, LiteFrameSub, other)


def rsub(lf, other):
    """Reverse element-wise subtraction (``other - lf``).

    Bound to ``__rsub__``, so ``other - lf`` is equivalent to
    ``lf.rsub(other)``.

    Parameters
    ----------
    other : scalar, LiteFrame, or column expression
        Value from which each element of ``lf`` is subtracted.

    Returns
    -------
    LiteFrame
        Lazy frame with element-wise difference of ``other`` and ``lf``.

    Examples
    --------
    >>> import maxframe.liteframe as mlf
    >>> lf = mlf.LiteFrame({'a': [1, 2]})
    >>> result = lf.rsub(10)
    >>> result.execute()
       a
    0  9
    1  8

    Using reverse ``-``:

    >>> result = 10 - lf
    >>> result.execute()
       a
    0  9
    1  8
    """
    return _build_binary_projection(lf, LiteFrameSub, other, reverse=True)


def mul(lf, other):
    """Element-wise multiplication (``lf * other``).

    Bound to the ``*`` operator (``__mul__``), so ``lf * other`` is
    equivalent to ``lf.mul(other)``.

    Parameters
    ----------
    other : scalar, LiteFrame, or column expression
        Value to multiply each element of ``lf`` by.

    Returns
    -------
    LiteFrame
        Lazy frame with element-wise product of ``lf`` and ``other``.

    Examples
    --------
    >>> import maxframe.liteframe as mlf
    >>> lf = mlf.LiteFrame({'a': [2, 3]})
    >>> result = lf.mul(4)
    >>> result.execute()
        a
    0   8
    1  12

    Using the ``*`` operator:

    >>> result = lf * 4
    >>> result.execute()
        a
    0   8
    1  12
    """
    return _build_binary_projection(lf, LiteFrameMul, other)


def rmul(lf, other):
    """Reverse element-wise multiplication (``other * lf``).

    Bound to ``__rmul__``, so ``other * lf`` is equivalent to
    ``lf.rmul(other)``.

    Parameters
    ----------
    other : scalar, LiteFrame, or column expression
        Value multiplied with each element of ``lf`` from the left.

    Returns
    -------
    LiteFrame
        Lazy frame with element-wise product of ``other`` and ``lf``.

    Examples
    --------
    >>> import maxframe.liteframe as mlf
    >>> lf = mlf.LiteFrame({'a': [2, 3]})
    >>> result = lf.rmul(4)
    >>> result.execute()
        a
    0   8
    1  12

    Using reverse ``*``:

    >>> result = 4 * lf
    >>> result.execute()
        a
    0   8
    1  12
    """
    return _build_binary_projection(lf, LiteFrameMul, other)


def div(lf, other):
    """Element-wise true division (``lf / other``).

    Bound to the ``/`` operator (``__truediv__``), so ``lf / other`` is
    equivalent to ``lf.div(other)``.

    Parameters
    ----------
    other : scalar, LiteFrame, or column expression
        Value to divide each element of ``lf`` by.

    Returns
    -------
    LiteFrame
        Lazy frame with element-wise quotient of ``lf`` and ``other``.

    Examples
    --------
    >>> import maxframe.liteframe as mlf
    >>> lf = mlf.LiteFrame({'a': [8, 9]})
    >>> result = lf.div(2)
    >>> result.execute()
         a
    0  4.0
    1  4.5

    Using the ``/`` operator:

    >>> result = lf / 2
    >>> result.execute()
         a
    0  4.0
    1  4.5
    """
    return _build_binary_projection(lf, LiteFrameDiv, other)


def rdiv(lf, other):
    """Reverse element-wise true division (``other / lf``).

    Bound to ``__rtruediv__``, so ``other / lf`` is equivalent to
    ``lf.rdiv(other)``.

    Parameters
    ----------
    other : scalar, LiteFrame, or column expression
        Value divided by each element of ``lf``.

    Returns
    -------
    LiteFrame
        Lazy frame with element-wise quotient of ``other`` and ``lf``.

    Examples
    --------
    >>> import maxframe.liteframe as mlf
    >>> lf = mlf.LiteFrame({'a': [2, 4]})
    >>> result = lf.rdiv(8)
    >>> result.execute()
         a
    0  4.0
    1  2.0

    Using reverse ``/``:

    >>> result = 8 / lf
    >>> result.execute()
         a
    0  4.0
    1  2.0
    """
    return _build_binary_projection(lf, LiteFrameDiv, other, reverse=True)


def floordiv(lf, other):
    """Element-wise floor division (``lf // other``).

    Bound to the ``//`` operator (``__floordiv__``), so ``lf // other`` is
    equivalent to ``lf.floordiv(other)``.

    Parameters
    ----------
    other : scalar, LiteFrame, or column expression
        Value to floor-divide each element of ``lf`` by.

    Returns
    -------
    LiteFrame
        Lazy frame with element-wise floored quotient of ``lf`` and ``other``.

    Examples
    --------
    >>> import maxframe.liteframe as mlf
    >>> lf = mlf.LiteFrame({'a': [7, 9]})
    >>> result = lf.floordiv(2)
    >>> result.execute()
       a
    0  3
    1  4

    Using the ``//`` operator:

    >>> result = lf // 2
    >>> result.execute()
       a
    0  3
    1  4
    """
    return _build_binary_projection(lf, LiteFrameFloorDiv, other)


def rfloordiv(lf, other):
    """Reverse element-wise floor division (``other // lf``).

    Bound to ``__rfloordiv__``, so ``other // lf`` is equivalent to
    ``lf.rfloordiv(other)``.

    Parameters
    ----------
    other : scalar, LiteFrame, or column expression
        Value floor-divided by each element of ``lf``.

    Returns
    -------
    LiteFrame
        Lazy frame with element-wise floored quotient of ``other`` and ``lf``.

    Examples
    --------
    >>> import maxframe.liteframe as mlf
    >>> lf = mlf.LiteFrame({'a': [2, 3]})
    >>> result = lf.rfloordiv(9)
    >>> result.execute()
       a
    0  4
    1  3

    Using reverse ``//``:

    >>> result = 9 // lf
    >>> result.execute()
       a
    0  4
    1  3
    """
    return _build_binary_projection(lf, LiteFrameFloorDiv, other, reverse=True)


def mod(lf, other):
    """Element-wise modulo (``lf % other``).

    Bound to the ``%`` operator (``__mod__``), so ``lf % other`` is
    equivalent to ``lf.mod(other)``.

    Parameters
    ----------
    other : scalar, LiteFrame, or column expression
        Modulus to apply to each element of ``lf``.

    Returns
    -------
    LiteFrame
        Lazy frame with element-wise remainder of ``lf`` divided by ``other``.

    Examples
    --------
    >>> import maxframe.liteframe as mlf
    >>> lf = mlf.LiteFrame({'a': [7, 9]})
    >>> result = lf.mod(3)
    >>> result.execute()
       a
    0  1
    1  0

    Using the ``%`` operator:

    >>> result = lf % 3
    >>> result.execute()
       a
    0  1
    1  0
    """
    return _build_binary_projection(lf, LiteFrameMod, other)


def rmod(lf, other):
    """Reverse element-wise modulo (``other % lf``).

    Bound to ``__rmod__``, so ``other % lf`` is equivalent to
    ``lf.rmod(other)``.

    Parameters
    ----------
    other : scalar, LiteFrame, or column expression
        Value whose remainder is taken modulo each element of ``lf``.

    Returns
    -------
    LiteFrame
        Lazy frame with element-wise remainder of ``other`` divided by ``lf``.

    Examples
    --------
    >>> import maxframe.liteframe as mlf
    >>> lf = mlf.LiteFrame({'a': [3, 4]})
    >>> result = lf.rmod(10)
    >>> result.execute()
       a
    0  1
    1  2

    Using reverse ``%``:

    >>> result = 10 % lf
    >>> result.execute()
       a
    0  1
    1  2
    """
    return _build_binary_projection(lf, LiteFrameMod, other, reverse=True)


def pow(lf, other):
    """Element-wise exponentiation (``lf ** other``).

    Bound to the ``**`` operator (``__pow__``), so ``lf ** other`` is
    equivalent to ``lf.pow(other)``.

    Parameters
    ----------
    other : scalar, LiteFrame, or column expression
        Exponent applied to each element of ``lf``.

    Returns
    -------
    LiteFrame
        Lazy frame with each element of ``lf`` raised to the power ``other``.

    Examples
    --------
    >>> import maxframe.liteframe as mlf
    >>> lf = mlf.LiteFrame({'a': [2, 3]})
    >>> result = lf.pow(2)
    >>> result.execute()
       a
    0   4
    1   9

    Using the ``**`` operator:

    >>> result = lf ** 2
    >>> result.execute()
       a
    0   4
    1   9
    """
    return _build_binary_projection(lf, LiteFramePow, other)


def rpow(lf, other):
    """Reverse element-wise exponentiation (``other ** lf``).

    Bound to ``__rpow__``, so ``other ** lf`` is equivalent to
    ``lf.rpow(other)``.

    Parameters
    ----------
    other : scalar, LiteFrame, or column expression
        Base raised to the power of each element of ``lf``.

    Returns
    -------
    LiteFrame
        Lazy frame with ``other`` raised to the power of each element of ``lf``.

    Examples
    --------
    >>> import maxframe.liteframe as mlf
    >>> lf = mlf.LiteFrame({'a': [2, 3]})
    >>> result = lf.rpow(2)
    >>> result.execute()
       a
    0   4
    1   8

    Using reverse ``**``:

    >>> result = 2 ** lf
    >>> result.execute()
       a
    0   4
    1   8
    """
    return _build_binary_projection(lf, LiteFramePow, other, reverse=True)


def lshift(lf, other):
    """Element-wise left bit shift (``lf << other``).

    Bound to the ``<<`` operator (``__lshift__``), so ``lf << other`` is
    equivalent to ``lf.lshift(other)``.

    Parameters
    ----------
    other : scalar, LiteFrame, or column expression
        Number of bit positions to shift left.

    Returns
    -------
    LiteFrame
        Lazy frame with each element of ``lf`` shifted left by ``other``.

    Examples
    --------
    >>> import maxframe.liteframe as mlf
    >>> lf = mlf.LiteFrame({'a': [1, 2]})
    >>> result = lf.lshift(2)
    >>> result.execute()
       a
    0   4
    1   8

    Using the ``<<`` operator:

    >>> result = lf << 2
    >>> result.execute()
       a
    0   4
    1   8
    """
    return _build_binary_projection(lf, LiteFrameLShift, other)


def rlshift(lf, other):
    """Reverse element-wise left bit shift (``other << lf``).

    Bound to ``__rlshift__``, so ``other << lf`` is equivalent to
    ``lf.rlshift(other)``.

    Parameters
    ----------
    other : scalar, LiteFrame, or column expression
        Integer value shifted left by each element of ``lf``.

    Returns
    -------
    LiteFrame
        Lazy frame with ``other`` shifted left by each element of ``lf``.

    Examples
    --------
    >>> import maxframe.liteframe as mlf
    >>> lf = mlf.LiteFrame({'a': [1, 2]})
    >>> result = lf.rlshift(4)
    >>> result.execute()
       a
    0   8
    1  16

    Using reverse ``<<``:

    >>> result = 4 << lf
    >>> result.execute()
       a
    0   8
    1  16
    """
    return _build_binary_projection(lf, LiteFrameLShift, other, reverse=True)


def rshift(lf, other):
    """Element-wise right bit shift (``lf >> other``).

    Bound to the ``>>`` operator (``__rshift__``), so ``lf >> other`` is
    equivalent to ``lf.rshift(other)``.

    Parameters
    ----------
    other : scalar, LiteFrame, or column expression
        Number of bit positions to shift right.

    Returns
    -------
    LiteFrame
        Lazy frame with each element of ``lf`` shifted right by ``other``.

    Examples
    --------
    >>> import maxframe.liteframe as mlf
    >>> lf = mlf.LiteFrame({'a': [8, 16]})
    >>> result = lf.rshift(2)
    >>> result.execute()
       a
    0  2
    1  4

    Using the ``>>`` operator:

    >>> result = lf >> 2
    >>> result.execute()
       a
    0  2
    1  4
    """
    return _build_binary_projection(lf, LiteFrameRShift, other)


def rrshift(lf, other):
    """Reverse element-wise right bit shift (``other >> lf``).

    Bound to ``__rrshift__``, so ``other >> lf`` is equivalent to
    ``lf.rrshift(other)``.

    Parameters
    ----------
    other : scalar, LiteFrame, or column expression
        Integer value shifted right by each element of ``lf``.

    Returns
    -------
    LiteFrame
        Lazy frame with ``other`` shifted right by each element of ``lf``.

    Examples
    --------
    >>> import maxframe.liteframe as mlf
    >>> lf = mlf.LiteFrame({'a': [1, 2]})
    >>> result = lf.rrshift(16)
    >>> result.execute()
       a
    0  8
    1  4

    Using reverse ``>>``:

    >>> result = 16 >> lf
    >>> result.execute()
       a
    0  8
    1  4
    """
    return _build_binary_projection(lf, LiteFrameRShift, other, reverse=True)
