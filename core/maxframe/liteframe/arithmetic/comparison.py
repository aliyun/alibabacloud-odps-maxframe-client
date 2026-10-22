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
    LiteFrameEq,
    LiteFrameGe,
    LiteFrameGt,
    LiteFrameLe,
    LiteFrameLt,
    LiteFrameNe,
)


def eq(lf, other):
    """Element-wise equality comparison.

    Bound to the ``==`` operator (``__eq__``), so ``lf == other`` is
    equivalent to ``lf.eq(other)`` for non-LiteFrame ``other``.

    Parameters
    ----------
    other : scalar, LiteFrame, or LiteFrame column expression
        Value or LiteFrame to compare against.

    Returns
    -------
    LiteFrame
        Lazy boolean LiteFrame indicating element-wise equality.

    See Also
    --------
    ne : Element-wise not-equal comparison.
    lt : Element-wise less-than comparison.

    Examples
    --------
    >>> import maxframe.liteframe as mlf
    >>> lf = mlf.LiteFrame({'a': [1, 2, 3]})
    >>> result = lf.eq(2)
    >>> result.execute()
           a
    0  False
    1   True
    2  False

    Using the ``==`` operator:

    >>> result = lf == 2
    >>> result.execute()
           a
    0  False
    1   True
    2  False
    """
    return _build_binary_projection(lf, LiteFrameEq, other)


def ne(lf, other):
    """Element-wise not-equal comparison.

    Bound to the ``!=`` operator (``__ne__``), so ``lf != other`` is
    equivalent to ``lf.ne(other)`` for non-LiteFrame ``other``.

    Parameters
    ----------
    other : scalar, LiteFrame, or LiteFrame column expression
        Value or LiteFrame to compare against.

    Returns
    -------
    LiteFrame
        Lazy boolean LiteFrame indicating element-wise inequality.

    See Also
    --------
    eq : Element-wise equality comparison.
    ne : Element-wise not-equal comparison.

    Examples
    --------
    >>> import maxframe.liteframe as mlf
    >>> lf = mlf.LiteFrame({'a': [1, 2, 3]})
    >>> result = lf.ne(2)
    >>> result.execute()
           a
    0   True
    1  False
    2   True

    Using the ``!=`` operator:

    >>> result = lf != 2
    >>> result.execute()
           a
    0   True
    1  False
    2   True
    """
    return _build_binary_projection(lf, LiteFrameNe, other)


def lt(lf, other):
    """Element-wise less-than comparison.

    Bound to the ``<`` operator (``__lt__``), so ``lf < other`` is
    equivalent to ``lf.lt(other)`` for non-LiteFrame ``other``.

    Parameters
    ----------
    other : scalar, LiteFrame, or LiteFrame column expression
        Value or LiteFrame to compare against.

    Returns
    -------
    LiteFrame
        Lazy boolean LiteFrame indicating where elements are less than ``other``.

    See Also
    --------
    le : Element-wise less-than-or-equal comparison.
    gt : Element-wise greater-than comparison.

    Examples
    --------
    >>> import maxframe.liteframe as mlf
    >>> lf = mlf.LiteFrame({'a': [1, 2, 3]})
    >>> result = lf.lt(2)
    >>> result.execute()
           a
    0   True
    1  False
    2  False

    Using the ``<`` operator:

    >>> result = lf < 2
    >>> result.execute()
           a
    0   True
    1  False
    2  False
    """
    return _build_binary_projection(lf, LiteFrameLt, other)


def le(lf, other):
    """Element-wise less-than-or-equal comparison.

    Bound to the ``<=`` operator (``__le__``), so ``lf <= other`` is
    equivalent to ``lf.le(other)`` for non-LiteFrame ``other``.

    Parameters
    ----------
    other : scalar, LiteFrame, or LiteFrame column expression
        Value or LiteFrame to compare against.

    Returns
    -------
    LiteFrame
        Lazy boolean LiteFrame indicating where elements are <= ``other``.

    See Also
    --------
    lt : Element-wise less-than comparison.
    ge : Element-wise greater-than-or-equal comparison.

    Examples
    --------
    >>> import maxframe.liteframe as mlf
    >>> lf = mlf.LiteFrame({'a': [1, 2, 3]})
    >>> result = lf.le(2)
    >>> result.execute()
           a
    0   True
    1   True
    2  False

    Using the ``<=`` operator:

    >>> result = lf <= 2
    >>> result.execute()
           a
    0   True
    1   True
    2  False
    """
    return _build_binary_projection(lf, LiteFrameLe, other)


def gt(lf, other):
    """Element-wise greater-than comparison.

    Bound to the ``>`` operator (``__gt__``), so ``lf > other`` is
    equivalent to ``lf.gt(other)`` for non-LiteFrame ``other``.

    Parameters
    ----------
    other : scalar, LiteFrame, or LiteFrame column expression
        Value or LiteFrame to compare against.

    Returns
    -------
    LiteFrame
        Lazy boolean LiteFrame indicating where elements are greater than ``other``.

    See Also
    --------
    ge : Element-wise greater-than-or-equal comparison.
    lt : Element-wise less-than comparison.

    Examples
    --------
    >>> import maxframe.liteframe as mlf
    >>> lf = mlf.LiteFrame({'a': [1, 2, 3]})
    >>> result = lf.gt(2)
    >>> result.execute()
           a
    0  False
    1  False
    2   True

    Using the ``>`` operator:

    >>> result = lf > 2
    >>> result.execute()
           a
    0  False
    1  False
    2   True
    """
    return _build_binary_projection(lf, LiteFrameGt, other)


def ge(lf, other):
    """Element-wise greater-than-or-equal comparison.

    Bound to the ``>=`` operator (``__ge__``), so ``lf >= other`` is
    equivalent to ``lf.ge(other)`` for non-LiteFrame ``other``.

    Parameters
    ----------
    other : scalar, LiteFrame, or LiteFrame column expression
        Value or LiteFrame to compare against.

    Returns
    -------
    LiteFrame
        Lazy boolean LiteFrame indicating where elements are >= ``other``.

    See Also
    --------
    gt : Element-wise greater-than comparison.
    le : Element-wise less-than-or-equal comparison.

    Examples
    --------
    >>> import maxframe.liteframe as mlf
    >>> lf = mlf.LiteFrame({'a': [1, 2, 3]})
    >>> result = lf.ge(2)
    >>> result.execute()
           a
    0  False
    1   True
    2   True

    Using the ``>=`` operator:

    >>> result = lf >= 2
    >>> result.execute()
           a
    0  False
    1   True
    2   True
    """
    return _build_binary_projection(lf, LiteFrameGe, other)
