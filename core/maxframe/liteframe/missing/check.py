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

from maxframe.liteframe.arithmetic.core import _build_unary_projection
from maxframe.liteframe.arithmetic.logical import invert
from maxframe.liteframe.expressions import LiteFrameIsNA


def isna(lf):
    """Detect missing values.

    Returns a boolean LiteFrame of the same shape where each cell is
    ``True`` where the corresponding value is null (NA) and ``False``
    otherwise.

    Returns
    -------
    LiteFrame
        Boolean LiteFrame marking null positions with ``True``.

    See Also
    --------
    notna : Detect non-missing values.
    dropna : Drop rows with missing values.
    fillna : Fill missing values.

    Examples
    --------
    >>> import maxframe.liteframe as mlf
    >>> lf = mlf.LiteFrame({'a': [1.0, None, 3.0]})
    >>> result = lf.isna()
    >>> result.execute()
           a
    0  False
    1   True
    2  False
    """
    return _build_unary_projection(lf, LiteFrameIsNA)


def notna(lf):
    """Detect non-missing values.

    Returns a boolean LiteFrame of the same shape where each cell is
    ``True`` where the corresponding value is not null (NA) and ``False``
    where it is null.

    Returns
    -------
    LiteFrame

    See Also
    --------
    isna : Detect missing values.
    dropna : Drop rows with missing values.

    Examples
    --------
    >>> import maxframe.liteframe as mlf
    >>> lf = mlf.LiteFrame({'a': [1.0, None, 3.0]})
    >>> result = lf.notna()
    >>> result.execute()
           a
    0   True
    1  False
    2   True
    """
    return invert(isna(lf))


isnull = isna
notnull = notna
