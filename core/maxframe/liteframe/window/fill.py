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

from maxframe.liteframe.operators.window import LiteFrameWindowTransform


def ffill(self, limit=None, columns=None):
    """Forward-fill: propagate the last non-null value forward.

    Consecutive nulls are replaced with the most recent non-null value
    observed earlier in the row order.  ``limit <= 0`` raises ``ValueError``
    (matching pandas).

    Parameters
    ----------
    limit : int or None, default None
        Maximum number of consecutive nulls to fill in each run.
    columns : str, list of str, or None, default None
        Column(s) to fill.  If None, all visible columns are filled.
        Non-selected columns are passed through unchanged.

    Returns
    -------
    LiteFrame
        Forward-filled LiteFrame with the same shape and columns.  Output
        dtypes equal the input dtypes (no widening, including decimal).

    See Also
    --------
    bfill : Backward-fill missing values.
    fillna : Fill missing values.

    Examples
    --------
    >>> import maxframe.liteframe as mlf
    >>> lf = mlf.LiteFrame({'a': [1.0, None, None, 4.0]})
    >>> result = lf.ffill()
    >>> result.execute()
         a
    0  1.0
    1  1.0
    2  1.0
    3  4.0
    """
    return LiteFrameWindowTransform.for_fill("ffill", limit=limit, columns=columns)(
        self
    )


def bfill(self, limit=None, columns=None):
    """Backward-fill: propagate the next non-null value backward.

    Consecutive nulls are replaced with the next non-null value observed
    later in the row order.  ``limit <= 0`` raises ``ValueError``
    (matching pandas).

    Parameters
    ----------
    limit : int or None, default None
        Maximum number of consecutive nulls to fill in each run.
    columns : str, list of str, or None, default None
        Column(s) to fill.  If None, all visible columns are filled.
        Non-selected columns are passed through unchanged.

    Returns
    -------
    LiteFrame
        Backward-filled LiteFrame with the same shape and columns.  Output
        dtypes equal the input dtypes (no widening, including decimal).

    See Also
    --------
    ffill : Forward-fill missing values.
    fillna : Fill missing values.

    Examples
    --------
    >>> import maxframe.liteframe as mlf
    >>> lf = mlf.LiteFrame({'a': [1.0, None, None, 4.0]})
    >>> result = lf.bfill()
    >>> result.execute()
         a
    0  1.0
    1  4.0
    2  4.0
    3  4.0
    """
    return LiteFrameWindowTransform.for_fill("bfill", limit=limit, columns=columns)(
        self
    )
