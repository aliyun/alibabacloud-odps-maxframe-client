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


def diff(self, periods=1, columns=None):
    """Compute ``x[i] - x[i-periods]`` for each row.

    First ``periods`` rows (for positive periods) or last ``|periods|``
    rows (for negative periods) are null.

    Parameters
    ----------
    periods : int, default 1
        Number of positions to diff.  Positive = forward, negative = backward.
    columns : str, list of str, or None, default None
        Column(s) to diff.  If None, all columns are diffed.
        Non-selected columns are passed through unchanged.

    Returns
    -------
    LiteFrame
        Diffed LiteFrame with same shape and columns.

    Raises
    ------
    TypeError
        If any transformed column is non-numeric.

    See Also
    --------
    shift : Shift rows by a given number of periods.
    pct_change : Compute percent change over given periods.

    Examples
    --------
    >>> import maxframe.liteframe as mlf
    >>> lf = mlf.LiteFrame({'a': [1, 2, 4, 7]})
    >>> result = lf.diff()
    >>> result.execute()
         a
    0  NaN
    1  1.0
    2  2.0
    3  3.0
    """
    return LiteFrameWindowTransform.for_positional(
        "diff", periods=periods, columns=columns
    )(self)
