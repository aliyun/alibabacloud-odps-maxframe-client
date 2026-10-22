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


def pct_change(self, periods=1, columns=None):
    """Compute ``(x[i] - x[i-periods]) / x[i-periods]`` for each row.

    First ``periods`` rows (for positive periods) or last ``|periods|``
    rows (for negative periods) are null.  Division by zero produces
    null/inf following Polars arithmetic semantics.

    Parameters
    ----------
    periods : int, default 1
        Number of positions for pct_change.  Positive = forward,
        negative = backward.
    columns : str, list of str, or None, default None
        Column(s) for pct_change.  If None, all columns are transformed.
        Non-selected columns are passed through unchanged.

    Returns
    -------
    LiteFrame
        Pct-change LiteFrame with same shape.  Transformed columns
        become Float64.

    Raises
    ------
    TypeError
        If any transformed column is non-numeric.

    See Also
    --------
    diff : Compute the difference between consecutive rows.
    shift : Shift rows by a given number of periods.

    Examples
    --------
    >>> import maxframe.liteframe as mlf
    >>> lf = mlf.LiteFrame({'a': [100, 110, 121]})
    >>> result = lf.pct_change()
    >>> result.execute()
         a
    0  NaN
    1  0.1
    2  0.1
    """
    return LiteFrameWindowTransform.for_positional(
        "pct_change", periods=periods, columns=columns
    )(self)
