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


def shift(self, periods=1, fill_value=None, columns=None):
    """Shift rows by ``periods`` positions.

    Positive ``periods`` shifts data forward (down); negative shifts
    backward (up).  Positions introduced by the shift are filled with
    ``fill_value`` (default: null).

    Parameters
    ----------
    periods : int, default 1
        Number of positions to shift.  Positive = forward, negative = backward.
    fill_value : scalar, default None
        Value to fill positions introduced by shifting.  None produces null.
    columns : str, list of str, or None, default None
        Column(s) to shift.  If None, all columns are shifted.
        Non-selected columns are passed through unchanged.

    Returns
    -------
    LiteFrame
        Shifted LiteFrame with the same shape and columns. Output dtypes
        depend on the input dtype and ``fill_value``: when ``fill_value``
        is None (the default) input dtypes are preserved; when a
        ``fill_value`` is supplied a column may be widened (e.g. an
        integer column with a float fill becomes Float64, a string fill
        becomes String). Boolean columns coerce numeric fills to bool;
        decimal columns derive precision from the fill value; a string
        fill on a boolean column is rejected at plan time. Non-selected
        (passthrough) columns keep their input dtypes.

    See Also
    --------
    diff : Compute the difference between consecutive rows.
    pct_change : Compute percent change over given periods.
    rolling : Provide rolling-window calculations.

    Examples
    --------
    >>> import maxframe.liteframe as mlf
    >>> lf = mlf.LiteFrame({'a': [1, 2, 3, 4]})
    >>> result = lf.shift(1)
    >>> result.execute()
         a
    0  NaN
    1  1.0
    2  2.0
    3  3.0
    """
    return LiteFrameWindowTransform.for_positional(
        "shift", periods=periods, fill_value=fill_value, columns=columns
    )(self)
