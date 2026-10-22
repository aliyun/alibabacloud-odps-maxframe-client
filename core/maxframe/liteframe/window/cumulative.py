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


def _make_cumulative(func_name):
    """Create a cumulative transform function (cumsum/cummax/cummin/cumprod).

    Builds the docstring dynamically so each function has a self-documenting
    signature visible to IDEs and ``help()``.
    """
    # "cumsum" -> "sum", "cummax" -> "max", etc.
    op_name = func_name[len("cum") :] if func_name.startswith("cum") else func_name

    # Expected output for input [1, 2, 3, 4]:
    _example_output = {
        "cumsum": "        0   1.0\n        1   3.0\n        2   6.0\n        3  10.0",
        "cummax": "        0  1.0\n        1  2.0\n        2  3.0\n        3  4.0",
        "cummin": "        0  1.0\n        1  1.0\n        2  1.0\n        3  1.0",
        "cumprod": "        0   1.0\n        1   2.0\n        2   6.0\n        3  24.0",
    }[func_name]

    def _cumulative(self, skipna=True, columns=None):
        return LiteFrameWindowTransform.for_cumulative(
            func_name, skipna=skipna, columns=columns
        )(self)

    _cumulative.__name__ = func_name
    _cumulative.__doc__ = f"""Cumulative {op_name} of column values, preserving row order.

    NaN positions are masked (the output is null where the input is null)
    when ``skipna=True``.  When ``skipna=False``, NaN propagates forward —
    once a null is encountered, all subsequent values become null.

    Parameters
    ----------
    skipna : bool, default True
        If True, skip null values in the cumulative computation but mask
        their positions to null in the output.  If False, null propagates
        forward (all subsequent values become null).
    columns : str, list of str, or None, default None
        Column(s) to transform.  If None, all columns are transformed.
        Non-selected columns are passed through unchanged.

    Returns
    -------
    LiteFrame
        Cumulative {op_name} LiteFrame with same shape and columns.

    Raises
    ------
    TypeError
        If any transformed column is non-numeric, or if {func_name}
        is called on a decimal column (cumprod only — polars cum_prod
        does not support decimal).

    Examples
    --------
    >>> import maxframe.liteframe as mlf
    >>> lf = mlf.LiteFrame({{'a': [1, 2, 3, 4]}})
    >>> result = lf.{func_name}()
    >>> result.execute()
       a
{_example_output}
    """
    return _cumulative


cumsum = _make_cumulative("cumsum")
cummax = _make_cumulative("cummax")
cummin = _make_cumulative("cummin")
cumprod = _make_cumulative("cumprod")
