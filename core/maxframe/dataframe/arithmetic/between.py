# Copyright 1999-2025 Alibaba Group Holding Ltd.
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


def between(series, left, right, inclusive="both"):
    """
    Return boolean Series equivalent to left <= series <= right.
    This function returns a boolean vector containing `True` wherever the
    corresponding Series element is between the boundary values `left` and
    `right`. NA values are treated as `False`.

    Parameters
    ----------
    left : scalar or list-like
        Left boundary.
    right : scalar or list-like
        Right boundary.
    inclusive : {"both", "neither", "left", "right"}
        Include boundaries. Whether to set each bound as closed or open.

    Returns
    -------
    Series
        Series representing whether each element is between left and
        right (inclusive).

    See Also
    --------
    Series.gt : Greater than of series and other.
    Series.lt : Less than of series and other.

    Notes
    -----
    This function is equivalent to ``(left <= ser) & (ser <= right)``

    Examples
    --------
    >>> import maxframe.dataframe as md
    >>> s = md.Series([2, 0, 4, 8, np.nan])

    Boundary values are included by default:

    >>> s.between(1, 4).execute()
    0     True
    1    False
    2     True
    3    False
    4    False
    dtype: bool

    With `inclusive` set to ``"neither"`` boundary values are excluded:

    >>> s.between(1, 4, inclusive="neither").execute()
    0     True
    1    False
    2    False
    3    False
    4    False
    dtype: bool

    `left` and `right` can be any scalar value:

    >>> s = md.Series(['Alice', 'Bob', 'Carol', 'Eve'])
    >>> s.between('Anna', 'Daniel').execute()
    0    False
    1     True
    2     True
    3    False
    dtype: bool
    """
    if isinstance(inclusive, bool):  # pragma: no cover
        # for pandas < 1.3.0
        if inclusive:
            inclusive = "both"
        else:
            inclusive = "neither"
    if inclusive == "both":
        lmask = series >= left
        rmask = series <= right
    elif inclusive == "left":
        lmask = series >= left
        rmask = series < right
    elif inclusive == "right":
        lmask = series > left
        rmask = series <= right
    elif inclusive == "neither":
        lmask = series > left
        rmask = series < right
    else:
        raise ValueError(
            "Inclusive has to be either string of 'both',"
            "'left', 'right', or 'neither'."
        )

    return lmask & rmask
