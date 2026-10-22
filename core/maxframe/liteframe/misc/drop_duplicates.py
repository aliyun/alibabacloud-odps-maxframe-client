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

from maxframe.liteframe.operators.drop_duplicates import (
    _VALID_KEEP,
    _VALID_METHOD,
    LiteFrameDropDuplicates,
)
from maxframe.liteframe.utils import normalize_col_name


def _validate_subset(self, subset):
    if subset is None:
        return None
    if isinstance(subset, str):
        subset = [subset]
    if len(subset) == 0:
        raise ValueError("subset must not be an empty list (use None for all columns)")
    cols = list(dict.fromkeys(normalize_col_name(c) for c in subset))
    for name in cols:
        if name not in self._data.dtypes.index:
            raise KeyError(f"Column '{name}' not found in LiteFrame")
    return cols


def _build(self, subset, keep, method, return_mask, return_value):
    """Shared constructor for drop_duplicates / duplicated."""
    if not (keep in ("first", "last") or keep is False):
        # Identity-aware: `0 == False` would otherwise let keep=0 pass and
        # take the wrong executor path (executors use `keep is False`).
        raise ValueError(f"keep must be one of {_VALID_KEEP}, got {keep!r}")
    if method not in _VALID_METHOD:
        raise ValueError(f"method must be one of {_VALID_METHOD}, got {method!r}")
    subset = _validate_subset(self, subset)
    op = LiteFrameDropDuplicates(
        subset=subset,
        keep=keep,
        return_mask=return_mask,
        return_value=return_value,
        method=method,
    )
    return op(self)


def drop_duplicates(self, subset=None, keep="first", method="auto"):
    """Return LiteFrame with duplicate rows removed.

    Parameters
    ----------
    subset : str or list of str, optional
        Only consider certain columns for identifying duplicates, by
        default use all columns.
    keep : {"first", "last", False}, default "first"
        - ``first``: drop duplicates except the first occurrence.
        - ``last``: drop duplicates except the last occurrence.
        - ``False``: drop all duplicates.
    method : {"auto", "tree", "subset_tree", "shuffle"}, default "auto"
        Tiling strategy for multi-chunk inputs.

    Returns
    -------
    LiteFrame
        Deduplicated LiteFrame.

    See Also
    --------
    duplicated : Return boolean mask marking duplicate rows.
    sort : Sort the LiteFrame by columns.

    Examples
    --------
    >>> import maxframe.liteframe as mlf
    >>> lf = mlf.LiteFrame({'a': [1, 1, 2], 'b': [4, 4, 6]})
    >>> result = lf.drop_duplicates()
    >>> result.execute()
       a  b
    0  1  4
    2  2  6
    """
    return _build(self, subset, keep, method, return_mask=False, return_value=True)


def duplicated(self, subset=None, keep="first", method="auto"):
    """Return boolean mask marking duplicate rows.

    Parameters
    ----------
    subset : str or list of str, optional
        Only consider certain columns for identifying duplicates, by
        default use all columns.
    keep : {"first", "last", False}, default "first"
        - ``first``: mark duplicates as ``True`` except for the first
          occurrence.
        - ``last``: mark duplicates as ``True`` except for the last
          occurrence.
        - ``False``: mark all duplicates as ``True``.
    method : {"auto", "tree", "subset_tree", "shuffle"}, default "auto"
        Tiling strategy for multi-chunk inputs.

    Returns
    -------
    LiteFrame
        A one-column boolean LiteFrame with the same row count as the input.

    See Also
    --------
    drop_duplicates : Return LiteFrame with duplicate rows removed.

    Examples
    --------
    >>> import maxframe.liteframe as mlf
    >>> lf = mlf.LiteFrame({'a': [1, 1, 2], 'b': [4, 4, 6]})
    >>> result = lf.duplicated()
    >>> result.execute()
           0
    0  False
    1   True
    2  False
    """
    return _build(self, subset, keep, method, return_mask=True, return_value=False)
