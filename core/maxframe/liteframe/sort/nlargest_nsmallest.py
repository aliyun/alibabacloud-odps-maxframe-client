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

import numpy as np

from maxframe.liteframe.operators.sort import LiteFrameSort
from maxframe.liteframe.utils import normalize_col_name
from maxframe.utils import is_float_dtype, is_integer_dtype

_VALID_KEEP = {"first", "last", "all"}


def _is_supported_key_dtype(dtype):
    """Whether a sort-key dtype is one pandas nlargest/nsmallest accepts.

    Frames built from pandas expose numpy/Arrow dtypes; a LiteFrame built from a
    polars frame exposes polars dtypes instead, which may be bare classes on
    older Polars, so the shared helpers resolve both representations.
    """
    return is_integer_dtype(dtype) or is_float_dtype(dtype)


def _validate(self, n, columns, keep):
    """Validate arguments and return normalized ``(by, n)``.

    ``n`` is normalized: negative values become 0, matching pandas
    ``nlargest``/``nsmallest`` which treat negative ``n`` as empty selection.
    """
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)):
        raise TypeError(f"n must be an int, got {type(n).__name__}")
    n = int(n)
    # pandas treats negative n as an empty selection (equivalent to 0);
    # normalize here so both backends behave consistently.
    if n < 0:
        n = 0

    if keep not in _VALID_KEEP:
        raise ValueError(f"Invalid keep '{keep}', must be one of {sorted(_VALID_KEEP)}")

    if isinstance(columns, str):
        by = [normalize_col_name(columns)]
    else:
        by = [normalize_col_name(c) for c in columns]

    # Reject duplicate column labels after normalization — both backends
    # fail at execution time with confusing errors on duplicate sort keys.
    dupes = {c for c in by if by.count(c) > 1}
    if dupes:
        raise ValueError(f"Duplicate column labels in 'columns': {sorted(dupes)}")

    # Early column validation — KeyError at call time, not execution time
    dtypes = self._data.dtypes
    missing = [c for c in by if c not in dtypes.index]
    if missing:
        raise KeyError(
            f"Columns {missing} not found in frame columns {list(dtypes.index)}"
        )

    # Dtype validation — pandas nlargest/nsmallest only supports numeric
    # columns (int, uint, float). String, bool, datetime, category, and
    # complex dtypes raise TypeError in SPE (pandas), while DPE (Polars)
    # silently accepts them, causing backend-dependent behavior.
    # Validate here so both backends fail consistently before execution.
    unsupported = [c for c in by if not _is_supported_key_dtype(dtypes[c])]
    if unsupported:
        raise TypeError(
            f"Columns {unsupported} have unsupported dtype(s) for "
            f"nlargest/nsmallest; only numeric columns are supported"
        )
    return by, n


def _build_top_k(self, n, columns, keep, ascending):
    """Validate the arguments and apply the top-n sort to ``self``."""
    by, n = _validate(self, n, columns, keep)
    op = LiteFrameSort(
        by=by,
        ascending=ascending,
        nrows=n,
        keep_kind=keep,
        na_position="last",
    )
    return op(self)


def nlargest(self, n, columns, keep="first"):
    """Return the first `n` rows ordered by `columns` in descending order.

    Equivalent to ``self.sort(columns, ascending=False).head(n)`` but
    more performant in distributed execution (tree-reduction top-k).

    Parameters
    ----------
    n : int
        Number of rows to return.
    columns : str or list of str
        Column label(s) to order by.
    keep : {'first', 'last', 'all'}, default 'first'
        Where there are ties at the nth boundary:
        - 'first': keep the first-occurring tied rows.
        - 'last': keep the last-occurring tied rows.
        - 'all': keep all tied rows (may return more than `n` rows).

    Returns
    -------
    LiteFrame
        The first `n` rows ordered by `columns` descending.

    Examples
    --------
    >>> import maxframe.dataframe as mdf
    >>> mlf = mdf.LiteFrame({'a': [3, 1, 2, 1], 'b': list('abcd')})
    >>> mlf.nlargest(2, 'a').execute()
       a  b
    0  3  a
    1  2  c
    """
    return _build_top_k(self, n, columns, keep, ascending=False)


def nsmallest(self, n, columns, keep="first"):
    """Return the first `n` rows ordered by `columns` in ascending order.

    Equivalent to ``self.sort(columns, ascending=True).head(n)`` but
    more performant in distributed execution (tree-reduction top-k).

    Parameters
    ----------
    n : int
        Number of rows to return.
    columns : str or list of str
        Column label(s) to order by.
    keep : {'first', 'last', 'all'}, default 'first'
        Where there are ties at the nth boundary:
        - 'first': keep the first-occurring tied rows.
        - 'last': keep the last-occurring tied rows.
        - 'all': keep all tied rows (may return more than `n` rows).

    Returns
    -------
    LiteFrame
        The first `n` rows ordered by `columns` ascending.

    Examples
    --------
    >>> import maxframe.dataframe as mdf
    >>> mlf = mdf.LiteFrame({'a': [3, 1, 2, 1], 'b': list('abcd')})
    >>> mlf.nsmallest(2, 'a').execute()
       a  b
    0  1  b
    1  1  d
    """
    return _build_top_k(self, n, columns, keep, ascending=True)
