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

"""LiteFrame.value_counts() API entry point.

Counts of unique values via ``groupby().size()`` + ``rename()`` + ``sort()``,
with broadcast-merge normalization for ``normalize=True``.
"""

from maxframe.liteframe.functional.flat_apply import _normalize_subset
from maxframe.liteframe.utils import resolve_unique_col_name


def _check_reserved_names(normalize, cols):
    """Raise if any of *cols* collides with the reserved output/intermediate
    column names ('count'/'proportion' for the result, 'size' for the
    intermediate produced by ``groupby().size()``)."""
    result_name = "proportion" if normalize else "count"
    collisions = {result_name, "size"} & set(cols)
    if collisions:
        raise ValueError(
            f"column(s) {sorted(collisions)} collide with reserved output "
            f"or intermediate column name(s)"
        )


def value_counts(
    self, subset=None, normalize=False, sort=True, ascending=False, dropna=True
):
    """Return a LiteFrame containing counts of unique values.

    Parameters
    ----------
    subset : str or list of str, default None
        Column(s) whose distinct value combinations are counted.  If None,
        all columns are used.
    normalize : bool, default False
        If True, return relative frequencies (proportions) instead of
        counts.  The output column is named ``"proportion"`` (float64)
        instead of ``"count"`` (int64).
    sort : bool, default True
        Sort by count/proportion.
    ascending : bool, default False
        Sort order (descending by default, matching pandas).
    dropna : bool, default True
        Exclude null-key rows.

    Returns
    -------
    LiteFrame
        Frame with the value column(s) and a ``"count"`` or ``"proportion"``
        column.

    Examples
    --------
    >>> import maxframe.liteframe as mlf
    >>> lf = mlf.LiteFrame({'a': [1, 2, 2, 3, 3, 3]})
    >>> result = lf.value_counts('a')
    >>> result.execute()
       a  count
    0  3      3
    1  2      2
    2  1      1
    """
    by = _normalize_subset(subset, self.columns)

    # 'count'/'proportion' is the result column; 'size' is the intermediate
    # produced by groupby().size().  A column named 'count' that is not in
    # subset does not appear in the output and is fine.
    _check_reserved_names(normalize, by)

    counts = self.groupby(by, dropna=dropna).size()  # [by_cols, "size"]

    if not normalize:
        result = counts.rename(columns={"size": "count"})
        if sort:
            result = result.sort("count", ascending=ascending)
        return result

    # Broadcast-merge: sum all group sizes, cross-join, divide.
    # Rename the total's "size" to a collision-free full column name so
    # a user column named "size_y" cannot break the merge.
    key_col = resolve_unique_col_name("__key", self.columns)
    total_name = resolve_unique_col_name("_total", list(self.columns) + ["size"])
    total = (
        counts.agg({"size": "sum"})
        .assign(**{key_col: 1})
        .rename(columns={"size": total_name})
    )
    merged = counts.assign(**{key_col: 1}).merge(
        total,
        on=key_col,
        how="left",
    )
    merged = merged.assign(proportion=merged["size"] / merged[total_name])
    result = merged[by + ["proportion"]]
    if sort:
        result = result.sort("proportion", ascending=ascending)
    return result
