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

from maxframe.liteframe.operators.merge import LiteFrameMerge
from maxframe.liteframe.utils import normalize_col_name


def _normalize_keys(keys):
    """Normalize merge key arguments to lowercase."""
    if keys is None:
        return None
    if isinstance(keys, str):
        return normalize_col_name(keys)
    return [normalize_col_name(k) for k in keys]


def merge(
    self,
    right,
    how="inner",
    on=None,
    left_on=None,
    right_on=None,
    suffixes=("", "_y"),
    method="auto",
    auto_merge="both",
    auto_merge_threshold=8,
    bloom_filter="auto",
    bloom_filter_options=None,
):
    """
    Merge LiteFrame with another LiteFrame using a database-style join.

    Parameters
    ----------
    right : LiteFrame
        Object to merge with.
    how : {'left', 'right', 'outer', 'inner'}, default 'inner'
        Type of merge to perform.
    on : str or list of str, optional
        Column(s) to join on. Must exist in both LiteFrames.
    left_on : str or list of str, optional
        Column(s) from left LiteFrame to join on.
    right_on : str or list of str, optional
        Column(s) from right LiteFrame to join on.
    suffixes : tuple of (str, str), default ('', '_y')
        Suffixes for overlapping column names.
    method : {'auto', 'shuffle', 'broadcast'}, default 'auto'
        Merge execution strategy.
    auto_merge : {'both', 'none', 'before', 'after'}, default 'both'
        Auto merge small chunks before/after merge.
    auto_merge_threshold : int, default 8
        Chunk count threshold for triggering auto-merge.
    bloom_filter : bool or 'auto', default 'auto'
        Whether to use bloom filter optimization.
    bloom_filter_options : dict, optional
        Bloom filter configuration.

    Returns
    -------
    LiteFrame
        Merged result.
    """
    on = _normalize_keys(on)
    left_on = _normalize_keys(left_on)
    right_on = _normalize_keys(right_on)
    op = LiteFrameMerge(
        how=how,
        on=on,
        left_on=left_on,
        right_on=right_on,
        suffixes=suffixes,
        method=method,
        auto_merge=auto_merge,
        auto_merge_threshold=auto_merge_threshold,
        bloom_filter=bloom_filter,
        bloom_filter_options=bloom_filter_options,
    )
    return op(self, right)
