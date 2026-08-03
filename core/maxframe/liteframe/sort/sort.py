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

from maxframe.liteframe.operators.sort import LiteFrameSort
from maxframe.liteframe.utils import normalize_col_name


def sort(
    self,
    by=None,
    ascending=True,
    kind="quicksort",
    na_position="last",
    parallel_kind="PSRS",
    psrs_kinds=None,
):
    """Sort the LiteFrame by the specified columns.

    Parameters
    ----------
    by : str, list of str, or None
        Column name(s) to sort by. If None, all columns are used.
    ascending : bool or list of bool, default True
        Sort ascending vs descending. Specify list for multiple sort orders.
    kind : str, default "quicksort"
        Sort algorithm hint (Polars chooses its own algorithm).
    na_position : {"first", "last"}, default "last"
        Where to place null values in the sort order.
    parallel_kind : str, default "PSRS"
        Parallel sort algorithm. Only "PSRS" is supported.
    psrs_kinds : list of str or None
        Sort algorithm for each PSRS stage.

    Returns
    -------
    LiteFrame
        Sorted LiteFrame.
    """
    if na_position not in ("first", "last"):
        raise ValueError(
            f"Invalid na_position '{na_position}', must be 'first' or 'last'"
        )
    if parallel_kind != "PSRS":
        raise ValueError(
            f"parallel_kind '{parallel_kind}' is not supported, only 'PSRS'"
        )
    if psrs_kinds is not None and len(psrs_kinds) != 3:
        raise ValueError(
            f"psrs_kinds must have exactly 3 elements, got {len(psrs_kinds)}"
        )

    # Resolve by=None to all columns
    if by is None:
        by = list(self._data.dtypes.index)
    else:
        if isinstance(by, str):
            by = [by]
        raw_by = by
        by = [normalize_col_name(b) for b in by]

        # Validate columns exist, report original name on error
        for raw_name, norm_name in zip(raw_by, by):
            if norm_name not in self._data.dtypes.index:
                raise KeyError(f"Column '{raw_name}' not found in LiteFrame")

    if isinstance(ascending, list) and len(ascending) != len(by):
        raise ValueError(
            f"ascending length {len(ascending)} does not match by length {len(by)}"
        )

    op = LiteFrameSort(
        by=by,
        ascending=ascending,
        kind=kind,
        na_position=na_position,
        parallel_kind=parallel_kind,
        psrs_kinds=psrs_kinds,
    )
    return op(self)


sort_values = sort
