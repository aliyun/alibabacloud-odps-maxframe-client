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

import math
from numbers import Integral

from maxframe.liteframe.arithmetic.core import (
    _build_fused_projection,
    _resolve_named_expr,
)
from maxframe.liteframe.operators.indexing import (
    LiteFrameIlocGetItem,
    LiteFrameIlocSetItem,
)


def process_iloc_indexes_liteframe(inp, indexes):
    """Validate and normalize iloc indexes for LiteFrame.

    Supports int, slice, and list indexes (no ndarray/bool).
    Returns a list of normalized indexes with length 2: [row_index, col_index].
    """
    if not isinstance(indexes, tuple):
        indexes = (indexes, slice(None))
    if len(indexes) > 2:
        raise IndexError("Too many indexers")
    if len(indexes) < 2:
        indexes = indexes + (slice(None),) * (2 - len(indexes))

    new_indexes = []
    for ax, index in enumerate(indexes):
        if isinstance(index, Integral):
            shape = inp.shape[ax]
            if not (isinstance(shape, float) and math.isnan(shape)):
                if index < -shape or index >= shape:
                    raise IndexError("single positional indexer is out-of-bounds")
            new_indexes.append(index)
        elif isinstance(index, slice):
            new_indexes.append(index)
        elif isinstance(index, list):
            shape = inp.shape[ax]
            if not (isinstance(shape, float) and math.isnan(shape)):
                for i in index:
                    if not isinstance(i, Integral):
                        raise ValueError(
                            "Location based indexing can only have [integer, integer slice] types"
                        )
                    if i < -shape or i >= shape:
                        raise IndexError("positional indexer is out-of-bounds")
            new_indexes.append(index)
        else:
            raise ValueError(
                "Location based indexing can only have [integer, integer slice] types"
            )
    return new_indexes


class LiteFrameIloc:
    """Purely positional indexer for LiteFrame, accessed via ``lf.iloc``.

    Supports integer, slice, and list indexers on both the row and column
    axes (e.g. ``lf.iloc[0]``, ``lf.iloc[:, 1:3]``, ``lf.iloc[[0, 2], 0]``).
    Indexers are 0-based and support negative offsets.  Only integer-based
    indexers are accepted — boolean masks and label-based selection are not
    supported here.  Row order follows the supplied indexers; column
    selection happens before row selection.
    """

    def __init__(self, lf):
        self._obj = lf

    def __getitem__(self, indexes):
        """Select rows and/or columns by positional index.

        Indexers are validated and normalized to a two-element
        ``[row_index, col_index]`` list.  Column selection is applied first
        (as a projection), then row selection.  A fully-open selection
        (``slice(None)`` on both axes) returns the frame unchanged.

        Parameters
        ----------
        indexes : int, slice, list, or tuple
            Positional indexer(s).  A single indexer selects rows (all
            columns implied); a two-element tuple selects
            ``(rows, columns)``.  Each axis accepts an integer, a slice, or
            a list of integers.  Negative offsets are supported.

        Returns
        -------
        LiteFrame
            LiteFrame restricted to the selected rows and columns.

        See Also
        --------
        LiteFrame.select : Explicit column selection.
        """
        processed = process_iloc_indexes_liteframe(self._obj, indexes)
        row_idx, col_idx = processed

        # No-op: both axes select everything
        if row_idx == slice(None) and col_idx == slice(None):
            return self._obj

        # Column selection first (via existing LiteFrameProjection)
        target = self._obj
        if col_idx != slice(None):
            if isinstance(col_idx, slice):
                columns = list(self._obj.columns)[col_idx]
            elif isinstance(col_idx, list):
                columns = [self._obj.columns[i] for i in col_idx]
            else:
                columns = [self._obj.columns[col_idx]]

            projections = [
                _resolve_named_expr(self._obj, c, self._obj.dtypes[c]) for c in columns
            ]
            target = _build_fused_projection(self._obj, projections)

        # Row selection (via LiteFrameIlocGetItem)
        if row_idx != slice(None):
            op = LiteFrameIlocGetItem(indexes=[row_idx, slice(None)])
            target = op(target)

        return target

    def __setitem__(self, indexes, value):
        if not isinstance(value, (int, float, str, bool)) and value is not None:
            raise NotImplementedError("Only scalar value is supported to set by iloc")

        processed = process_iloc_indexes_liteframe(self._obj, indexes)
        op = LiteFrameIlocSetItem(indexes=processed, value=value)
        result = op(self._obj)
        self._obj._data.detach(self._obj)
        self._obj._data = result.data
        result.data.attach(self._obj)
