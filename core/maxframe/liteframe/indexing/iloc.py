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
from typing import List

import numpy as np
import pandas as pd
import pyarrow as pa

from maxframe import opcodes
from maxframe.config import options
from maxframe.core import EntityData
from maxframe.liteframe.arithmetic.core import (
    _build_fused_projection,
    _resolve_named_expr,
)
from maxframe.liteframe.core import FrameMetadata, RangeInfo
from maxframe.liteframe.operators.core import LiteFrameOperator, LiteFrameOperatorMixin
from maxframe.serialization.serializables import AnyField, KeyField, ListField
from maxframe.tensor.indexing.core import calc_shape
from maxframe.utils import wrap_arrow_dtype


class LiteFrameHeadTailOptimizedMixin(LiteFrameOperatorMixin):
    """Mixin for LiteFrame operators that can be optimized by head pushdown."""

    __slots__ = ()

    @classmethod
    def _is_head(cls, row_index):
        return (
            isinstance(row_index, slice)
            and (row_index.start is None or row_index.start == 0)
            and row_index.stop is not None
            and row_index.stop > 0
            and (row_index.step is None or row_index.step == 1)
        )

    def can_be_optimized(self):
        row_index = self.indexes[0] if self.indexes else None
        return (
            self._is_head(row_index)
            and row_index.stop <= options.optimize.head_optimize_threshold
        )


class LiteFrameIlocGetItem(LiteFrameHeadTailOptimizedMixin, LiteFrameOperator):
    _op_type_ = opcodes.LITEFRAME_ILOC_GETITEM

    _input = KeyField("input")
    indexes = ListField("indexes", default=None)

    @property
    def input(self):
        return self._input

    @classmethod
    def _set_inputs(cls, op, inputs: List[EntityData]):
        super()._set_inputs(op, inputs)
        op._input = op._inputs[0]

    @property
    def row_index(self):
        """Convenience accessor for the row index (indexes[0])."""
        return self.indexes[0] if self.indexes else None

    def __call__(self, liteframe):
        inp = liteframe
        n_rows = inp.shape[0]
        row_index = self.indexes[0]

        # When row count is unknown (np.nan), we cannot compute exact output size
        shape_unknown = n_rows is None or (
            isinstance(n_rows, float) and math.isnan(n_rows)
        )

        # Compute output row count
        if isinstance(row_index, int):
            out_nrows = 1
        elif isinstance(row_index, slice):
            if shape_unknown:
                out_nrows = np.nan
            else:
                out_nrows = calc_shape((n_rows,), (row_index,))[0]
        else:
            raise TypeError(
                f"LiteFrameIlocGetItem only supports int or slice row_index, got {type(row_index)}"
            )

        # Compute output frame_metadata
        input_metadata = inp.frame_metadata
        out_frame_metadata = None
        out_dtypes = inp._data._physical_dtypes  # physical dtypes only, no range cols

        if input_metadata is not None and input_metadata.range_columns:
            input_range_cols = input_metadata.range_columns
            if isinstance(row_index, int):
                if shape_unknown:
                    # Cannot normalize index when row count is unknown
                    out_frame_metadata = FrameMetadata(range_columns={})
                    materialized_dtypes = {
                        rc_name: wrap_arrow_dtype(pa.int64())
                        for rc_name in input_range_cols
                    }
                    if materialized_dtypes:
                        mat_series = pd.Series(materialized_dtypes)
                        out_dtypes = pd.concat([mat_series, out_dtypes])
                else:
                    n_rows_int = int(n_rows)
                    # Single row: range becomes a single-element range
                    normalized = row_index if row_index >= 0 else n_rows_int + row_index
                    new_range_cols = {}
                    for rc_name, rc_info in input_range_cols.items():
                        new_range_cols[rc_name] = RangeInfo(
                            start=normalized, stop=normalized + 1, step=1
                        )
                    out_frame_metadata = FrameMetadata(range_columns=new_range_cols)
            elif isinstance(row_index, slice):
                if shape_unknown:
                    # Cannot compute range shift when row count is unknown
                    out_frame_metadata = FrameMetadata(range_columns={})
                    materialized_dtypes = {
                        rc_name: wrap_arrow_dtype(pa.int64())
                        for rc_name in input_range_cols
                    }
                    if materialized_dtypes:
                        mat_series = pd.Series(materialized_dtypes)
                        out_dtypes = pd.concat([mat_series, out_dtypes])
                else:
                    n_rows_int = int(n_rows)
                    start, stop, step = row_index.indices(n_rows_int)
                    if step == 1:
                        # Contiguous slice: shift range start/stop
                        new_range_cols = {}
                        for rc_name, rc_info in input_range_cols.items():
                            new_range_cols[rc_name] = RangeInfo(
                                start=rc_info.start + start
                                if rc_info.start is not None
                                else start,
                                stop=rc_info.start + stop
                                if rc_info.start is not None
                                else stop,
                                step=1,
                            )
                        out_frame_metadata = FrameMetadata(range_columns=new_range_cols)
                    else:
                        # Non-contiguous slice: materialize range columns
                        materialized_dtypes = {}
                        for rc_name in input_range_cols:
                            materialized_dtypes[rc_name] = wrap_arrow_dtype(pa.int64())
                        if materialized_dtypes:
                            mat_series = pd.Series(materialized_dtypes)
                            out_dtypes = pd.concat([mat_series, out_dtypes])
                        out_frame_metadata = FrameMetadata(range_columns={})

        n_range = (
            len(out_frame_metadata.range_columns)
            if out_frame_metadata and out_frame_metadata.range_columns
            else 0
        )
        shape = (out_nrows, len(out_dtypes) + n_range)

        # Preserve hidden columns
        hidden_columns = None
        if input_metadata is not None and input_metadata.hidden_columns:
            hidden_columns = input_metadata.hidden_columns

        if hidden_columns:
            if out_frame_metadata is not None:
                out_frame_metadata = FrameMetadata(
                    range_columns=out_frame_metadata.range_columns,
                    hidden_columns=hidden_columns,
                )
            else:
                out_frame_metadata = FrameMetadata(hidden_columns=hidden_columns)

        return self.new_liteframe(
            [inp],
            shape=shape,
            physical_dtypes=out_dtypes,
            frame_metadata=out_frame_metadata,
        )


def _col_index_touches_range(col_index, n_range, n_logical):
    """Check whether a column index targets any virtual range column.

    Parameters
    ----------
    col_index : int, slice, or list
        The column indexer (already normalized by process_iloc_indexes_liteframe).
    n_range : int
        Number of virtual range columns.
    n_logical : int
        Total logical columns (range + physical).
    """
    if n_range == 0:
        return False
    if isinstance(col_index, Integral):
        normalized = col_index if col_index >= 0 else col_index + n_logical
        return normalized < n_range
    elif isinstance(col_index, slice):
        if col_index == slice(None):
            # slice(None) selects all columns but the executor shifts it
            # past range columns without materializing them.
            return False
        start, stop, step = col_index.indices(n_logical)
        if step > 0:
            return start < n_range
        else:
            return stop + 1 < n_range
    else:
        # list
        normalized = [i if i >= 0 else i + n_logical for i in col_index]
        return any(i < n_range for i in normalized)


class LiteFrameIlocSetItem(LiteFrameOperatorMixin, LiteFrameOperator):
    _op_type_ = opcodes.LITEFRAME_ILOC_SETITEM

    indexes = ListField("indexes", default=None)
    value = AnyField("value", default=None)

    def __call__(self, liteframe):
        inp = liteframe
        out_metadata = inp.frame_metadata

        # If col_index targets a virtual range column, the executor materializes
        # all range columns into physical ones, so the output metadata must
        # reflect that by clearing range_columns.
        _, col_index = self.indexes
        if out_metadata and out_metadata.range_columns:
            n_range = len(out_metadata.range_columns)
            n_logical = n_range + len(inp._data._physical_dtypes)
            touches_range = _col_index_touches_range(col_index, n_range, n_logical)
            if touches_range:
                # Range columns become physical; clear range metadata and
                # add their Int64 dtypes to physical_dtypes.
                materialized_dtypes = {
                    rc_name: wrap_arrow_dtype(pa.int64())
                    for rc_name in out_metadata.range_columns
                }
                mat_series = pd.Series(materialized_dtypes)
                new_phys = pd.concat([mat_series, inp._data._physical_dtypes])
                out_metadata = FrameMetadata(
                    range_columns={},
                    hidden_columns=out_metadata.hidden_columns,
                )
                return self.new_liteframe(
                    [inp],
                    shape=inp.shape,
                    physical_dtypes=new_phys,
                    frame_metadata=out_metadata,
                )

        return self.new_liteframe(
            [inp],
            shape=inp.shape,
            physical_dtypes=inp._data._physical_dtypes,
            frame_metadata=out_metadata,
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
    """Positional indexer for LiteFrame."""

    def __init__(self, lf):
        self._obj = lf

    def __getitem__(self, indexes):
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
