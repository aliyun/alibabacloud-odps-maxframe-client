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

from maxframe import opcodes
from maxframe.liteframe.core import RANGE_COL_NAME, FrameMetadata, RangeInfo
from maxframe.liteframe.operators.core import LiteFrameOperator, LiteFrameOperatorMixin
from maxframe.protocol import DefaultIndexType
from maxframe.serialization.serializables import (
    AnyField,
    EnumField,
    FieldTypes,
    Int64Field,
    ListField,
    SeriesField,
    StringField,
)


class LiteFrameDataSource(LiteFrameOperatorMixin, LiteFrameOperator):
    __slots__ = ()

    dtypes = SeriesField("dtypes")
    nrows = Int64Field("nrows", default=None)
    default_index_type = EnumField(
        "default_index_type",
        DefaultIndexType,
        FieldTypes.int8,
        default=None,
        primitive=True,
    )
    hidden_columns = ListField("hidden_columns", FieldTypes.string, default=None)

    def _new_liteframe_from_source(self, shape=None, raw_chunk_size=None):
        if shape is None:
            shape = (np.nan, len(self.dtypes))
        # Apply nrows limit if set by head pushdown optimization
        if self.nrows is not None:
            n = shape[0]
            if n is not None and not (isinstance(n, float) and np.isnan(n)):
                shape = (min(int(n), self.nrows), shape[1])
            else:
                shape = (self.nrows, shape[1])
        range_columns = None
        physical_dtypes = self.dtypes
        if self.default_index_type == DefaultIndexType.range:
            range_columns = {RANGE_COL_NAME: RangeInfo(start=0, stop=shape[0], step=1)}
            # Range columns are virtual; exclude them from physical_dtypes
            # to avoid overlap between range and physical columns.
            if RANGE_COL_NAME in physical_dtypes.index:
                physical_dtypes = physical_dtypes.drop(RANGE_COL_NAME)
            # Range columns are visible, so include them in shape[1]
            shape = (shape[0], len(physical_dtypes) + 1)

        hidden_columns = self.hidden_columns if self.hidden_columns else None
        frame_metadata = None
        if range_columns or hidden_columns:
            frame_metadata = FrameMetadata(
                range_columns=range_columns,
                hidden_columns=hidden_columns,
            )

        kw = dict(
            shape=shape,
            physical_dtypes=physical_dtypes,
            frame_metadata=frame_metadata,
        )
        if raw_chunk_size is not None:
            kw["raw_chunk_size"] = raw_chunk_size
        return self.new_liteframe([], **kw)


class LiteFrameColumnPruneMixin:
    def get_columns(self):
        return list(self.dtypes.index)

    def set_pruned_columns(self, columns):
        raise NotImplementedError


class LiteFrameReadODPSTable(LiteFrameColumnPruneMixin, LiteFrameDataSource):
    _op_type_ = opcodes.READ_ODPS_TABLE

    table_name = StringField("table_name")
    partitions = ListField("partitions", default=None)
    columns = AnyField("columns", default=None)

    def can_fuse_with_custom_code(self) -> bool:
        return False

    def set_pruned_columns(self, columns):
        new_cols = list(columns)
        new_cols_set = set(new_cols)
        # Handle range column pruning: RANGE_COL_NAME is virtual, not in self.dtypes
        if self.default_index_type == DefaultIndexType.range:
            if RANGE_COL_NAME not in new_cols_set:
                self.default_index_type = None
        # Filter to physical columns only (exclude range column name)
        physical_cols = [c for c in new_cols if c != RANGE_COL_NAME]
        self.dtypes = self.dtypes[physical_cols]
        self.columns = [c for c in (self.columns or []) if c in new_cols_set]
        # Update hidden_columns to only include those still present
        if self.hidden_columns:
            remaining_hidden = [c for c in self.hidden_columns if c in new_cols_set]
            self.hidden_columns = remaining_hidden or None


class LiteFrameFromLocal(LiteFrameDataSource):
    _op_type_ = opcodes.LITEFRAME_FROM_LOCAL

    data = AnyField("data")
