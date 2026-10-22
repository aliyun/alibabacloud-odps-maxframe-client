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
from maxframe.liteframe.operators.core import (
    LiteFrameDataFormat,
    LiteFrameOperator,
    LiteFrameOperatorMixin,
)
from maxframe.protocol import DefaultIndexType
from maxframe.serialization.serializables import (
    AnyField,
    BoolField,
    DictField,
    EnumField,
    FieldTypes,
    Float32Field,
    Int64Field,
    KeyField,
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
        if getattr(self, "read_stage", None) is not None:
            return self.new_scalar(None, dtype=np.dtype("O"))
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
            # Reject name collision: if the source already has a physical
            # __index__ column, silently dropping it would lose user data.
            if RANGE_COL_NAME in physical_dtypes.index:
                raise ValueError(
                    f"Column name collision: source already has a physical "
                    f"column named {RANGE_COL_NAME!r}, which conflicts with "
                    f"the virtual range index column. Rename the physical "
                    f"column or disable default_index_type=range."
                )
            range_columns = {RANGE_COL_NAME: RangeInfo(start=0, stop=shape[0], step=1)}
            # Range columns are visible, so include them in shape[1]
            shape = (shape[0], len(physical_dtypes) + 1)
        else:
            # Materialized range columns (e.g. a subdag/temp-table round-trip)
            # are read back as physical int64 columns, then reconstructed as
            # virtual range columns during execution. Here we only mark them
            # virtual so the visible schema and column order match the original
            # tileable; the per-chunk (start, stop, step) is inferred from the
            # data by the read executor.
            declared_range_cols = self.extra_params.get("range_columns")
            if declared_range_cols:
                present = [c for c in declared_range_cols if c in physical_dtypes.index]
                if present:
                    physical_dtypes = physical_dtypes.drop(present)
                    range_columns = {name: RangeInfo() for name in present}
                    # shape[1] already accounts for the range columns because
                    # they are part of self.dtypes, and they remain visible.

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


class LiteFrameLakeDataSource(LiteFrameColumnPruneMixin, LiteFrameDataSource):
    """File-based data source operator for Parquet, CSV, and JSON formats.

    A single operator with a ``data_format`` discriminator that selects
    the appropriate handler at execution and codegen time. Format-specific
    I/O parameters (``sep``, ``header``, ``names``, ``compression``,
    ``head_bytes``, ``lines``) are packed into ``read_kwargs``.
    """

    _op_type_ = opcodes.LITEFRAME_LAKE_DATA_SOURCE

    path = AnyField("path")
    storage_options = DictField("storage_options", default=None)
    read_kwargs = DictField("read_kwargs", default=None)
    read_stage = StringField("read_stage", default=None)
    is_partitioned = BoolField("is_partitioned", default=False)
    merge_small_files = BoolField("merge_small_files", default=True)
    merge_small_file_options = DictField("merge_small_file_options", default=None)
    memory_scale = Float32Field("memory_scale", default=None)
    chunk_bytes = StringField("chunk_bytes", default=None)
    # The data source's physical (original-case) column names, captured during
    # the get_dtypes stage. Predicate-pushdown pushers use it to map logical
    # (lowercase) column names to physical names before emitting case-sensitive
    # source filters (Lance SQL, PyArrow expressions). None when get_dtypes did
    # not run (no session), in which case pushdown is skipped to avoid case
    # mismatches.
    physical_columns = ListField("physical_columns", FieldTypes.string, default=None)

    # Format discriminator
    data_format = EnumField(
        "data_format",
        LiteFrameDataFormat,
        FieldTypes.int8,
        default=None,
        primitive=True,
    )

    # Column selection (all formats)
    columns = ListField("columns", default=None)

    # Parquet-specific fields
    filters = AnyField("filters", default=None)
    groups_as_chunks = BoolField("groups_as_chunks", default=True)

    # CSV/JSON dtype overrides
    dtype = DictField("dtype", default=None)

    def _prune_column_field(self, field_name, new_cols):
        """Filter a column-selection field to only keep names in *new_cols*.

        Comparison is case-insensitive because *new_cols* comes from the
        lowercased ``dtypes.index`` while the field value may retain the
        original file column casing needed by the reader.
        """
        value = getattr(self, field_name, None)
        if value is not None:
            lowered_set = {c.lower() for c in new_cols}
            setattr(self, field_name, [c for c in value if c.lower() in lowered_set])

    def set_pruned_columns(self, columns):
        new_cols = list(columns)
        if self.default_index_type == DefaultIndexType.range:
            if RANGE_COL_NAME in new_cols:
                new_cols = [c for c in new_cols if c != RANGE_COL_NAME]
            else:
                self.default_index_type = None
        self.dtypes = self.dtypes[new_cols]
        self._prune_column_field("columns", new_cols)
        # Materialize [] so a None columns field is not read as "unspecified"
        # by the Lake/Lance reader (which would restore the full file width).
        if not new_cols and self.columns is None:
            self.columns = []
        if self.dtype is not None:
            lowered = {c.lower() for c in new_cols}
            self.dtype = {k: v for k, v in self.dtype.items() if k.lower() in lowered}


class LiteFrameReadODPSTable(LiteFrameColumnPruneMixin, LiteFrameDataSource):
    _op_type_ = opcodes.READ_ODPS_TABLE

    table_name = StringField("table_name")
    partitions = ListField("partitions", default=None)
    columns = AnyField("columns", default=None)
    # Physical ODPS table column names for the requested logical columns
    # (``dtypes.index`` / ``columns``), used when table column names differ
    # from the logical names (e.g. an int-string ``"0"`` is stored as the
    # ODPS-legal ``_column_0``).  Positionally aligned with ``dtypes.index``;
    # ``None`` means the table names are unchanged.
    table_column_names = ListField(
        "table_column_names", FieldTypes.string, default=None
    )
    # Whether partition spec columns (e.g. ``pt`` for ``pt=00``) are appended
    # as physical columns in the output.  ``partitions`` only selects which
    # partitions to read; this flag decides whether their spec columns appear
    # in the planned ``dtypes`` (and thus the materialized result).
    append_partitions = BoolField("append_partitions", default=None)
    # Accepts either a SQL filter string or List[List[Tuple]] (CNF format:
    # outer OR, inner AND) consumed by convert_filters_to_sql.
    filters = AnyField("filters", default=None)

    def can_fuse_with_custom_code(self) -> bool:
        return False

    def set_pruned_columns(self, columns):
        new_cols = list(columns)
        new_cols_set = set(new_cols)
        # Handle default range index column pruning: RANGE_COL_NAME is virtual,
        # not in self.dtypes
        if self.default_index_type == DefaultIndexType.range:
            if RANGE_COL_NAME not in new_cols_set:
                self.default_index_type = None
        # Drop materialized range columns that were pruned out so they are
        # neither read from the table nor reconstructed as virtual columns.
        # Range column info (dict of {name: RangeInfo}) is carried as an
        # extra_param by the io_replacer, not as a declared field.
        range_columns = self.extra_params.get("range_columns")
        if range_columns:
            range_columns = {
                k: v for k, v in range_columns.items() if k in new_cols_set
            } or None
            self.extra_params["range_columns"] = range_columns
        range_set = set((range_columns or {}).keys())
        # Filter to physical columns only (exclude all range column names)
        physical_cols = [
            c for c in new_cols if c != RANGE_COL_NAME and c not in range_set
        ]
        # Capture the full (pre-pruning) column list: table_column_names is
        # positionally aligned with it, so non-sequential pruning below can
        # map each kept column back to its original index.
        full_dtypes_index = self.dtypes.index.tolist()
        self.dtypes = self.dtypes[physical_cols]
        cur_columns = self.columns
        if cur_columns is None:
            cur_columns = physical_cols
        self.columns = [c for c in cur_columns if c in new_cols_set]
        # Keep table_column_names positionally aligned with the (pruned)
        # logical columns.  When columns were declared explicitly the physical
        # names share their order; otherwise map each kept column back to its
        # position in the full column list so non-sequential pruning (e.g.
        # keeping columns at indices [0, 3, 5]) stays correctly aligned.
        if self.table_column_names is not None:
            if len(cur_columns) == len(self.table_column_names):
                index_by_logical = {logical: i for i, logical in enumerate(cur_columns)}
                kept = [
                    index_by_logical[c] for c in self.columns if c in index_by_logical
                ]
                self.table_column_names = [self.table_column_names[i] for i in kept]
            else:
                table_column_names = self.table_column_names
                index_by_logical = {
                    logical: i for i, logical in enumerate(full_dtypes_index)
                }
                self.table_column_names = [
                    table_column_names[index_by_logical[c]]
                    if c in index_by_logical
                    else c
                    for c in self.dtypes.index
                ]
        # Update hidden_columns to only include those still present
        if self.hidden_columns:
            remaining_hidden = [c for c in self.hidden_columns if c in new_cols_set]
            self.hidden_columns = remaining_hidden or None


class LiteFrameReadLance(LiteFrameLakeDataSource):
    _op_type_ = opcodes.READ_LANCE

    columns = ListField("columns", default=None)
    version = Int64Field("version", default=None)
    asof = StringField("asof", default=None)
    with_row_address = BoolField("with_row_address", default=False)


class LiteFrameFromLocal(LiteFrameDataSource):
    _op_type_ = opcodes.LITEFRAME_FROM_LOCAL

    data = AnyField("data")


class LiteFrameFromTensor(LiteFrameDataSource):
    """Create a LiteFrame from a MaxFrame tensor.

    The tensor may be 1-d (producing a single-column LiteFrame) or 2-d
    (producing a multi-column LiteFrame, one column per tensor column).
    """

    _op_type_ = opcodes.LITEFRAME_FROM_TENSOR

    input = KeyField("input")

    @classmethod
    def _set_inputs(cls, op: "LiteFrameFromTensor", inputs):
        super()._set_inputs(op, inputs)
        op.input = op.inputs[0]

    def __call__(self, input_tensor):
        shape = (input_tensor.shape[0], len(self.dtypes))
        return self.new_liteframe(
            [input_tensor],
            shape=shape,
            physical_dtypes=self.dtypes,
        )
