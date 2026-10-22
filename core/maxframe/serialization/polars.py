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

from typing import Dict, List

from maxframe.serialization.core import Serializer, buffered
from maxframe.utils.polars import (
    has_pl_object_type,
    pl_get_ext_type_metadata,
    pl_get_ext_type_name_and_storage,
    pl_make_ext_or_object_dtype,
)

try:
    import polars as pl
    import pyarrow as pa
except ImportError:
    pl = None
    pa = None


def _write_arrow_ipc(table):
    """Serialize a PyArrow Table to Arrow IPC bytes."""
    sink = pa.BufferOutputStream()
    writer = pa.RecordBatchStreamWriter(sink, table.schema)
    writer.write_table(table)
    writer.close()
    return bytes(sink.getvalue())


class _PolarsFrameSerializerBase(Serializer):
    """Base serializer for Polars DataFrame / LazyFrame.

    Subclasses must implement ``_wrap_result`` to control the return type
    (e.g. identity for DataFrame, ``df.lazy()`` for LazyFrame).
    """

    @classmethod
    def _wrap_result(cls, df):
        """Wrap a deserialized DataFrame into the target type."""
        raise NotImplementedError

    @buffered
    def serial(self, obj, context: Dict):
        incompatible_cols = [c for c in obj.columns if has_pl_object_type(obj[c].dtype)]
        if not incompatible_cols:
            # No Arrow-incompatible columns — original Arrow IPC path
            return [], [_write_arrow_ipc(obj.to_arrow())], True

        original_columns = list(obj.columns)
        # Collect metadata and values for each incompatible column.
        # Values are yielded as subs so the framework serializes them
        # recursively (e.g. AbstractExternalBlob → ExternalBlobSerializer).
        col_meta = {}
        col_values = []
        for c in incompatible_cols:
            ext_name, storage_str = pl_get_ext_type_name_and_storage(obj[c].dtype)
            metadata = pl_get_ext_type_metadata(obj[c].dtype)
            col_meta[c] = [ext_name, storage_str, metadata]
            col_values.append(obj[c].to_list())

        clean = obj.drop(incompatible_cols)
        arrow_buf = _write_arrow_ipc(clean.to_arrow())

        # subs[0] = Arrow IPC bytes, subs[1:] = value lists per column
        return (
            [original_columns, list(col_meta.items())],
            [arrow_buf] + col_values,
            False,
        )

    def deserial(self, serialized, context: Dict, subs: List):
        if isinstance(serialized, list) and len(serialized) == 2:
            original_columns, col_meta_items = serialized
            col_meta = dict(col_meta_items)

            arrow_bytes = subs[0]
            value_lists = subs[1:]

            reader = pa.RecordBatchStreamReader(pa.BufferReader(arrow_bytes))
            df = pl.from_arrow(reader.read_all())

            for i, (col_name, (ext_name, storage_str, metadata)) in enumerate(
                col_meta.items()
            ):
                dtype = pl_make_ext_or_object_dtype(ext_name, storage_str, metadata)
                series = pl.Series(col_name, value_lists[i], dtype=dtype)
                df = df.with_columns(series)

            return self._wrap_result(df.select(original_columns))

        reader = pa.RecordBatchStreamReader(pa.BufferReader(subs[0]))
        return self._wrap_result(pl.from_arrow(reader.read_all()))


class PolarsDataFrameSerializer(_PolarsFrameSerializerBase):
    @classmethod
    def _wrap_result(cls, df):
        return df


class PolarsLazyFrameSerializer(_PolarsFrameSerializerBase):
    @classmethod
    def _wrap_result(cls, df):
        return df.lazy()

    @buffered
    def serial(self, obj, context: Dict):
        return super().serial(obj.collect(), context)


class PolarsSeriesSerializer(Serializer):
    @buffered
    def serial(self, obj, context: Dict):
        if not has_pl_object_type(obj.dtype):
            # Non-object Series — original Arrow IPC path
            return [], [_write_arrow_ipc(obj.to_frame().to_arrow())], True

        # Arrow-incompatible Series: extract ext_name, storage, metadata, and values
        ext_name, storage_str = pl_get_ext_type_name_and_storage(obj.dtype)
        metadata = pl_get_ext_type_metadata(obj.dtype)
        values = obj.to_list()
        # subs[0] = value list (framework serializes recursively)
        return (
            [obj.name, ext_name, storage_str, metadata],
            [values],
            False,
        )

    def deserial(self, serialized, context: Dict, subs: List):
        if isinstance(serialized, list) and len(serialized) == 4:
            name, ext_name, storage_str, metadata = serialized
            dtype = pl_make_ext_or_object_dtype(ext_name, storage_str, metadata)
            return pl.Series(name, subs[0], dtype=dtype)

        reader = pa.RecordBatchStreamReader(pa.BufferReader(subs[0]))
        df = pl.from_arrow(reader.read_all())
        return df.to_series(0)


if pl is not None:
    PolarsDataFrameSerializer.register(pl.DataFrame)
    PolarsSeriesSerializer.register(pl.Series)
    PolarsLazyFrameSerializer.register(pl.LazyFrame)
