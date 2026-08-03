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

try:
    import polars as pl
    import pyarrow as pa
except ImportError:
    pl = None
    pa = None


class PolarsDataFrameSerializer(Serializer):
    @buffered
    def serial(self, obj, context: Dict):
        # pl.DataFrame -> pa.Table (zero-copy) -> Arrow IPC buffer
        arrow_table = obj.to_arrow()
        sink = pa.BufferOutputStream()
        writer = pa.RecordBatchStreamWriter(sink, arrow_table.schema)
        writer.write_table(arrow_table)
        writer.close()
        return [], [sink.getvalue()], True

    def deserial(self, serialized: List, context: Dict, subs: List):
        reader = pa.RecordBatchStreamReader(pa.BufferReader(subs[0]))
        return pl.from_arrow(reader.read_all())


class PolarsSeriesSerializer(Serializer):
    @buffered
    def serial(self, obj, context: Dict):
        # Series -> single-column DataFrame -> Arrow IPC
        arrow_table = obj.to_frame().to_arrow()
        sink = pa.BufferOutputStream()
        writer = pa.RecordBatchStreamWriter(sink, arrow_table.schema)
        writer.write_table(arrow_table)
        writer.close()
        return [], [sink.getvalue()], True

    def deserial(self, serialized: List, context: Dict, subs: List):
        reader = pa.RecordBatchStreamReader(pa.BufferReader(subs[0]))
        df = pl.from_arrow(reader.read_all())
        return df.to_series(0)


class PolarsLazyFrameSerializer(Serializer):
    @buffered
    def serial(self, obj, context: Dict):
        # Collect LazyFrame -> DataFrame, then Arrow IPC
        collected = obj.collect()
        arrow_table = collected.to_arrow()
        sink = pa.BufferOutputStream()
        writer = pa.RecordBatchStreamWriter(sink, arrow_table.schema)
        writer.write_table(arrow_table)
        writer.close()
        return [], [sink.getvalue()], True

    def deserial(self, serialized: List, context: Dict, subs: List):
        reader = pa.RecordBatchStreamReader(pa.BufferReader(subs[0]))
        df = pl.from_arrow(reader.read_all())
        return df.lazy()


if pl is not None:
    PolarsDataFrameSerializer.register(pl.DataFrame)
    PolarsSeriesSerializer.register(pl.Series)
    PolarsLazyFrameSerializer.register(pl.LazyFrame)
