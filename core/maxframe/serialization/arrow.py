# Copyright 1999-2024 Alibaba Group Holding Ltd.
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

from typing import Any, Dict, List, Union

from ..utils import arrow_type_from_str
from .core import Serializer, buffered

try:
    import pyarrow as pa

    PA_ARRAY_TYPES = Union[pa.Array, pa.ChunkedArray]
    PA_RECORD_TYPES = Union[pa.Table, pa.RecordBatch]
except ImportError:  # pragma: no cover
    pa = None
    PA_ARRAY_TYPES = Any
    PA_RECORD_TYPES = Any


_TYPE_CHAR_ARROW_ARRAY = "A"
_TYPE_CHAR_ARROW_CHUNKED_ARRAY = "C"


class ArrowArraySerializer(Serializer):
    @buffered
    def serial(self, obj: PA_ARRAY_TYPES, context: Dict):
        data_type = str(obj.type)
        if isinstance(obj, pa.Array):
            array_type = _TYPE_CHAR_ARROW_ARRAY
            buffers = obj.buffers()
            sizes = len(obj)
        elif isinstance(obj, pa.ChunkedArray):
            array_type = _TYPE_CHAR_ARROW_CHUNKED_ARRAY
            buffers = [c.buffers() for c in obj.chunks]
            sizes = [len(c) for c in obj.chunks]
        else:  # pragma: no cover
            raise NotImplementedError(f"Array type {type(obj)} not supported")
        return [array_type, data_type, sizes], buffers, True

    def deserial(self, serialized: List, context: Dict, subs: List):
        array_type, data_type_str, sizes = serialized[:3]
        data_type = arrow_type_from_str(data_type_str)
        if array_type == _TYPE_CHAR_ARROW_ARRAY:
            return pa.Array.from_buffers(data_type, sizes, subs)
        elif array_type == _TYPE_CHAR_ARROW_CHUNKED_ARRAY:
            chunks = [
                pa.Array.from_buffers(data_type, size, bufs)
                for size, bufs in zip(sizes, subs)
            ]
            return pa.chunked_array(chunks)
        else:  # pragma: no cover
            raise NotImplementedError(f"Array type {array_type} not supported")


class ArrowBatchSerializer(Serializer):
    @buffered
    def serial(self, obj: PA_RECORD_TYPES, context: Dict):
        sink = pa.BufferOutputStream()
        writer = pa.RecordBatchStreamWriter(sink, obj.schema)
        if isinstance(obj, pa.Table):
            batch_type = "T"
            writer.write_table(obj)
        else:
            batch_type = "B"
            writer.write_batch(obj)
        writer.close()

        buf = sink.getvalue()
        buffers = [buf]
        return [batch_type], buffers, True

    def deserial(self, serialized: List, context: Dict, subs: List):
        reader = pa.RecordBatchStreamReader(pa.BufferReader(subs[0]))
        if serialized[0] == "T":  # Table
            return reader.read_all()
        else:  # RecordBatch
            return reader.read_next_batch()


if pa is not None:  # pragma: no branch
    ArrowArraySerializer.register(pa.Array)
    ArrowArraySerializer.register(pa.ChunkedArray)
    ArrowBatchSerializer.register(pa.Table)
    ArrowBatchSerializer.register(pa.RecordBatch)
