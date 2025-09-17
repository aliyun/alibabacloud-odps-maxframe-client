# Copyright 1999-2025 Alibaba Group Holding Ltd.
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

from ..utils import arrow_type_from_str, extract_class_name
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
_TYPE_CHAR_ARROW_REDUCED = "R"


class ArrowDataTypeSerializer(Serializer):
    def serial(self, obj: pa.DataType, context):
        return [str(obj)], [], True

    def deserial(self, serialized, context, subs):
        return arrow_type_from_str(serialized[0])


class ArrowArraySerializer(Serializer):
    @buffered
    def serial(self, obj: PA_ARRAY_TYPES, context: Dict):
        if not isinstance(obj, (pa.Array, pa.ChunkedArray)):
            raise NotImplementedError(f"Array type {type(obj)} not supported")

        if obj.type.num_fields == 0:
            # use legacy serialization in case arrow changes deserializer method
            data_type = str(obj.type)
            if isinstance(obj, pa.Array):
                array_type = _TYPE_CHAR_ARROW_ARRAY
                buffers = obj.buffers()
                sizes = len(obj)
            else:  # ChunkedArray
                array_type = _TYPE_CHAR_ARROW_CHUNKED_ARRAY
                buffers = [c.buffers() for c in obj.chunks]
                sizes = [len(c) for c in obj.chunks]
            return [array_type, data_type, sizes], buffers, False

        meth, extracted = obj.__reduce__()
        meth_name = extract_class_name(meth)
        return [_TYPE_CHAR_ARROW_REDUCED, meth_name, None], list(extracted), False

    def deserial(self, serialized: List, context: Dict, subs: List):
        array_type, data_type_str, sizes = serialized[:3]
        if array_type == _TYPE_CHAR_ARROW_REDUCED:
            if data_type_str == "pyarrow.lib#chunked_array":
                return pa.chunked_array(*subs)
            elif data_type_str == "pyarrow.lib#_restore_array":
                return pa.lib._restore_array(*subs)
            else:
                raise NotImplementedError(f"Unknown array type: {array_type}")
        if array_type == _TYPE_CHAR_ARROW_ARRAY:
            data_type = arrow_type_from_str(data_type_str)
            return pa.Array.from_buffers(data_type, sizes, subs)
        elif array_type == _TYPE_CHAR_ARROW_CHUNKED_ARRAY:
            data_type = arrow_type_from_str(data_type_str)
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
    ArrowDataTypeSerializer.register(pa.DataType)
    ArrowArraySerializer.register(pa.Array)
    ArrowArraySerializer.register(pa.ChunkedArray)
    ArrowBatchSerializer.register(pa.Table)
    ArrowBatchSerializer.register(pa.RecordBatch)
