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

import io
import struct
import zlib
from typing import Any, List, Tuple, Type, Union, overload

import msgpack
import numpy as np
import pandas as pd

_MSG_EXT_DATA_TYPE = 43


def on_serialize_shape(shape: Tuple[int]):
    def _to_shape_num(x):
        if np.isnan(x):
            return -1
        if isinstance(x, np.generic):
            return x.item()
        return x

    if shape:
        return tuple(_to_shape_num(s) for s in shape)
    return shape


def on_deserialize_shape(shape: Tuple[int]):
    if shape:
        return tuple(s if s != -1 else np.nan for s in shape)
    return shape


def on_serialize_numpy_type(value: np.dtype):
    if value is pd.NaT:
        value = None
    return value.item() if isinstance(value, np.generic) else value


def on_serialize_nsplits(value: Tuple[Tuple[int]]):
    if value is None:
        return None
    new_nsplits = []
    for dim_splits in value:
        new_nsplits.append(tuple(None if pd.isna(v) else v for v in dim_splits))
    return tuple(new_nsplits)


@overload
def serialize_serializable(serializable, compress: bool = False) -> bytes:
    ...


@overload
def serialize_serializable(
    serializable,
    compress: bool = False,
    with_ext_data: bool = True,
    allowed_ext_types: Tuple[Type] = None,
) -> Tuple[bytes, List[Any]]:
    ...


def serialize_serializable(
    serializable,
    compress: bool = False,
    with_ext_data: bool = False,
    allowed_ext_types: Union[Type, Tuple[Type]] = None,
):
    from ..serialization import serialize

    bio = io.BytesIO()
    header, buffers = serialize(serializable)
    buf_sizes = [getattr(buf, "nbytes", len(buf)) for buf in buffers]
    header[0]["buf_sizes"] = buf_sizes

    ext_dict, ext_list = {}, []

    def encode_ext(obj):
        if isinstance(obj, np.generic) and obj.shape == () and not np.isnan(obj):
            return obj.item()
        elif with_ext_data:
            if allowed_ext_types and not isinstance(obj, allowed_ext_types):
                return obj

            # Try reusing stored objects
            try:
                ret_id = ext_dict[id(obj)]
            except KeyError:
                ret_id = ext_dict[id(obj)] = len(ext_list)
                ext_list.append(obj)
            return msgpack.ExtType(_MSG_EXT_DATA_TYPE, struct.pack("<I", ret_id))
        else:
            return obj

    s_header = msgpack.dumps(header, default=encode_ext)

    bio.write(struct.pack("<Q", len(s_header)))
    bio.write(s_header)
    for buf in buffers:
        bio.write(buf)
    ser_graph = bio.getvalue()

    if compress:
        ser_graph = zlib.compress(ser_graph)
    if with_ext_data:
        return ser_graph, ext_list
    return ser_graph


def deserialize_serializable(ser_serializable: bytes, ext_data: List[Any] = None):
    from ..serialization import deserialize

    def on_ext_data(code: int, data: bytes):
        if code == _MSG_EXT_DATA_TYPE:
            return ext_data[struct.unpack("<I", data)[0]]
        else:  # pragma: no cover
            raise ValueError("Unknown ext type %s" % code)

    try:
        decompressed_data = zlib.decompress(ser_serializable)
        data_to_process = decompressed_data
    except (zlib.error, TypeError):
        data_to_process = ser_serializable

    bio = io.BytesIO(data_to_process)
    s_header_length = struct.unpack("Q", bio.read(8))[0]
    header2 = msgpack.loads(bio.read(s_header_length), ext_hook=on_ext_data)

    buffers2 = [bio.read(s) for s in header2[0]["buf_sizes"]]
    return deserialize(header2, buffers2)
