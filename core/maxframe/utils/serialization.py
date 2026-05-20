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

from maxframe.utils.utils import no_default

_MSG_EXT_DATA_TYPE = 43
_MSG_NA_TYPE = 44
_MSG_NO_DEFAULT = 45

# Cache for deserialize function to avoid repeated imports
_deserialize_func = None


def _get_deserialize_func():
    """Lazily load and cache the deserialize function."""
    global _deserialize_func
    if _deserialize_func is None:
        from maxframe.serialization import deserialize

        _deserialize_func = deserialize
    return _deserialize_func


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


def dump_msgpack_with_extensions(
    obj, with_ext_data=False, allowed_ext_types: Union[Type, Tuple[Type]] = None
):
    ext_dict, ext_list = {}, []

    def encode_ext(obj):
        if isinstance(obj, np.generic) and obj.shape == () and not np.isnan(obj):
            return obj.item()
        elif obj is pd.NA:
            return msgpack.ExtType(_MSG_NA_TYPE, b"")
        elif obj is no_default:
            return msgpack.ExtType(_MSG_NO_DEFAULT, b"")
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

    s_header = msgpack.dumps(obj, default=encode_ext)
    if with_ext_data:
        return s_header, ext_list
    return s_header


def load_msgpack_with_extensions(data: bytes, ext_data: List[Any] = None):
    def on_ext_data(code: int, data: bytes):
        if code == _MSG_EXT_DATA_TYPE:
            return ext_data[struct.unpack("<I", data)[0]]
        elif code == _MSG_NA_TYPE:
            return pd.NA
        elif code == _MSG_NO_DEFAULT:
            return no_default
        else:  # pragma: no cover
            raise ValueError("Unknown ext type %s" % code)

    return msgpack.loads(data, ext_hook=on_ext_data)


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
    from maxframe.serialization import serialize

    bio = io.BytesIO()
    header, buffers = serialize(serializable)
    buf_sizes = [getattr(buf, "nbytes", len(buf)) for buf in buffers]
    header[0]["buf_sizes"] = buf_sizes

    if with_ext_data:
        s_header, ext_list = dump_msgpack_with_extensions(
            header, with_ext_data=True, allowed_ext_types=allowed_ext_types
        )
    else:
        s_header = dump_msgpack_with_extensions(header)
        ext_list = None

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
    deserialize = _get_deserialize_func()

    # Try decompression with a simple header check first
    # Most serialized data is not compressed, so check header byte
    try:
        # zlib compressed data starts with specific bytes, check for magic header
        # If first 2 bytes look like zlib header (0x78 0x9C or 0x78 0xDA), try decompress
        if len(ser_serializable) >= 2 and ser_serializable[0] == 0x78:
            data_to_process = zlib.decompress(ser_serializable)
        else:
            data_to_process = ser_serializable
    except (zlib.error, TypeError, IndexError):
        data_to_process = ser_serializable

    # Direct byte access instead of BytesIO for better performance
    # Read header length (8 bytes for unsigned long long)
    s_header_length = struct.unpack_from("Q", data_to_process, 0)[0]
    offset = 8

    # Extract header data
    header_data = data_to_process[offset : offset + s_header_length]
    offset += s_header_length

    header2 = load_msgpack_with_extensions(header_data, ext_data)
    buf_sizes = header2[0]["buf_sizes"]

    # Extract buffers directly from bytes
    buffers2 = []
    for size in buf_sizes:
        buffers2.append(data_to_process[offset : offset + size])
        offset += size

    return deserialize(header2, buffers2)
