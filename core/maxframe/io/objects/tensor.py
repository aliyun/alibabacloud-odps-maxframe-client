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

import itertools
import struct
from io import BytesIO
from typing import Any, Dict

import msgpack
import numpy as np

from ...config import options
from ...core import OutputType
from ...lib import wrapped_pickle as pickle
from ...lib.version import parse as parse_version
from ...typing_ import SlicesType, TileableType
from ..odpsio import ODPSVolumeReader, ODPSVolumeWriter
from .core import AbstractObjectIOHandler, register_object_io_handler

try:
    from maxframe import __version__ as mf_version
except ImportError:
    mf_version = None


@register_object_io_handler(OutputType.tensor)
class TensorIOHandler(AbstractObjectIOHandler):
    def _prepare_meta_for_serial(self, tileable: TileableType) -> Dict[str, Any]:
        meta = super()._prepare_meta_for_serial(tileable)
        meta["nsplits"] = (
            getattr(tileable, "nsplits", None) or (np.nan,) * tileable.ndim
        )
        return meta

    def write_object_meta(
        self,
        writer: ODPSVolumeWriter,
        tileable: TileableType,
        extra_meta: Dict[str, Any] = None,
    ):
        extra_meta = extra_meta or dict()
        extra_meta["nsplits"] = (
            getattr(tileable, "nsplits", None) or ((np.nan,),) * tileable.ndim
        )
        super().write_object_meta(writer, tileable, extra_meta=extra_meta)

    @staticmethod
    def _get_chunk_file_name(params: Dict[str, Any]) -> str:
        # fixme remove this when all v1.0 clients removed
        mf_ver = options.session.client_version or mf_version
        if mf_ver:
            client_major = parse_version(
                options.session.client_version or mf_version
            ).major
        else:
            client_major = None

        if client_major == 1:
            # returns v1.0 file name as we do not support tiled tensors
            return "0,0.dat"

        ndim = len(params.get("shape") or (0,))
        data_index = params.get("index") or (0,) * ndim
        return ",".join(str(ix) for ix in data_index) + ".dat"

    def _read_single_chunk(
        self, reader: ODPSVolumeReader, params: Dict[str, Any]
    ) -> Any:
        file_name = self._get_chunk_file_name(params)
        body = reader.read_file(file_name)
        bio = BytesIO(body)
        (header_len,) = struct.unpack("<I", bio.read(4))
        header_data = msgpack.loads(bio.read(header_len))

        pickled = bio.read(header_data[0])
        bufs = [bio.read(size) for size in header_data[1:]]
        return pickle.loads(pickled, buffers=bufs)

    def read_object_body(
        self,
        reader: ODPSVolumeReader,
        params: Dict[str, Any],
        extra_params: Dict[str, Any] = None,
        slices: SlicesType = None,
    ) -> Any:
        if "index" in params:
            return self._read_single_chunk(reader, params)

        tileable_params = params.copy()
        tileable_params.update(extra_params or {})

        # todo implements slices argument for head and tail fetching
        assert slices is None

        chunk_shape = tuple(len(x) for x in params["nsplits"])
        block_array = np.empty(shape=chunk_shape, dtype="O")
        for idx in itertools.product(*(range(sp) for sp in chunk_shape)):
            chunk_params = tileable_params.copy()
            chunk_params.pop("nsplits")
            chunk_params["index"] = idx
            block_array[idx] = self._read_single_chunk(reader, chunk_params)

        return np.block(block_array.tolist())

    def write_object_body(
        self, writer: ODPSVolumeWriter, params: Dict[str, Any], value: Any
    ):
        def data_gen():
            bufs = []
            pickled = pickle.dumps(value, buffer_callback=bufs.append)
            header_data = msgpack.dumps(
                [len(pickled)] + [len(buf.raw()) for buf in bufs]
            )
            yield struct.pack("<I", len(header_data))
            yield header_data
            yield pickled
            for buf in bufs:
                yield buf

        file_name = self._get_chunk_file_name(params)
        writer.write_file(file_name, data_gen())
