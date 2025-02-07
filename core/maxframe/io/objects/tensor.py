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

import struct
from io import BytesIO
from typing import Any, Dict

import msgpack
import numpy as np

from ...lib import wrapped_pickle as pickle
from ...tensor.core import TensorData
from ...typing_ import SlicesType, TileableType
from ..odpsio import ODPSVolumeReader, ODPSVolumeWriter
from .core import AbstractObjectIOHandler, register_object_io_handler


@register_object_io_handler(TensorData)
class TensorIOHandler(AbstractObjectIOHandler):
    def write_object_meta(
        self,
        writer: ODPSVolumeWriter,
        tileable: TileableType,
        extra_meta: Dict[str, Any] = None,
    ):
        # fixme upload in real slices when tensors are supported in DPE
        extra_meta = extra_meta or dict()
        extra_meta["nsplits"] = ((np.nan,),)

        super().write_object_meta(writer, tileable, extra_meta=extra_meta)

    def _read_object_body(
        self,
        reader: ODPSVolumeReader,
        tileable: TileableType,
        meta: Dict[str, Any],
        slices: SlicesType = None,
    ) -> Any:
        # fixme read data with slices when tensors are supported in DPE
        body = reader.read_file("0,0.dat")
        bio = BytesIO(body)
        (header_len,) = struct.unpack("<I", bio.read(4))
        header_data = msgpack.loads(bio.read(header_len))

        pickled = bio.read(header_data[0])
        bufs = [bio.read(size) for size in header_data[1:]]
        return pickle.loads(pickled, buffers=bufs)

    def _write_object_body(
        self, writer: ODPSVolumeWriter, tileable: TileableType, value: Any
    ):
        # fixme upload in real slices when tensors are supported in DPE
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

        writer.write_file("0,0.dat", data_gen())
