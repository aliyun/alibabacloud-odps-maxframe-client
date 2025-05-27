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

from abc import ABCMeta, abstractmethod
from typing import Any, Dict, Type, Union

import msgpack
import numpy as np

from ...core import Entity, EntityData, OutputType, get_output_types
from ...lib import wrapped_pickle as pickle
from ...typing_ import SlicesType, TileableType
from ...utils import deserialize_serializable, serialize_serializable
from ..odpsio.volumeio import ODPSVolumeReader, ODPSVolumeWriter

_MetaType = Dict[str, Any]

_META_FILE_NAME = ".meta"
_META_PICKLED_KEYS_KEY = ".pickled_keys"
_META_SERIALIZED_VALUE_KEY = ".serialized_values"


_io_handler_dispatcher = dict()


def register_object_io_handler(out_type: OutputType):
    def wrapper(handler_cls):
        _io_handler_dispatcher[out_type] = handler_cls
        return handler_cls

    return wrapper


def get_object_io_handler(
    tileable_data_type: Union[Entity, EntityData, Type[EntityData], OutputType]
) -> Type["AbstractObjectIOHandler"]:
    output_type = tileable_data_type
    if not isinstance(tileable_data_type, OutputType):
        if hasattr(tileable_data_type, "output_type"):
            output_type = tileable_data_type.output_type
        else:
            [output_type] = get_output_types(tileable_data_type)
    return _io_handler_dispatcher[output_type]


class AbstractObjectIOHandler(metaclass=ABCMeta):
    def _prepare_meta_for_serial(self, tileable: TileableType) -> Dict[str, Any]:
        return tileable.params.copy()

    def read_object_meta(
        self, reader: ODPSVolumeReader, tileable: TileableType
    ) -> Dict[str, Any]:
        file_bin = reader.read_file(_META_FILE_NAME)
        meta_obj = msgpack.loads(file_bin)
        pickled_keys = meta_obj.pop(_META_PICKLED_KEYS_KEY, None) or []
        serialized_values = meta_obj.pop(_META_SERIALIZED_VALUE_KEY, None) or {}
        for k in pickled_keys:
            try:
                meta_obj[k] = deserialize_serializable(serialized_values[k])
            except:
                meta_obj[k] = pickle.loads(meta_obj[k])
        return meta_obj

    @abstractmethod
    def read_object_body(
        self,
        reader: ODPSVolumeReader,
        params: Dict[str, Any],
        extra_params: Dict[str, Any] = None,
        slices: SlicesType = None,
    ) -> Any:
        raise NotImplementedError

    def read_object(
        self,
        reader: ODPSVolumeReader,
        tileable: TileableType,
        extra_params: Dict[str, Any] = None,
        slices: SlicesType = None,
    ) -> Any:
        obj_meta = self.read_object_meta(reader, tileable)
        return self.read_object_body(reader, obj_meta, extra_params, slices)

    @abstractmethod
    def write_object_body(
        self, writer: ODPSVolumeWriter, params: Dict[str, Any], value: Any
    ):
        raise NotImplementedError

    @classmethod
    def _transform_serializable_meta(cls, meta: Dict[str, Any]) -> Dict[str, Any]:
        to_pack = meta.copy()
        pickled_keys = []
        serialized_values = {}
        for k, v in meta.items():
            if not isinstance(v, (str, bytes, int, float, bool, type(None))):
                to_pack[k] = pickle.dumps(v)
                serialized_values[k] = serialize_serializable(v)
                pickled_keys.append(k)
        to_pack[_META_PICKLED_KEYS_KEY] = pickled_keys
        to_pack[_META_SERIALIZED_VALUE_KEY] = serialized_values
        return to_pack

    def write_object_meta(
        self,
        writer: ODPSVolumeWriter,
        tileable: TileableType,
        extra_meta: Dict[str, Any] = None,
    ):
        meta_obj = self._prepare_meta_for_serial(tileable)
        if extra_meta:
            meta_obj.update(extra_meta)

        def encode_np_num(obj):
            if isinstance(obj, np.generic) and obj.shape == ():
                return obj.item()
            return obj

        meta_obj = self._transform_serializable_meta(meta_obj)
        writer.write_file(
            _META_FILE_NAME, msgpack.dumps(meta_obj, default=encode_np_num)
        )

    def write_object(
        self, writer: ODPSVolumeWriter, tileable: TileableType, value: Any
    ):
        self.write_object_meta(writer, tileable)
        self.write_object_body(writer, tileable.params, value)


@register_object_io_handler(OutputType.object)
class ObjectIOHandler(AbstractObjectIOHandler):
    def read_object_body(
        self,
        reader: ODPSVolumeReader,
        params: Dict[str, Any],
        extra_params: Dict[str, Any] = None,
        slices: SlicesType = None,
    ) -> Any:
        return pickle.loads(reader.read_file("data"))

    def write_object_body(
        self, writer: ODPSVolumeWriter, params: Dict[str, Any], value: Any
    ):
        writer.write_file("data", pickle.dumps(value))
