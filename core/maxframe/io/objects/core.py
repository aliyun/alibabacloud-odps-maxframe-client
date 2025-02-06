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

from ...core import Entity, EntityData
from ...core.entity import ObjectData, TileableData
from ...lib import wrapped_pickle as pickle
from ...typing_ import SlicesType, TileableType
from ...utils import TypeDispatcher
from ..odpsio.volumeio import ODPSVolumeReader, ODPSVolumeWriter

_MetaType = Dict[str, Any]

_META_FILE_NAME = ".meta"
_META_PICKLED_KEYS_KEY = ".pickled_keys"


_io_handler_dispatcher = TypeDispatcher()


def register_object_io_handler(tileable_data_type: Type[TileableData]):
    def wrapper(handler_cls):
        _io_handler_dispatcher.register(tileable_data_type, handler_cls)
        return handler_cls

    return wrapper


def get_object_io_handler(
    tileable_data_type: Union[Entity, EntityData, Type[EntityData]]
) -> Type["AbstractObjectIOHandler"]:
    if not isinstance(tileable_data_type, type):
        if isinstance(tileable_data_type, Entity):
            tileable_data_type = tileable_data_type.data
        tileable_data_type = type(tileable_data_type)
    return _io_handler_dispatcher.get_handler(tileable_data_type)


class AbstractObjectIOHandler(metaclass=ABCMeta):
    def _prepare_meta_for_serial(
        self, tileable: TileableType, meta: Dict[str, Any]
    ) -> Dict[str, Any]:
        to_pack = meta.copy()
        pickled_keys = []
        for k, v in meta.items():
            if not isinstance(v, (str, bytes, int, float, bool)):
                to_pack[k] = pickle.dumps(v)
                pickled_keys.append(k)
        to_pack[".pickled_keys"] = pickled_keys
        return to_pack

    def _prepare_meta_for_deserial(
        self, tileable: TileableType, meta: Dict[str, Any]
    ) -> Dict[str, Any]:
        pickled_keys = meta.pop(".pickled_keys", None) or []
        for k in pickled_keys:
            meta[k] = pickle.loads(meta[k])
        return meta

    def read_object_meta(
        self, reader: ODPSVolumeReader, tileable: TileableType
    ) -> Dict[str, Any]:
        meta_obj = msgpack.loads(reader.read_file(_META_FILE_NAME))
        return self._prepare_meta_for_deserial(tileable, meta_obj)

    @abstractmethod
    def _read_object_body(
        self,
        reader: ODPSVolumeReader,
        tileable: TileableType,
        meta: Dict[str, Any],
        slices: SlicesType = None,
    ) -> Any:
        raise NotImplementedError

    def read_object(
        self,
        reader: ODPSVolumeReader,
        tileable: TileableType,
        slices: SlicesType = None,
    ) -> Any:
        meta = self.read_object_meta(reader, tileable)
        return self._read_object_body(reader, tileable, meta, slices)

    @abstractmethod
    def _write_object_body(
        self, writer: ODPSVolumeWriter, tileable: TileableType, value: Any
    ):
        raise NotImplementedError

    def write_object_meta(
        self,
        writer: ODPSVolumeWriter,
        tileable: TileableType,
        extra_meta: Dict[str, Any] = None,
    ):
        meta_obj = tileable.params.copy()
        if extra_meta:
            meta_obj.update(extra_meta)
        meta_obj = self._prepare_meta_for_serial(tileable, meta_obj)
        packed = msgpack.dumps(meta_obj)
        writer.write_file(_META_FILE_NAME, packed)

    def write_object(
        self, writer: ODPSVolumeWriter, tileable: TileableType, value: Any
    ):
        self.write_object_meta(writer, tileable)
        self._write_object_body(writer, tileable, value)


@register_object_io_handler(ObjectData)
class ObjectIOHandler(AbstractObjectIOHandler):
    def _read_object_body(
        self,
        reader: ODPSVolumeReader,
        tileable: TileableType,
        meta: Dict[str, Any],
        slices: SlicesType = None,
    ) -> Any:
        return pickle.loads(reader.read_file("data"))

    def _write_object_body(
        self, writer: ODPSVolumeWriter, tileable: TileableType, value: Any
    ):
        writer.write_file("data", pickle.dumps(value))
