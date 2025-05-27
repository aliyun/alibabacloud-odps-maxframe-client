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

from typing import Any, Dict, Type

from ...serialization import load_type
from ...serialization.serializables import StringField
from ...utils import extract_class_name
from .core import Entity
from .executable import _ToObjectMixin
from .tileables import TileableData


class ObjectData(TileableData, _ToObjectMixin):
    __slots__ = ()
    type_name = "Object"
    # workaround for removed field since v0.1.0b5
    # todo remove this when all versions below v1.0.0rc1 is eliminated
    _legacy_deprecated_non_primitives = ["_chunks"]
    _legacy_new_non_primitives = ["object_class"]

    object_class = StringField("object_class", default=None)

    @classmethod
    def get_entity_class(cls) -> Type["Object"]:
        if getattr(cls, "_entity_class", None) is not None:
            return cls._entity_class
        assert cls.__qualname__[-4:] == "Data"
        target_class_name = extract_class_name(cls)[:-4]
        cls._entity_class = load_type(target_class_name, Object)
        return cls._entity_class

    def __new__(cls, op=None, nsplits=None, **kw):
        if cls is ObjectData:
            obj_cls = kw.get("object_class")
            if isinstance(obj_cls, str):
                obj_cls = load_type(obj_cls, (Object, ObjectData))
            if isinstance(obj_cls, type) and issubclass(obj_cls, Object):
                obj_cls = obj_cls.get_data_class()

            if obj_cls is not None and cls is not obj_cls:
                return obj_cls(op=op, nsplits=nsplits, **kw)
        return super().__new__(cls)

    def __init__(self, op=None, nsplits=None, **kw):
        obj_cls = kw.pop("object_class", None)
        if isinstance(obj_cls, type):
            if isinstance(obj_cls, type) and issubclass(obj_cls, Object):
                obj_cls = obj_cls.get_data_class()
            kw["object_class"] = extract_class_name(obj_cls)

        super().__init__(_op=op, _nsplits=nsplits, **kw)
        if self.object_class is None and type(self) is not ObjectData:
            cls = type(self)
            self.object_class = extract_class_name(cls)

    def __repr__(self):
        return (
            f"{type(self).__name__[:-4]} "
            f"<op={type(self.op).__name__}, key={self.key}>"
        )

    @property
    def params(self):
        # params return the properties which useful to rebuild a new tileable object
        return dict(object_class=self.object_class)

    @params.setter
    def params(self, new_params: Dict[str, Any]):
        params = new_params.copy()
        params.pop("object_class", None)
        if params:  # pragma: no cover
            raise TypeError(f"Unknown params: {list(params)}")

    def refresh_params(self):
        # refresh params when chunks updated
        # nothing needs to do for Object
        pass


class Object(Entity, _ToObjectMixin):
    __slots__ = ()
    _allow_data_type_ = (ObjectData,)
    type_name = "Object"

    def __new__(cls, data=None, **kw):
        if (
            cls is Object
            and isinstance(data, ObjectData)
            and type(data) is not ObjectData
        ):
            return data.get_entity_class()(data, **kw)
        return super().__new__(cls)

    @classmethod
    def get_data_class(cls) -> Type[ObjectData]:
        if getattr(cls, "_data_class", None) is not None:
            return cls._data_class
        target_class_name = extract_class_name(cls) + "Data"
        cls._data_class = load_type(target_class_name, ObjectData)
        return cls._data_class


OBJECT_TYPE = (Object, ObjectData)
