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

from typing import Any, Dict

from .core import Entity
from .executable import _ToObjectMixin
from .tileables import TileableData


class ObjectData(TileableData, _ToObjectMixin):
    __slots__ = ()
    type_name = "Object"
    # workaround for removed field since v0.1.0b5
    # todo remove this when all versions below v0.1.0b5 is eliminated
    _legacy_deprecated_non_primitives = ["_chunks"]

    def __init__(self, op=None, nsplits=None, **kw):
        super().__init__(_op=op, _nsplits=nsplits, **kw)

    def __repr__(self):
        return f"Object <op={type(self.op).__name__}, key={self.key}>"

    @property
    def params(self):
        # params return the properties which useful to rebuild a new tileable object
        return dict()

    @params.setter
    def params(self, new_params: Dict[str, Any]):
        params = new_params.copy()
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


OBJECT_TYPE = (Object, ObjectData)
