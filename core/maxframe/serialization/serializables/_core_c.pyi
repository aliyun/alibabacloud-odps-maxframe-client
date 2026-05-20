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

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

if TYPE_CHECKING:
    from maxframe.serialization.serializables.field_type import AbstractFieldType

class Field:
    _tag: str
    _default_value: Any
    _default_factory: Optional[Callable]
    _on_serialize: Optional[Callable]
    _on_deserialize: Optional[Callable]
    name: str
    get: Callable
    set: Callable
    _delete_method: Callable

    @property
    def tag(self) -> str: ...
    @property
    def on_serialize(self) -> Optional[Callable]: ...
    @property
    def on_deserialize(self) -> Optional[Callable]: ...
    @property
    def field_type(self) -> "AbstractFieldType": ...
    def __init__(
        self,
        tag: str,
        default: Any = ...,
        default_factory: Optional[Callable] = None,
        on_serialize: Optional[Callable] = None,
        on_deserialize: Optional[Callable] = None,
    ) -> None: ...
    def __get__(self, instance: Any, owner: Optional[type] = None) -> Any: ...
    def __set__(self, instance: Any, value: Any) -> None: ...
    def __delete__(self, instance: Any) -> None: ...

class SerializableProps:
    _LEGACY_NAME_HASH: int
    _NAME_HASH: int

    _FIELDS: Dict[str, Field]
    _FIELD_ORDER: List[str]
    _FIELD_TO_NAME_HASH: Dict[str, int]
    _PRIMITIVE_FIELDS: List[str]
    _CLS_TO_PRIMITIVE_FIELD_COUNT: Dict[int, int]
    _NON_PRIMITIVE_FIELDS: List[str]
    _CLS_TO_NON_PRIMITIVE_FIELD_COUNT: Dict[int, int]
