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

import operator
import weakref
from typing import Dict, List, Tuple, Type

import msgpack

from ..core import Placeholder, Serializer, buffered, load_type
from .field import Field
from .field_type import DictType, ListType, PrimitiveFieldType, TupleType


def _is_field_primitive_compound(field: Field):
    if field.on_serialize is not None or field.on_deserialize is not None:
        return False

    def check_type(field_type):
        if isinstance(field_type, PrimitiveFieldType):
            return True
        if isinstance(field_type, (ListType, TupleType)):
            if all(
                check_type(element_type) or element_type is Ellipsis
                for element_type in field_type._field_types
            ):
                return True
        if isinstance(field_type, DictType):
            if all(
                isinstance(element_type, PrimitiveFieldType) or element_type is Ellipsis
                for element_type in (field_type.key_type, field_type.value_type)
            ):
                return True
        return False

    return check_type(field.field_type)


class SerializableMeta(type):
    def __new__(mcs, name: str, bases: Tuple[Type], properties: Dict):
        # All the fields including misc fields.
        all_fields = dict()

        for base in bases:
            if hasattr(base, "_FIELDS"):
                all_fields.update(base._FIELDS)

        properties_without_fields = {}
        properties_field_slot_names = []
        for k, v in properties.items():
            if not isinstance(v, Field):
                properties_without_fields[k] = v
                continue

            field = all_fields.get(k)
            if field is None:
                properties_field_slot_names.append(k)
            else:
                v.name = field.name
                v.get = field.get
                v.set = field.set
                v.__delete__ = field.__delete__
            all_fields[k] = v

        # Make field order deterministic to serialize it as list instead of dict.
        field_order = list(all_fields)
        all_fields = dict(sorted(all_fields.items(), key=operator.itemgetter(0)))
        primitive_fields = []
        non_primitive_fields = []
        for v in all_fields.values():
            if _is_field_primitive_compound(v):
                primitive_fields.append(v)
            else:
                non_primitive_fields.append(v)

        slots = set(properties.pop("__slots__", set()))
        slots.update(properties_field_slot_names)

        properties = properties_without_fields
        properties["_FIELDS"] = all_fields
        properties["_FIELD_ORDER"] = field_order
        properties["_PRIMITIVE_FIELDS"] = primitive_fields
        properties["_NON_PRIMITIVE_FIELDS"] = non_primitive_fields
        properties["__slots__"] = tuple(slots)

        clz = type.__new__(mcs, name, bases, properties)
        # Bind slot member_descriptor with field.
        for name in properties_field_slot_names:
            member_descriptor = getattr(clz, name)
            field = all_fields[name]
            field.name = member_descriptor.__name__
            field.get = member_descriptor.__get__
            field.set = member_descriptor.__set__
            field.__delete__ = member_descriptor.__delete__
            setattr(clz, name, field)

        return clz


class Serializable(metaclass=SerializableMeta):
    __slots__ = ("__weakref__",)

    _cache_primitive_serial = False
    _ignore_non_existing_keys = False

    _FIELDS: Dict[str, Field]
    _FIELD_ORDER: List[str]
    _PRIMITIVE_FIELDS: List[str]
    _NON_PRIMITIVE_FIELDS: List[str]

    def __init__(self, *args, **kwargs):
        fields = self._FIELDS
        field_order = self._FIELD_ORDER
        assert len(args) <= len(field_order)
        if args:  # pragma: no cover
            values = dict(zip(field_order, args))
            values.update(kwargs)
        else:
            values = kwargs
        for k, v in values.items():
            try:
                fields[k].set(self, v)
            except KeyError:
                if not self._ignore_non_existing_keys:
                    raise

    def __on_deserialize__(self):
        pass

    def __repr__(self):
        values = ", ".join(
            [
                "{}={!r}".format(slot, getattr(self, slot, None))
                for slot in self.__slots__
            ]
        )
        return "{}({})".format(self.__class__.__name__, values)

    def copy_to(self, target: "Serializable") -> "Serializable":
        copied_fields = target._FIELDS
        for k, field in self._FIELDS.items():
            try:
                # Slightly faster than getattr.
                value = field.get(self, k)
                try:
                    copied_fields[k].set(target, value)
                except KeyError:
                    copied_fields["_" + k].set(target, value)
            except AttributeError:
                continue
        return target

    def copy(self) -> "Serializable":
        return self.copy_to(type(self)())


_primitive_serial_cache = weakref.WeakKeyDictionary()


class _NoFieldValue:
    pass


_no_field_value = _NoFieldValue()


class SerializableSerializer(Serializer):
    """
    Leverage DictSerializer to perform serde.
    """

    @classmethod
    def _get_field_values(cls, obj: Serializable, fields):
        values = []
        for field in fields:
            try:
                value = field.get(obj)
                if field.on_serialize is not None:
                    value = field.on_serialize(value)
            except AttributeError:
                # Most field values are not None, serialize by list is more efficient than dict.
                value = _no_field_value
            values.append(value)
        return values

    @buffered
    def serial(self, obj: Serializable, context: Dict):
        if obj._cache_primitive_serial and obj in _primitive_serial_cache:
            primitive_vals = _primitive_serial_cache[obj]
        else:
            primitive_vals = self._get_field_values(obj, obj._PRIMITIVE_FIELDS)
            # replace _no_field_value as {} to make them msgpack-serializable
            primitive_vals = [
                v if v is not _no_field_value else {} for v in primitive_vals
            ]
            if obj._cache_primitive_serial:
                primitive_vals = msgpack.dumps(primitive_vals)
                _primitive_serial_cache[obj] = primitive_vals

        compound_vals = self._get_field_values(obj, obj._NON_PRIMITIVE_FIELDS)
        cls_module = f"{type(obj).__module__}#{type(obj).__qualname__}"
        return [cls_module, primitive_vals], [compound_vals], False

    @staticmethod
    def _set_field_value(obj: Serializable, field: Field, value):
        if value is _no_field_value:
            return
        if type(value) is Placeholder:
            if field.on_deserialize is not None:
                value.callbacks.append(
                    lambda v: field.set(obj, field.on_deserialize(v))
                )
            else:
                value.callbacks.append(lambda v: field.set(obj, v))
        else:
            if field.on_deserialize is not None:
                field.set(obj, field.on_deserialize(value))
            else:
                field.set(obj, value)

    def deserial(self, serialized: List, context: Dict, subs: List) -> Serializable:
        obj_class_name, primitives = serialized
        obj_class = load_type(obj_class_name, Serializable)

        if type(primitives) is not list:
            primitives = msgpack.loads(primitives)

        obj = obj_class.__new__(obj_class)

        if primitives:
            for field, value in zip(obj_class._PRIMITIVE_FIELDS, primitives):
                if value != {}:
                    self._set_field_value(obj, field, value)

        if obj_class._NON_PRIMITIVE_FIELDS:
            for field, value in zip(obj_class._NON_PRIMITIVE_FIELDS, subs[0]):
                self._set_field_value(obj, field, value)
        obj.__on_deserialize__()
        return obj


class NoFieldValueSerializer(Serializer):
    def serial(self, obj, context):
        return [], [], True

    def deserial(self, serialized, context, subs):
        return _no_field_value


SerializableSerializer.register(Serializable)
NoFieldValueSerializer.register(_NoFieldValue)
