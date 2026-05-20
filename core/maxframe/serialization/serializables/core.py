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

from typing import Dict, Tuple, Type

from maxframe.serialization.core import Serializer
from maxframe.serialization.serializables._core_c import (
    SerializableProps,
    SerializableSerializer,
    _no_field_value,
    _NoFieldValue,
    build_serializable_props,
)
from maxframe.utils import classproperty, extract_class_name


class SerializableMeta(type):
    def __new__(mcs, name: str, bases: Tuple[Type], properties: Dict):
        (
            props,
            properties_without_fields,
            properties_field_slot_names,
            slots,
        ) = build_serializable_props(name, bases, properties)

        properties = properties_without_fields

        # Add _PROPS to properties
        properties["_PROPS"] = props
        properties["__slots__"] = tuple(slots)
        # Build class props for convenience
        properties["_FIELDS"] = classproperty(lambda cls: props._FIELDS)
        properties["_FIELD_ORDER"] = classproperty(lambda cls: props._FIELD_ORDER)

        clz = type.__new__(mcs, name, bases, properties)
        props._FULL_CLASS_NAME = extract_class_name(clz)
        # Bind slot member_descriptor with field.
        for name in properties_field_slot_names:
            member_descriptor = getattr(clz, name)
            field = props._FIELDS[name]
            field.name = member_descriptor.__name__
            field.get = member_descriptor.__get__
            field.set = member_descriptor.__set__
            field._delete_method = member_descriptor.__delete__
            setattr(clz, name, field)

        return clz


class Serializable(metaclass=SerializableMeta):
    __slots__ = ("__weakref__",)

    _cache_primitive_serial = False
    _ignore_non_existing_keys = False

    _PROPS: SerializableProps

    def __init__(self, *args, **kwargs):
        fields = type(self)._PROPS._FIELDS
        field_order = type(self)._PROPS._FIELD_ORDER
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
        copied_fields = type(target)._PROPS._FIELDS
        for k, field in type(self)._PROPS._FIELDS.items():
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


class NoFieldValueSerializer(Serializer):
    def serial(self, obj, context):
        return [], [], True

    def deserial(self, serialized, context, subs):
        return _no_field_value


# Register serializers for Serializable
SerializableSerializer.register(Serializable)
NoFieldValueSerializer.register(_NoFieldValue)
