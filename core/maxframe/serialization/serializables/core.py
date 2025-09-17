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

import logging
import weakref
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple, Type

import msgpack

from ...errors import MaxFrameDeprecationError
from ...lib.mmh3 import hash
from ...utils import extract_class_name, no_default
from ..core import Placeholder, Serializer, buffered, load_type
from .field import Field
from .field_type import DictType, ListType, PrimitiveFieldType, TupleType

try:
    from ..deserializer import get_legacy_class_name
except ImportError:
    get_legacy_class_name = lambda x: x

logger = logging.getLogger(__name__)
_deprecate_log_key = "_SER_DEPRECATE_LOGGED"


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
        legacy_name = properties.get("_legacy_name", name)
        legacy_name_hash = hash(
            get_legacy_class_name(f"{properties.get('__module__')}.{legacy_name}")
        )
        name_hash = hash(
            f"{properties.get('__module__')}.{properties.get('__qualname__')}"
        )
        all_fields = dict()
        # mapping field names to base classes
        field_to_cls_hash = dict()
        # mapping legacy name hash to name hashes
        legacy_to_new_name_hash = {legacy_name_hash: name_hash}

        for base in bases:
            if not hasattr(base, "_FIELDS"):
                continue
            all_fields.update(base._FIELDS)
            field_to_cls_hash.update(base._FIELD_TO_NAME_HASH)
            legacy_to_new_name_hash.update(base._LEGACY_TO_NEW_NAME_HASH)

        properties_without_fields = {}
        properties_field_slot_names = []
        for k, v in properties.items():
            if not isinstance(v, Field):
                properties_without_fields[k] = v
                continue

            field = all_fields.get(k)
            # record the field for the class being created
            field_to_cls_hash[k] = name_hash
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
        primitive_fields = []
        primitive_field_names = set()
        non_primitive_fields = []
        for field_name, v in all_fields.items():
            if _is_field_primitive_compound(v):
                primitive_fields.append(v)
                primitive_field_names.add(field_name)
            else:
                non_primitive_fields.append(v)

        # count number of fields for every base class
        cls_to_primitive_field_count = OrderedDict()
        cls_to_non_primitive_field_count = OrderedDict()
        for field_name in field_order:
            cls_hash = field_to_cls_hash[field_name]
            if field_name in primitive_field_names:
                cls_to_primitive_field_count[cls_hash] = (
                    cls_to_primitive_field_count.get(cls_hash, 0) + 1
                )
            else:
                cls_to_non_primitive_field_count[cls_hash] = (
                    cls_to_non_primitive_field_count.get(cls_hash, 0) + 1
                )

        slots = set(properties.pop("__slots__", set()))
        slots.update(properties_field_slot_names)

        properties = properties_without_fields

        # todo remove this prop when all versions below v1.0.0rc1 is eliminated
        properties["_LEGACY_NAME_HASH"] = legacy_name_hash
        properties["_NAME_HASH"] = name_hash
        properties["_LEGACY_TO_NEW_NAME_HASH"] = legacy_to_new_name_hash

        properties["_FIELDS"] = all_fields
        properties["_FIELD_ORDER"] = field_order
        properties["_FIELD_TO_NAME_HASH"] = field_to_cls_hash
        properties["_PRIMITIVE_FIELDS"] = primitive_fields
        properties["_CLS_TO_PRIMITIVE_FIELD_COUNT"] = OrderedDict(
            cls_to_primitive_field_count
        )
        properties["_NON_PRIMITIVE_FIELDS"] = non_primitive_fields
        properties["_CLS_TO_NON_PRIMITIVE_FIELD_COUNT"] = OrderedDict(
            cls_to_non_primitive_field_count
        )
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

    _LEGACY_NAME_HASH: int
    _NAME_HASH: int
    _LEGACY_TO_NEW_NAME_HASH: Dict[int, int]

    _FIELDS: Dict[str, Field]
    _FIELD_ORDER: List[str]
    _FIELD_TO_NAME_HASH: Dict[str, int]
    _PRIMITIVE_FIELDS: List[str]
    _CLS_TO_PRIMITIVE_FIELD_COUNT: Dict[int, int]
    _NON_PRIMITIVE_FIELDS: List[str]
    _CLS_TO_NON_PRIMITIVE_FIELD_COUNT: Dict[int, int]

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


def _to_primitive_placeholder(v: Any) -> Any:
    if v is _no_field_value or v is no_default:
        return {}
    return v


def _restore_primitive_placeholder(v: Any) -> Any:
    if type(v) is dict:
        if v == {}:
            return _no_field_value
        else:
            return v
    else:
        return v


class SerializableSerializer(Serializer):
    """
    Leverage DictSerializer to perform serde.
    """

    @classmethod
    def _log_legacy(cls, context: Dict, key: Any, msg: str, *args, **kwargs):
        level = kwargs.pop("level", logging.WARNING)
        try:
            logged_keys = context[_deprecate_log_key]
        except KeyError:
            logged_keys = context[_deprecate_log_key] = set()
        if key not in logged_keys:
            logged_keys.add(key)
            logger.log(level, msg, *args, **kwargs)

    @classmethod
    def _get_obj_field_count_key(cls, obj: Serializable, legacy: bool = False):
        return f"FC_{obj._NAME_HASH if not legacy else obj._LEGACY_NAME_HASH}"

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
            primitive_vals = [_to_primitive_placeholder(v) for v in primitive_vals]
            if obj._cache_primitive_serial:
                primitive_vals = msgpack.dumps(primitive_vals)
                _primitive_serial_cache[obj] = primitive_vals

        compound_vals = self._get_field_values(obj, obj._NON_PRIMITIVE_FIELDS)
        cls_module = extract_class_name(type(obj))

        field_count_key = self._get_obj_field_count_key(obj)
        if not self.is_public_data_exist(context, field_count_key):
            # store field distribution for current Serializable
            counts = [
                list(obj._CLS_TO_PRIMITIVE_FIELD_COUNT.items()),
                list(obj._CLS_TO_NON_PRIMITIVE_FIELD_COUNT.items()),
            ]
            field_count_data = msgpack.dumps(counts)
            self.put_public_data(
                context, self._get_obj_field_count_key(obj), field_count_data
            )
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

    @classmethod
    def _prune_server_fields(
        cls,
        client_cls_to_field_count: Optional[Dict[int, int]],
        server_cls_to_field_count: Dict[int, int],
        server_fields: list,
        legacy_to_new_hash: Dict[int, int],
    ) -> list:
        if set(client_cls_to_field_count.keys()) == set(
            server_cls_to_field_count.keys()
        ):
            return server_fields

        new_to_legacy_hash = {v: k for k, v in legacy_to_new_hash.items()}
        ret_server_fields = []
        server_pos = 0
        for cls_hash, count in server_cls_to_field_count.items():
            if (
                cls_hash in client_cls_to_field_count
                or new_to_legacy_hash.get(cls_hash) in client_cls_to_field_count
            ):
                ret_server_fields.extend(server_fields[server_pos : server_pos + count])
            server_pos += count
        return ret_server_fields

    @classmethod
    def _set_field_values(
        cls,
        obj: Serializable,
        values: List[Any],
        client_cls_to_field_count: Optional[Dict[int, int]],
        is_primitive: bool = True,
    ):
        obj_class = type(obj)
        legacy_to_new_hash = obj_class._LEGACY_TO_NEW_NAME_HASH

        if is_primitive:
            server_cls_to_field_count = obj_class._CLS_TO_PRIMITIVE_FIELD_COUNT
            field_def_list = obj_class._PRIMITIVE_FIELDS
        else:
            server_cls_to_field_count = obj_class._CLS_TO_NON_PRIMITIVE_FIELD_COUNT
            field_def_list = obj_class._NON_PRIMITIVE_FIELDS

        server_fields = cls._prune_server_fields(
            client_cls_to_field_count,
            server_cls_to_field_count,
            field_def_list,
            legacy_to_new_hash,
        )

        field_num, server_field_num = 0, 0
        for cls_hash, count in client_cls_to_field_count.items():
            # cut values and fields given field distribution
            # at client and server end
            cls_fields = server_fields[server_field_num : field_num + count]
            cls_values = values[field_num : field_num + count]
            for field, value in zip(cls_fields, cls_values):
                if is_primitive:
                    value = _restore_primitive_placeholder(value)
                if not is_primitive or value is not _no_field_value:
                    cls._set_field_value(obj, field, value)
            field_num += count
            try:
                server_field_num += server_cls_to_field_count[cls_hash]
            except KeyError:
                try:
                    server_field_num += server_cls_to_field_count[
                        legacy_to_new_hash[cls_hash]
                    ]
                except KeyError:
                    # it is possible that certain type of field does not exist
                    #  at server side
                    pass

    def deserial(self, serialized: List, context: Dict, subs: List) -> Serializable:
        obj_class_name, primitives = serialized
        obj_class = load_type(obj_class_name, Serializable)

        if type(primitives) is not list:
            primitives = msgpack.loads(primitives)

        obj = obj_class.__new__(obj_class)

        field_count_data = self.get_public_data(
            context, self._get_obj_field_count_key(obj)
        )
        if field_count_data is None:
            # try using legacy field count key to get counts
            field_count_data = self.get_public_data(
                context, self._get_obj_field_count_key(obj, legacy=True)
            )

            if field_count_data is None:
                self._log_legacy(
                    context,
                    ("MISSING_CLASS", obj_class_name),
                    "Field count info of %s not found in serialized data",
                    obj_class_name,
                    level=logging.ERROR,
                )
                raise MaxFrameDeprecationError(
                    "Failed to deserialize request. Please upgrade your "
                    "MaxFrame client to the latest release."
                )
            else:
                self._log_legacy(
                    context,
                    ("LEGACY_CLASS", obj_class_name),
                    "Class %s used in legacy client",
                    obj_class_name,
                )

        cls_to_prim_key, cls_to_non_prim_key = msgpack.loads(field_count_data)
        cls_to_prim_key = dict(cls_to_prim_key)
        cls_to_non_prim_key = dict(cls_to_non_prim_key)

        if primitives:
            self._set_field_values(obj, primitives, cls_to_prim_key, True)
        if obj_class._NON_PRIMITIVE_FIELDS:
            self._set_field_values(obj, subs[0], cls_to_non_prim_key, False)
        ret = obj.__on_deserialize__()
        return obj if ret is None else ret


class NoFieldValueSerializer(Serializer):
    def serial(self, obj, context):
        return [], [], True

    def deserial(self, serialized, context, subs):
        return _no_field_value


SerializableSerializer.register(Serializable)
NoFieldValueSerializer.register(_NoFieldValue)
