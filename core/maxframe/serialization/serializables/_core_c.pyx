# distutils: language = c++
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

import logging
import os
import weakref
from collections import OrderedDict

cimport cython
from cpython cimport PyObject
from libc.stdint cimport uint64_t, uintptr_t
from libcpp.unordered_map cimport unordered_map
from libcpp.utility cimport pair

from maxframe.errors import MaxFrameDeprecationError
from maxframe.serialization.core cimport Placeholder, Serializer
from maxframe.serialization.core import load_type
from maxframe.utils import no_default, str_to_bool
from maxframe.utils.serialization import (
    dump_msgpack_with_extensions,
    load_msgpack_with_extensions,
)

# Import Serializable - will be defined later in core.py
# We need to delay the registration until after Serializable is defined

try:
    from maxframe.serialization.deserializer import get_legacy_class_name
except ImportError:
    get_legacy_class_name = lambda x: x

from maxframe.lib.mmh3 import hash

logger = logging.getLogger(__name__)
_deprecate_log_key = "_SER_DEPRECATE_LOGGED"
_primitive_serial_cache = weakref.WeakKeyDictionary()
cdef dict _field_count_cache = {}

# Cache for loaded Serializable class to avoid repeated imports
_Serializable_class = None

_MSG_NA_TYPE = 44
_MSG_NO_DEFAULT = 45

# Global CI flag check
cdef bint _is_ci = bool(str_to_bool(os.environ.get("CI")))


class _NoFieldValue:
    pass


_no_field_value = _NoFieldValue()


cdef inline uint64_t _fast_id(PyObject * obj) nogil:
    return <uintptr_t>obj


def set_is_ci(is_ci):
    global _is_ci
    _is_ci = is_ci


cdef class Field:
    """
    Base field class for serializable attributes.
    Child classes must implement field_type property.
    """
    cdef:
        public str _tag
        public object _default_value
        public object _default_factory
        public object _on_serialize
        public object _on_deserialize
        public str name
        public object get
        public object set
        public object _delete_method

    def __init__(
        self,
        str tag,
        object default=no_default,
        object default_factory=None,
        object on_serialize=None,
        object on_deserialize=None,
    ):
        if default is not no_default and default_factory is not None:
            raise ValueError("default and default_factory can not be specified both")

        self._tag = tag
        self._default_value = default
        self._default_factory = default_factory
        self._on_serialize = on_serialize
        self._on_deserialize = on_deserialize
        self.name = None
        self.get = None
        self.set = None
        self._delete_method = None

    @property
    def tag(self):
        return self._tag

    @property
    def on_serialize(self):
        return self._on_serialize

    @property
    def on_deserialize(self):
        return self._on_deserialize

    @property
    def field_type(self):
        """
        Abstract property - child classes must override.
        """
        raise NotImplementedError("Child classes must implement field_type property")

    @cython.optimize.unpack_method_calls(False)
    def __get__(self, instance, owner):
        try:
            return self.get(instance, owner)
        except AttributeError:
            if self._default_value is not no_default:
                val = self._default_value
                self.set(instance, val)
                return val
            elif self._default_factory is not None:
                val = self._default_factory()
                self.set(instance, val)
                return val
            else:
                raise

    @cython.optimize.unpack_method_calls(False)
    def __set__(self, instance, value):
        if _is_ci:
            from maxframe.core import is_kernel_mode

            if not is_kernel_mode():
                field_type = self.field_type
                try:
                    to_check_value = value
                    if to_check_value is not None and self._on_serialize:
                        to_check_value = self._on_serialize(to_check_value)
                    field_type.validate(to_check_value)
                except (TypeError, ValueError) as e:
                    raise type(e)(
                        f"Failed to set `{self.name}` for {type(instance).__name__} "
                        f"when environ CI=true is set: {str(e)}"
                    )
        self.set(instance, value)

    @cython.optimize.unpack_method_calls(False)
    def __delete__(self, instance):
        if self._delete_method is not None:
            self._delete_method(instance)


cdef class SerializableProps:
    cdef:
        public int _LEGACY_NAME_HASH
        public int _NAME_HASH
        public unordered_map[int, int] _LEGACY_TO_NEW_NAME_HASH
        public str _FULL_CLASS_NAME

        public dict _FIELDS
        public list _FIELD_ORDER
        public dict _FIELD_TO_NAME_HASH
        public list _PRIMITIVE_FIELDS
        public object _CLS_TO_PRIMITIVE_FIELD_COUNT
        public list _NON_PRIMITIVE_FIELDS
        public object _CLS_TO_NON_PRIMITIVE_FIELD_COUNT

    def __init__(self):
        self._LEGACY_NAME_HASH = 0
        self._NAME_HASH = 0
        # _LEGACY_TO_NEW_NAME_HASH is a C++ unordered_map, initialized automatically
        self._FULL_CLASS_NAME = None

        self._FIELDS = {}
        self._FIELD_ORDER = []
        self._FIELD_TO_NAME_HASH = {}
        self._PRIMITIVE_FIELDS = []
        self._CLS_TO_PRIMITIVE_FIELD_COUNT = OrderedDict()
        self._NON_PRIMITIVE_FIELDS = []
        self._CLS_TO_NON_PRIMITIVE_FIELD_COUNT = OrderedDict()


def build_serializable_props(name, bases, properties):
    """
    Build SerializableProps from class creation parameters.

    Parameters
    ----------
    name : str
        Class name
    bases : tuple
        Base classes
    properties : dict
        Class properties dictionary
    get_legacy_class_name_func : callable, optional
        Function to get legacy class name

    Returns
    -------
    tuple
        (props, properties_without_fields, properties_field_slot_names, slots)
    """
    cdef SerializableProps base_props, props
    cdef pair[int, int] int_pair
    cdef dict all_fields, field_to_cls_hash
    cdef list primitive_fields, non_primitive_fields, properties_field_slot_names
    cdef set primitive_field_names, slots

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
        if not hasattr(base, "_PROPS"):
            continue
        base_props = base._PROPS
        all_fields.update(base_props._FIELDS)
        field_to_cls_hash.update(base_props._FIELD_TO_NAME_HASH)
        for int_pair in base_props._LEGACY_TO_NEW_NAME_HASH:
            legacy_to_new_name_hash[int_pair.first] = int_pair.second

    properties_without_fields = {}
    properties_field_slot_names = []
    for k, v in properties.items():
        # Field is now defined in this module, no need to import
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
            v._delete_method = field._delete_method
        all_fields[k] = v

    # Make field order deterministic to serialize it as list instead of dict.
    field_order = list(all_fields)
    primitive_fields = []
    primitive_field_names = set()
    non_primitive_fields = []

    # Use the local _is_field_primitive_compound function defined above
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

    # Create and populate SerializableProps object
    props = SerializableProps()
    props._LEGACY_NAME_HASH = legacy_name_hash
    props._NAME_HASH = name_hash
    for int_pair in legacy_to_new_name_hash.items():
        props._LEGACY_TO_NEW_NAME_HASH[int_pair.first] = int_pair.second
    props._FIELDS = all_fields
    props._FIELD_ORDER = field_order
    props._FIELD_TO_NAME_HASH = field_to_cls_hash
    props._PRIMITIVE_FIELDS = primitive_fields
    props._CLS_TO_PRIMITIVE_FIELD_COUNT = cls_to_primitive_field_count
    props._NON_PRIMITIVE_FIELDS = non_primitive_fields
    props._CLS_TO_NON_PRIMITIVE_FIELD_COUNT = cls_to_non_primitive_field_count

    return props, properties_without_fields, properties_field_slot_names, slots


# Helper function for checking if field is primitive compound
cdef inline bint _is_field_primitive_compound(field):
    """Check if a field is a primitive compound type (optimized for Cython)"""
    from maxframe.serialization.serializables.field_type import (
        DictType,
        ListType,
        PrimitiveFieldType,
        TupleType,
    )

    if getattr(field, "_primitive", False):
        return True
    if field.on_serialize is not None or field.on_deserialize is not None:
        return False

    cdef bint result
    field_type = field.field_type

    if isinstance(field_type, PrimitiveFieldType):
        return True
    if isinstance(field_type, (ListType, TupleType)):
        result = all(
            _check_type_primitive(element_type) or element_type is Ellipsis
            for element_type in field_type._field_types
        )
        return result
    if isinstance(field_type, DictType):
        result = all(
            isinstance(element_type, PrimitiveFieldType) or element_type is Ellipsis
            for element_type in (field_type.key_type, field_type.value_type)
        )
        return result
    return False


cdef inline bint _check_type_primitive(field_type):
    """Helper for checking if a type is primitive"""
    from maxframe.serialization.serializables.field_type import (
        DictType,
        ListType,
        PrimitiveFieldType,
        TupleType,
    )

    if isinstance(field_type, PrimitiveFieldType):
        return True
    if isinstance(field_type, (ListType, TupleType)):
        return all(
            _check_type_primitive(element_type) or element_type is Ellipsis
            for element_type in field_type._field_types
        )
    if isinstance(field_type, DictType):
        return all(
            isinstance(element_type, PrimitiveFieldType) or element_type is Ellipsis
            for element_type in (field_type.key_type, field_type.value_type)
        )
    return False


# Define serializers as Python classes (not cdef classes) since Serializer is not an extension type
cdef class SerializableSerializer(Serializer):
    """
    Cython-optimized serializer for Serializable objects.
    Leverages typed fields and optimized loops for better performance.
    """
    # Use original class path for serializer ID
    serializer_id = Serializer.calc_default_serializer_id(
        "maxframe.serialization.serializables.core.SerializableSerializer"
    )

    def _log_legacy(self, dict context, key, str msg, *args, **kwargs):
        cdef int level = kwargs.pop("level", logging.WARNING)
        cdef set logged_keys
        try:
            logged_keys = context[_deprecate_log_key]
        except KeyError:
            logged_keys = context[_deprecate_log_key] = set()
        if key not in logged_keys:
            logged_keys.add(key)
            logger.log(level, msg, *args, **kwargs)

    cdef list _get_field_values(self, obj, list fields, default=no_default):
        """Extract field values from an object (optimized)"""
        cdef int idx
        cdef list values
        cdef object value
        cdef Field field

        values = [default] * len(fields)

        for idx, field in enumerate(fields):
            try:
                value = field.get(obj)
                if field._on_serialize is not None:
                    value = field._on_serialize(value)
                values[idx] = value
            except AttributeError:
                pass
        return values

    cpdef serial(self, obj, dict context):
        """Serialize a Serializable object"""
        cdef object primitive_vals
        cdef object compound_vals
        cdef object obj_class
        cdef str cls_module
        cdef str field_count_key
        cdef object field_count_data
        cdef list counts
        cdef SerializableProps props
        cdef bint cache_primitive
        cdef uint64_t obj_id

        obj_id = _fast_id(<PyObject *>obj)
        if obj_id in context:
            return Placeholder(obj_id)
        context[obj_id] = obj

        obj_class = type(obj)
        props = obj_class._PROPS

        cache_primitive = obj._cache_primitive_serial
        if cache_primitive and obj in _primitive_serial_cache:
            primitive_vals = _primitive_serial_cache[obj]
        else:
            primitive_vals = self._get_field_values(obj, props._PRIMITIVE_FIELDS)

            # fixme for compatibility in deployment, remove this block in v2.8 release
            for idx in range(len(<list>primitive_vals)):
                if (<list>primitive_vals)[idx] is no_default:
                    (<list>primitive_vals)[idx] = {}

            if cache_primitive:
                primitive_vals = dump_msgpack_with_extensions(primitive_vals)
                _primitive_serial_cache[obj] = primitive_vals

        compound_vals = self._get_field_values(
            obj, props._NON_PRIMITIVE_FIELDS, _no_field_value
        )

        # Use cached field count key for this class if available
        field_count_key = "FC_" + str(props._NAME_HASH)

        if not self.is_public_data_exist(context, field_count_key):
            # Use cached field_count_data for this class if available
            if obj_class in _field_count_cache:
                field_count_data = _field_count_cache[obj_class]
            else:
                # store field distribution for current Serializable
                counts = [
                    list(props._CLS_TO_PRIMITIVE_FIELD_COUNT.items()),
                    list(props._CLS_TO_NON_PRIMITIVE_FIELD_COUNT.items()),
                ]
                field_count_data = dump_msgpack_with_extensions(counts)
                _field_count_cache[obj_class] = field_count_data

            self.put_public_data(context, field_count_key, field_count_data)

        return [props._FULL_CLASS_NAME, primitive_vals], [compound_vals], False

    cdef void _set_field_value(self, obj, Field field, value):
        """Set a single field value during deserialization"""
        # Use type check instead of identity check for _no_field_value
        # to handle cases where module is reloaded
        if type(value) is _NoFieldValue:
            return
        if type(value) is Placeholder:
            if field._on_deserialize is not None:
                (<Placeholder>value).callbacks.append(
                    lambda v, f=field, od=field._on_deserialize: f.set(obj, od(v))
                )
            else:
                (<Placeholder>value).callbacks.append(
                    lambda v, f=field: f.set(obj, v)
                )
        else:
            if field._on_deserialize is not None:
                field.set(obj, field._on_deserialize(value))
            else:
                field.set(obj, value)

    cdef list _prune_server_fields(
        self,
        dict client_cls_to_field_count,
        server_cls_to_field_count,
        list server_fields,
        unordered_map[int, int] legacy_to_new_hash,
    ):
        """Prune server fields to match client field distribution"""
        cdef unordered_map[int, int] new_to_legacy_hash
        cdef list ret_server_fields = []
        cdef int server_pos = 0
        cdef int count
        cdef int cls_hash
        cdef int legacy_hash
        cdef list client_keys
        cdef bint all_match
        cdef pair[int, int] hash_pair

        # Fast path: if keys match exactly, return server_fields directly
        if client_cls_to_field_count is server_cls_to_field_count:
            return server_fields

        # Optimized comparison: check length and keys without creating sets
        client_keys = list(client_cls_to_field_count.keys())

        if len(client_keys) == len(server_cls_to_field_count):
            # Check if all keys match
            all_match = True
            for cls_hash in client_keys:
                if cls_hash not in server_cls_to_field_count:
                    all_match = False
                    break
            if all_match:
                return server_fields

        for hash_pair in legacy_to_new_hash:
            new_to_legacy_hash[hash_pair.second] = hash_pair.first

        server_pos = 0
        for cls_hash, count in server_cls_to_field_count.items():
            legacy_hash = -1
            if new_to_legacy_hash.find(cls_hash) != new_to_legacy_hash.end():
                legacy_hash = new_to_legacy_hash[cls_hash]
            if (
                cls_hash in client_cls_to_field_count
                or legacy_hash in client_cls_to_field_count
            ):
                ret_server_fields.extend(server_fields[server_pos : server_pos + count])
            server_pos += count
        return ret_server_fields

    cdef _set_field_values(
        self,
        obj,
        list values,
        dict client_cls_to_field_count,
        bint is_primitive=True,
    ):
        """Set multiple field values during deserialization"""
        cdef object obj_class = type(obj)
        cdef SerializableProps props = obj_class._PROPS
        cdef object server_cls_to_field_count
        cdef list field_def_list
        cdef list server_fields
        cdef int field_num = 0
        cdef int server_field_num = 0
        cdef int count
        cdef int cls_hash
        cdef list cls_fields
        cdef list cls_values
        cdef object field
        cdef object value
        cdef bint check_default

        if is_primitive:
            server_cls_to_field_count = props._CLS_TO_PRIMITIVE_FIELD_COUNT
            field_def_list = props._PRIMITIVE_FIELDS
        else:
            server_cls_to_field_count = props._CLS_TO_NON_PRIMITIVE_FIELD_COUNT
            field_def_list = props._NON_PRIMITIVE_FIELDS

        server_fields = self._prune_server_fields(
            client_cls_to_field_count,
            server_cls_to_field_count,
            field_def_list,
            props._LEGACY_TO_NEW_NAME_HASH,
        )

        field_num = 0
        server_field_num = 0
        check_default = not is_primitive
        for cls_hash, count in client_cls_to_field_count.items():
            # cut values and fields given field distribution
            # at client and server end
            cls_fields = server_fields[server_field_num : server_field_num + count]
            cls_values = values[field_num : field_num + count]
            for field, value in zip(cls_fields, cls_values):
                # make it compatible with legacy serialization where
                #  handling of no_default for msgpack is not added
                if is_primitive and type(value) is dict and len(<dict>value) == 0:
                    value = no_default
                # Skip _no_field_value using type check (handles module reload)
                # For non-primitive fields, also skip no_default
                if type(value) is not _NoFieldValue and (
                    check_default or value is not no_default
                ):
                    self._set_field_value(obj, field, value)
            field_num += count
            try:
                server_field_num += server_cls_to_field_count[cls_hash]
            except KeyError:
                # Try looking up in the C++ unordered_map using .count()
                if props._LEGACY_TO_NEW_NAME_HASH.find(
                    cls_hash
                ) != props._LEGACY_TO_NEW_NAME_HASH.end():
                    legacy_hash = props._LEGACY_TO_NEW_NAME_HASH[cls_hash]
                    try:
                        server_field_num += server_cls_to_field_count[legacy_hash]
                    except KeyError:
                        # it is possible that certain type of field does not exist
                        #  at server side
                        pass
                else:
                    # it is possible that certain type of field does not exist
                    #  at server side
                    pass

    cpdef deserial(self, list serialized, dict context, list subs):
        """Deserialize a Serializable object"""
        global _Serializable_class

        cdef object obj_class_name = serialized[0]
        cdef object primitives = serialized[1]
        cdef object obj_class
        cdef object obj
        cdef str field_count_key
        cdef object field_count_data
        cdef list cls_to_prim_key
        cdef list cls_to_non_prim_key
        cdef dict cls_to_prim_key_dict
        cdef dict cls_to_non_prim_key_dict
        cdef object ret
        cdef SerializableProps props

        # Cache Serializable class to avoid repeated imports
        if _Serializable_class is None:
            from maxframe.serialization.serializables.core import Serializable
            _Serializable_class = Serializable

        obj_class = load_type(obj_class_name, _Serializable_class)

        if type(primitives) is not list:
            primitives = load_msgpack_with_extensions(primitives)

        obj = obj_class.__new__(obj_class)
        props = obj_class._PROPS

        field_count_key = "FC_" + str(props._NAME_HASH)
        field_count_data = self.get_public_data(context, field_count_key)
        if field_count_data is None:
            # try using legacy field count key to get counts
            field_count_key = "FC_" + str(props._LEGACY_NAME_HASH)
            field_count_data = self.get_public_data(context, field_count_key)

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

        if isinstance(field_count_data, bytes):
            cls_to_prim_key, cls_to_non_prim_key = load_msgpack_with_extensions(
                field_count_data
            )
            field_count_data = (dict(cls_to_prim_key), dict(cls_to_non_prim_key))
            self.put_public_data(context, field_count_key, field_count_data)

        cls_to_prim_key_dict, cls_to_non_prim_key_dict = field_count_data
        if primitives:
            self._set_field_values(obj, primitives, cls_to_prim_key_dict, True)

        if props._NON_PRIMITIVE_FIELDS:
            self._set_field_values(obj, subs[0], cls_to_non_prim_key_dict, False)
        ret = obj.__on_deserialize__()
        return obj if ret is None else ret
