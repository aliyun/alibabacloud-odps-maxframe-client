# distutils: language = c++
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

import asyncio
import contextvars
import copy
import datetime
import hashlib
import importlib
import os
import re
from collections import OrderedDict
from functools import partial, wraps
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from cpython cimport PyObject
from libc.stdint cimport int32_t, int64_t, uint32_t, uint64_t, uintptr_t
from libcpp.unordered_map cimport unordered_map

from pandas.api.extensions import ExtensionDtype
from pandas.api.types import pandas_dtype

from .._utils import NamedType

from .._utils cimport TypeDispatcher

from ..lib import wrapped_pickle as pickle
from ..lib.dtypes_extension import ArrowDtype
from ..utils import NoDefault, arrow_type_from_str, no_default, str_to_bool

# resolve pandas pickle compatibility between <1.2 and >=1.3
try:
    from pandas.core.internals import blocks as pd_blocks
    if not hasattr(pd_blocks, "new_block") and hasattr(pd_blocks, "make_block"):
        # register missing func that would cause errors
        pd_blocks.new_block = pd_blocks.make_block
except (ImportError, AttributeError):
    pass

try:
    import pyarrow as pa
except ImportError:
    pa = None

try:
    import pytz
    from pytz import BaseTzInfo as PyTZ_BaseTzInfo
except ImportError:
    PyTZ_BaseTzInfo = type(None)
try:
    import zoneinfo
    from zoneinfo import ZoneInfo
except ImportError:
    ZoneInfo = type(None)

BUFFER_PICKLE_PROTOCOL = max(pickle.DEFAULT_PROTOCOL, 5)
cdef bint HAS_PICKLE_BUFFER = pickle.HIGHEST_PROTOCOL >= 5
cdef bint _PANDAS_HAS_MGR = hasattr(pd.Series([0]), "_mgr")
cdef bint _ARROW_DTYPE_NOT_SUPPORTED = ArrowDtype is None


cdef TypeDispatcher _serial_dispatcher = TypeDispatcher()
cdef dict _deserializers = dict()

cdef uint32_t _MAX_STR_PRIMITIVE_LEN = 1024
# prime modulus for serializer ids
# use the largest prime number smaller than 32767
cdef int32_t _SERIALIZER_ID_PRIME = 32749

# ids for basic serializers
cdef:
    int PICKLE_SERIALIZER = 0
    int PRIMITIVE_SERIALIZER = 1
    int BYTES_SERIALIZER = 2
    int STR_SERIALIZER = 3
    int TUPLE_SERIALIZER = 4
    int LIST_SERIALIZER = 5
    int DICT_SERIALIZER = 6
    int PY_DATETIME_SERIALIZER = 7
    int PY_DATE_SERIALIZER = 8
    int PY_TIMEDELTA_SERIALIZER = 9
    int PY_TZINFO_SERIALIZER = 10
    int DTYPE_SERIALIZER = 11
    int COMPLEX_SERIALIZER = 12
    int SLICE_SERIALIZER = 13
    int REGEX_SERIALIZER = 14
    int NO_DEFAULT_SERIALIZER = 15
    int ARROW_BUFFER_SERIALIZER = 16
    int RANGE_SERIALIZER = 17
    int PLACEHOLDER_SERIALIZER = 4096


cdef dict _type_cache = dict()


cdef object pickle_serial_hook = contextvars.ContextVar("pickle_serial_hook", default=None)
cdef object pickle_deserial_hook = contextvars.ContextVar("pickle_deserial_hook", default=None)

cdef class PickleHookOptions:
    cdef:
        object _serial_hook
        object _pre_serial_hook
        object _deserial_hook
        object _pre_deserial_hook

    def __init__(self, serial_hook: object = None, deserial_hook: object = None):
        self._serial_hook = serial_hook
        self._deserial_hook = deserial_hook

    def __enter__(self):
        self._pre_serial_hook = pickle_serial_hook.set(self._serial_hook)
        self._pre_deserial_hook = pickle_deserial_hook.set(self._deserial_hook)

    def __exit__(self, exc_type, exc_val, exc_tb):
        pickle_serial_hook.reset(self._pre_serial_hook)
        pickle_deserial_hook.reset(self._pre_deserial_hook)


cdef bint unpickle_allowed


def reload_unpickle_flag():
    global unpickle_allowed
    unpickle_allowed = str_to_bool(
        os.getenv("MAXFRAME_SERIALIZE_UNPICKLE_ALLOWED", "1")
    )


reload_unpickle_flag()


cdef object _load_by_name(str class_name):
    if class_name in _type_cache:
        cls = _type_cache[class_name]
    else:
        try:
            from .deserializer import safe_load_by_name

            cls = safe_load_by_name(class_name)
        except ImportError:
            if pickle.is_unpickle_forbidden():
                raise

            mod_name, cls_name = class_name.rsplit("#", 1)

            try:
                cls = importlib.import_module(mod_name)
            except ImportError as ex:
                raise ImportError(
                    f"Failed to import {mod_name} when loading "
                    f"class {class_name}, {ex}"
                ) from None

            for sub_cls_name in cls_name.split("."):
                cls = getattr(cls, sub_cls_name)
        _type_cache[class_name] = cls
    return cls


cpdef object load_type(str class_name, object parent_class):
    cls = _load_by_name(class_name)
    if not isinstance(cls, type):
        raise ValueError(f"Class {class_name} not a type, cannot be deserialized")
    if not issubclass(cls, parent_class):
        raise ValueError(f"Class {class_name} not a {parent_class}")
    return cls


cpdef object load_member(str class_name, object restrict_type):
    member = _load_by_name(class_name)
    if not isinstance(member, restrict_type):
        raise ValueError(
            f"Class {class_name} not a {restrict_type}, cannot be deserialized"
        )
    return member


cpdef void clear_type_cache():
    _type_cache.clear()


cdef Serializer get_deserializer(int32_t deserializer_id):
    return _deserializers[deserializer_id]


cdef class Serializer:
    serializer_id = None
    _public_data_context_key = 0x7fffffff - 1

    def __cinit__(self):
        # make the value can be referenced with C code
        self._serializer_id = self.serializer_id

    cpdef bint is_public_data_exist(self, dict context, object key):
        cdef dict public_dict = context.get(self._public_data_context_key, None)
        if public_dict is None:
            return False
        return key in public_dict

    cpdef put_public_data(self, dict context, object key, object value):
        cdef dict public_dict = context.get(self._public_data_context_key, None)
        if public_dict is None:
            public_dict = context[self._public_data_context_key] = {}
        public_dict[key] = value

    cpdef get_public_data(self, dict context, object key):
        cdef dict public_dict = context.get(self._public_data_context_key, None)
        if public_dict is None:
            return None
        return public_dict.get(key)

    cpdef serial(self, object obj, dict context):
        """
        Returns intermediate serialization result of certain object.
        The returned value can be a Placeholder or a tuple comprising
        of three parts: a header, a group of subcomponents and
        a finalizing flag.

        * Header is a pickle-serializable tuple
        * Subcomponents are parts or buffers for iterative
          serialization.
        * Flag is a boolean value. If true, subcomponents should be
          buffers (for instance, bytes, memory views, GPU buffers,
          etc.) that can be read and written directly. If false,
          subcomponents will be serialized iteratively.

        Parameters
        ----------
        obj: Any
            Object to serialize
        context: Dict
            Serialization context to help creating Placeholder objects
            for reducing duplicated serialization

        Returns
        -------
        result: Placeholder | Tuple[Tuple, List, bool]
            Intermediate result of serialization
        """
        raise NotImplementedError

    cpdef deserial(self, list serialized, dict context, list subs):
        """
        Returns deserialized object given serialized headers and
        deserialized subcomponents.

        Parameters
        ----------
        serialized: List
            Serialized object header as a tuple
        context
            Serialization context for instantiation of Placeholder
            objects
        subs: List
            Deserialized subcomponents

        Returns
        -------
        result: Any
            Deserialized objects
        """
        raise NotImplementedError

    cpdef on_deserial_error(
        self,
        list serialized,
        dict context,
        list subs_serialized,
        int error_index,
        object exc,
    ):
        """
        Returns rewritten exception when subcomponent deserialization fails

        Parameters
        ----------
        serialized: List
            Serialized object header as a tuple
        context
            Serialization context for instantiation of Placeholder
            objects
        subs_serialized: List
            Serialized subcomponents
        error_index: int
            Index of subcomponent causing error
        exc: BaseException
            Exception raised

        Returns
        -------
        exc: BaseException | None
            Rewritten exception. If None, original exception is kept.
        """
        return None

    @classmethod
    def calc_default_serializer_id(cls):
        s = f"{cls.__module__}.{cls.__qualname__}"
        h = hashlib.md5(s.encode())
        return int(h.hexdigest(), 16) % _SERIALIZER_ID_PRIME

    @classmethod
    def register(cls, obj_type, name=None):
        if (
            cls.serializer_id is None
            or cls.serializer_id == getattr(super(cls, cls), "serializer_id", None)
        ):
            # a class should have its own serializer_id
            # inherited serializer_id not acceptable
            cls.serializer_id = cls.calc_default_serializer_id()

        inst = cls()
        if name is not None:
            obj_type = NamedType(name, obj_type)
        _serial_dispatcher.register(obj_type, inst)
        if _deserializers.get(cls.serializer_id) is not None:
            assert type(_deserializers[cls.serializer_id]) is cls
        else:
            _deserializers[cls.serializer_id] = inst

    @classmethod
    def unregister(cls, obj_type, name=None):
        if name is not None:
            obj_type = NamedType(name, obj_type)
        _serial_dispatcher.unregister(obj_type)
        _deserializers.pop(cls.serializer_id, None)

    @classmethod
    def dump_handlers(cls):
        return _serial_dispatcher.dump_handlers()

    @classmethod
    def load_handlers(cls, *args):
        _serial_dispatcher.load_handlers(*args)


cdef inline uint64_t _fast_id(PyObject * obj) nogil:
    return <uintptr_t>obj


def fast_id(obj):
    """C version of id() used for serialization"""
    return _fast_id(<PyObject *>obj)


def buffered(func):
    """
    Wrapper for serial() method to reduce duplicated serialization
    """
    @wraps(func)
    def wrapped(self, obj: Any, dict context):
        cdef uint64_t obj_id = _fast_id(<PyObject*>obj)
        if obj_id in context:
            return Placeholder(_fast_id(<PyObject*>obj))
        else:
            context[obj_id] = obj
            return func(self, obj, context)

    return wrapped


def pickle_buffers(obj):
    cdef list buffers = [None]

    if HAS_PICKLE_BUFFER:

        def buffer_cb(x):
            x = x.raw()
            if x.ndim > 1:
                # ravel n-d memoryview
                x = x.cast(x.format)
            buffers.append(memoryview(x))

        buffers[0] = pickle.dumps(
            obj,
            buffer_callback=buffer_cb,
            protocol=BUFFER_PICKLE_PROTOCOL,
        )
    else:  # pragma: no cover
        buffers[0] = pickle.dumps(obj)
    return buffers


def unpickle_buffers(buffers):
    result = pickle.loads(buffers[0], buffers=buffers[1:])

    # as pandas prior to 1.1.0 use _data instead of _mgr to hold BlockManager,
    # deserializing from high versions may produce mal-functioned pandas objects,
    # thus the patch is needed
    if _PANDAS_HAS_MGR:
        return result
    else:  # pragma: no cover
        if hasattr(result, "_mgr") and isinstance(result, (pd.DataFrame, pd.Series)):
            result._data = getattr(result, "_mgr")
            delattr(result, "_mgr")
        return result


cdef class PickleContainer:
    cdef:
        list buffers

    def __init__(self, list buffers):
        self.buffers = buffers

    cpdef get(self):
        if not unpickle_allowed:
            raise ValueError("Unpickle not allowed in this environment")
        return unpickle_buffers(self.buffers)

    cpdef list get_buffers(self):
        return self.buffers

    def __copy__(self):
        return PickleContainer(self.buffers)

    def __deepcopy__(self, memo=None):
        return PickleContainer(copy.deepcopy(self.buffers, memo))

    def __maxframe_tokenize__(self):
        return self.buffers

    def __reduce__(self):
        return PickleContainer, (self.buffers, )


cdef class PickleSerializer(Serializer):
    serializer_id = PICKLE_SERIALIZER

    cpdef serial(self, obj: Any, dict context):
        cdef uint64_t obj_id
        cdef object serial_hook

        serial_hook = pickle_serial_hook.get()
        if serial_hook is not None:
            serial_hook()

        obj_id = _fast_id(<PyObject*>obj)
        if obj_id in context:
            return Placeholder(obj_id)
        context[obj_id] = obj

        if type(obj) is PickleContainer:
            return [], (<PickleContainer>obj).get_buffers(), True
        return [], pickle_buffers(obj), True

    cpdef deserial(self, list serialized, dict context, list subs):
        from .deserializer import deserial_pickle
        cdef object deserial_hook

        deserial_hook = pickle_deserial_hook.get()
        if deserial_hook is not None:
            deserial_hook()
        return deserial_pickle(serialized, context, subs)


cdef set _primitive_types = {
    type(None),
    bool,
    int,
    float,
}


cdef class PrimitiveSerializer(Serializer):
    serializer_id = PRIMITIVE_SERIALIZER

    cpdef serial(self, object obj, dict context):
        return [obj,], [], True

    cpdef deserial(self, list obj, dict context, list subs):
        return obj[0]


cdef class BytesSerializer(Serializer):
    serializer_id = BYTES_SERIALIZER

    cpdef serial(self, obj: Any, dict context):
        cdef uint64_t obj_id
        obj_id = _fast_id(<PyObject*>obj)
        if obj_id in context:
            return Placeholder(obj_id)
        context[obj_id] = obj

        return [], [obj], True

    cpdef deserial(self, list serialized, dict context, list subs):
        return subs[0]


cdef class StrSerializer(Serializer):
    serializer_id = STR_SERIALIZER

    cpdef serial(self, obj: Any, dict context):
        cdef uint64_t obj_id
        obj_id = _fast_id(<PyObject*>obj)
        if obj_id in context:
            return Placeholder(obj_id)
        context[obj_id] = obj

        return [], [(<str>obj).encode()], True

    cpdef deserial(self, list serialized, dict context, list subs):
        buffer = subs[0]
        if type(buffer) is memoryview:
            buffer = buffer.tobytes()
        return buffer.decode()


cdef class CollectionSerializer(Serializer):
    obj_type = None

    cdef object _obj_type

    def __cinit__(self):
        # make the value can be referenced with C code
        self._obj_type = self.obj_type

    cdef tuple _serial_iterable(self, obj: Any):
        cdef list idx_to_propagate = []
        cdef list obj_to_propagate = []
        cdef list obj_list = <list>obj if type(obj) is list else list(obj)
        cdef int64_t idx
        cdef object item

        for idx in range(len(obj_list)):
            item = obj_list[idx]

            if type(item) is bytes and len(<bytes>item) < _MAX_STR_PRIMITIVE_LEN:
                # treat short strings as primitives
                continue
            elif type(item) is str and len(<str>item) < _MAX_STR_PRIMITIVE_LEN:
                # treat short strings as primitives
                continue
            elif type(item) in _primitive_types:
                continue

            if obj is obj_list:
                obj_list = list(obj)

            obj_list[idx] = None
            idx_to_propagate.append(idx)
            obj_to_propagate.append(item)

        return [obj_list, idx_to_propagate], obj_to_propagate, False

    cpdef serial(self, obj: Any, dict context):
        cdef uint64_t obj_id
        obj_id = _fast_id(<PyObject*>obj)
        if obj_id in context:
            return Placeholder(obj_id)
        context[obj_id] = obj

        return self._serial_iterable(obj)

    cdef list _deserial_iterable(self, list serialized, list subs):
        cdef list res_list, idx_to_propagate
        cdef int64_t i

        res_list, idx_to_propagate = serialized

        for i in range(len(idx_to_propagate)):
            res_list[idx_to_propagate[i]] = subs[i]
        return res_list


cdef class TupleSerializer(CollectionSerializer):
    serializer_id = TUPLE_SERIALIZER
    obj_type = tuple

    cpdef serial(self, obj: Any, dict context):
        cdef uint64_t obj_id
        cdef list header
        cdef object data, is_leaf

        obj_id = _fast_id(<PyObject*>obj)
        if obj_id in context:
            return Placeholder(obj_id)
        context[obj_id] = obj

        header, data, is_leaf = self._serial_iterable(obj)
        if hasattr(type(obj), "_fields"):
            header.append(type(obj).__module__ + "#" + type(obj).__qualname__)
        else:
            header.append(None)
        return header, data, is_leaf

    cpdef deserial(self, list serialized, dict context, list subs):
        cdef list res
        cdef str tuple_type_name = serialized[-1]

        res = self._deserial_iterable(serialized[:-1], subs)
        for v in res:
            assert type(v) is not Placeholder

        if tuple_type_name is None:
            return tuple(res)
        else:
            tuple_type = load_type(tuple_type_name, tuple)
            return tuple_type(*res)


cdef class ListSerializer(CollectionSerializer):
    serializer_id = LIST_SERIALIZER
    obj_type = list

    cpdef deserial(self, list serialized, dict context, list subs):
        cdef int64_t idx
        cdef list res = self._deserial_iterable(serialized, subs)

        result = list(res)

        for idx, v in enumerate(res):
            if type(v) is Placeholder:
                cb = partial(result.__setitem__, idx)
                (<Placeholder>v).callbacks.append(cb)
        return result


def _dict_key_replacer(ret, key, real_key):
    ret[real_key] = ret.pop(key)


def _dict_value_replacer(context, ret, key, real_value):
    if type(key) is Placeholder:
        key = context[(<Placeholder>key).id]
    ret[key] = real_value


cdef:
    object _TYPE_CHAR_ORDERED_DICT = "O"


cdef class DictSerializer(CollectionSerializer):
    serializer_id = DICT_SERIALIZER

    cpdef serial(self, obj: Any, dict context):
        cdef uint64_t obj_id
        cdef list key_obj, value_obj
        cdef list key_bufs, value_bufs

        if type(obj) is dict and len(<dict>obj) == 0:
            return [], [], True

        obj_id = _fast_id(<PyObject*>obj)
        if obj_id in context:
            return Placeholder(obj_id)
        context[obj_id] = obj

        if isinstance(obj, OrderedDict):
            ser_type = _TYPE_CHAR_ORDERED_DICT
        else:
            ser_type = None

        key_obj, key_bufs, _ = self._serial_iterable(obj.keys())
        value_obj, value_bufs, _ = self._serial_iterable(obj.values())
        ser_obj = [key_obj, value_obj, len(key_bufs), ser_type]
        return ser_obj, key_bufs + value_bufs, False

    cpdef deserial(self, list serialized, dict context, list subs):
        cdef int64_t i, num_key_bufs
        cdef list key_subs, value_subs, keys, values

        if not serialized:
            return {}
        if len(serialized) == 1:
            # serialized directly
            return serialized[0]

        key_serialized, value_serialized, num_key_bufs, ser_type = serialized
        key_subs = subs[:num_key_bufs]
        value_subs = subs[num_key_bufs:]

        keys = self._deserial_iterable(<list>key_serialized, key_subs)
        values = self._deserial_iterable(<list>value_serialized, value_subs)

        if ser_type == _TYPE_CHAR_ORDERED_DICT:
            ret = OrderedDict(zip(keys, values))
        else:
            ret = dict(zip(keys, values))

        for i in range(len(keys)):
            k, v = keys[i], values[i]
            if type(k) is Placeholder:
                (<Placeholder>k).callbacks.append(
                    partial(_dict_key_replacer, ret, k)
                )
            if type(v) is Placeholder:
                (<Placeholder>v).callbacks.append(
                    partial(_dict_value_replacer, context, ret, k)
                )
        return ret


cdef class PyDatetimeSerializer(Serializer):
    serializer_id = PY_DATETIME_SERIALIZER

    cpdef serial(self, obj: datetime.datetime, dict context):
        cdef list ser_tz = (
            _serial_tz(obj.tzinfo) if obj.tzinfo is not None else None
        )
        return [obj.timestamp(), ser_tz], [], True

    cpdef deserial(self, list serialized, dict context, list subs):
        cdef object tz = (
            _deserialize_tz(serialized[1]) if serialized[1] is not None else None
        )
        return datetime.datetime.fromtimestamp(serialized[0], tz)


cdef class PyDateSerializer(Serializer):
    serializer_id = PY_DATE_SERIALIZER

    cpdef serial(self, obj: datetime.date, dict context):
        return [obj.toordinal()], [], True

    cpdef deserial(self, list serialized, dict context, list subs):
        return datetime.date.fromordinal(serialized[0])


cdef class PyTimedeltaSerializer(Serializer):
    serializer_id = PY_TIMEDELTA_SERIALIZER

    cpdef serial(self, obj: datetime.timedelta, dict context):
        return [obj.days, obj.seconds, obj.microseconds], [], True

    cpdef deserial(self, list serialized, dict context, list subs):
        return datetime.timedelta(
            days=serialized[0],
            seconds=serialized[1],
            microseconds=serialized[2],
        )


cdef:
    object _TYPE_CHAR_TZ_BASE = "S"
    object _TYPE_CHAR_TZ_ZONEINFO = "ZI"
    object _TYPE_CHAR_TZ_PYTZ = "PT"


cdef inline list _serial_tz(
    obj: datetime.tzinfo, dt: Optional[datetime.datetime] = None
):
    cdef object type_char
    if isinstance(obj, PyTZ_BaseTzInfo):
        return [_TYPE_CHAR_TZ_PYTZ, obj.zone]
    elif isinstance(obj, ZoneInfo):
        return [_TYPE_CHAR_TZ_ZONEINFO, obj.key]
    else:
        dt = dt or datetime.datetime.now()
        return [
            _TYPE_CHAR_TZ_BASE,
            obj.tzname(dt),
            int(obj.utcoffset(dt).total_seconds()),
        ]


cdef inline object _deserialize_tz(list serialized):
    if serialized[0] == _TYPE_CHAR_TZ_PYTZ:
        return pytz.timezone(serialized[1])
    elif serialized[0] == _TYPE_CHAR_TZ_ZONEINFO:
        return zoneinfo.ZoneInfo(serialized[1])
    else:
        if serialized[2] == 0:
            return datetime.timezone.utc
        return datetime.timezone(
            datetime.timedelta(seconds=serialized[2]), name=serialized[1]
        )


cdef class TZInfoSerializer(Serializer):
    serializer_id = PY_TZINFO_SERIALIZER

    cpdef serial(self, object obj: datetime.tzinfo, dict context):
        return _serial_tz(obj), [], True

    cpdef deserial(self, list serialized, dict context, list subs):
        return _deserialize_tz(serialized)


cdef:
    object _TYPE_CHAR_DTYPE_NUMPY = "N"
    object _TYPE_CHAR_DTYPE_PANDAS_ARROW = "PA"
    object _TYPE_CHAR_DTYPE_PANDAS_CATEGORICAL = "PC"
    object _TYPE_CHAR_DTYPE_PANDAS_INTERVAL = "PI"
    object _TYPE_CHAR_DTYPE_PANDAS_EXTENSION = "PE"


cdef class DtypeSerializer(Serializer):
    serializer_id = DTYPE_SERIALIZER

    @staticmethod
    def _sort_fields(list fields):
        return sorted(fields, key=lambda k: fields[k][1])

    cpdef serial(self, obj: Union[np.dtype, ExtensionDtype], dict context):
        if isinstance(obj, np.dtype):
            try:
                return [
                    _TYPE_CHAR_DTYPE_NUMPY, np.lib.format.dtype_to_descr(obj), None
                ], [], True
            except ValueError:
                fields = obj.fields
                new_fields = self._sort_fields(fields)
                desc = np.lib.format.dtype_to_descr(obj[new_fields])
                dtype_new_order = list(fields)
                return [_TYPE_CHAR_DTYPE_NUMPY, desc, dtype_new_order], [], True
        elif isinstance(obj, ExtensionDtype):
            if _ARROW_DTYPE_NOT_SUPPORTED:
                raise ImportError("ArrowDtype is not supported in current environment")
            if isinstance(obj, ArrowDtype):
                return [_TYPE_CHAR_DTYPE_PANDAS_ARROW, str(obj.pyarrow_dtype)], [], True
            elif isinstance(obj, pd.CategoricalDtype):
                return [
                    _TYPE_CHAR_DTYPE_PANDAS_CATEGORICAL, obj.ordered
                ], [obj.categories], False
            elif isinstance(obj, pd.IntervalDtype):
                return [
                    _TYPE_CHAR_DTYPE_PANDAS_INTERVAL, obj.closed
                ], [obj.subdtype], False
            else:
                return [_TYPE_CHAR_DTYPE_PANDAS_EXTENSION, repr(obj)], [], True
        else:
            raise NotImplementedError(f"Does not support serializing dtype {obj!r}")

    cpdef deserial(self, list serialized, dict context, list subs):
        cdef str ser_type = serialized[0]
        if ser_type == _TYPE_CHAR_DTYPE_NUMPY:
            try:
                dt = np.lib.format.descr_to_dtype(serialized[1])
            except AttributeError:
                dt = np.dtype(serialized[1])

            if serialized[2] is not None:
                # fill dtype_new_order field
                dt = dt[serialized[2]]
            return dt
        elif ser_type == _TYPE_CHAR_DTYPE_PANDAS_ARROW:
            if _ARROW_DTYPE_NOT_SUPPORTED:
                raise ImportError("ArrowDtype is not supported in current environment")
            return ArrowDtype(arrow_type_from_str(serialized[1]))
        elif ser_type == _TYPE_CHAR_DTYPE_PANDAS_CATEGORICAL:
            return pd.CategoricalDtype(subs[0], serialized[1])
        elif ser_type == _TYPE_CHAR_DTYPE_PANDAS_INTERVAL:
            return pd.IntervalDtype(subs[0], serialized[1])
        elif ser_type == _TYPE_CHAR_DTYPE_PANDAS_EXTENSION:
            if serialized[1] == "StringDtype":  # for legacy pandas version
                return pd.StringDtype()
            return pandas_dtype(serialized[1])
        else:
            raise NotImplementedError(f"Unknown serialization type {ser_type}")


cdef class ComplexSerializer(Serializer):
    serializer_id = COMPLEX_SERIALIZER

    cpdef serial(self, object obj: complex, dict context):
        cdef complex cplx = <complex>obj
        return [cplx.real, cplx.imag], [], True

    cpdef deserial(self, list serialized, dict context, list subs):
        return complex(*serialized[:2])


cdef class SliceSerializer(Serializer):
    serializer_id = SLICE_SERIALIZER

    cpdef serial(self, object obj: slice, dict context):
        cdef list elems = [obj.start, obj.stop, obj.step]
        for x in elems:
            if x is not None and not isinstance(x, int):
                return [], elems, False
        return elems, [], True

    cpdef deserial(self, list serialized, dict context, list subs):
        if len(serialized) == 0:
            return slice(subs[0], subs[1], subs[2])
        return slice(*serialized[:3])


cdef class RangeSerializer(Serializer):
    serializer_id = RANGE_SERIALIZER

    cpdef serial(self, object obj: range, dict context):
        return [obj.start, obj.stop, obj.step], [], True

    cpdef deserial(self, list serialized, dict context, list subs):
        return range(*serialized[:3])


cdef class RegexSerializer(Serializer):
    serializer_id = REGEX_SERIALIZER

    cpdef serial(self, object obj: re.Pattern, dict context):
        cdef uint64_t obj_id
        obj_id = _fast_id(<PyObject*>obj)
        if obj_id in context:
            return Placeholder(obj_id)
        context[obj_id] = obj

        return [obj.flags], [(<str>(obj.pattern)).encode()], True

    cpdef deserial(self, list serialized, dict context, list subs):
        return re.compile((<bytes>(subs[0])).decode(), serialized[0])


cdef class NoDefaultSerializer(Serializer):
    serializer_id = NO_DEFAULT_SERIALIZER

    cpdef serial(self, object obj, dict context):
        return [], [], True

    cpdef deserial(self, list obj, dict context, list subs):
        return no_default


cdef class ArrowBufferSerializer(Serializer):
    serializer_id = ARROW_BUFFER_SERIALIZER

    cpdef serial(self, object obj, dict context):
        return [], [obj], True

    cpdef deserial(self, list obj, dict context, list subs):
        if not isinstance(subs[0], pa.Buffer):
            return pa.py_buffer(subs[0])
        return subs[0]


cdef class Placeholder:
    """
    Placeholder object to reduce duplicated serialization

    The object records object identifier and keeps callbacks
    to replace itself in parent objects.
    """
    def __init__(self, uint64_t id_):
        self.id = id_
        self.callbacks = []

    def __hash__(self):
        return self.id

    def __eq__(self, other):  # pragma: no cover
        if type(other) is not Placeholder:
            return False
        return self.id == other.id

    def __repr__(self):
        return (
            f"Placeholder(id={self.id}, "
            f"callbacks=[list of {len(self.callbacks)}])"
        )


cdef class PlaceholderSerializer(Serializer):
    serializer_id = PLACEHOLDER_SERIALIZER

    cpdef serial(self, obj: Any, dict context):
        return [], [], True

    cpdef deserial(self, list serialized, dict context, list subs):
        return Placeholder(0)


PickleSerializer.register(object)
for _primitive in _primitive_types:
    PrimitiveSerializer.register(_primitive)
BytesSerializer.register(bytes)
BytesSerializer.register(memoryview)
StrSerializer.register(str)
ListSerializer.register(list)
TupleSerializer.register(tuple)
DictSerializer.register(dict)
PyDatetimeSerializer.register(datetime.datetime)
PyDateSerializer.register(datetime.date)
PyTimedeltaSerializer.register(datetime.timedelta)
TZInfoSerializer.register(datetime.tzinfo)
DtypeSerializer.register(np.dtype)
DtypeSerializer.register(ExtensionDtype)
ComplexSerializer.register(complex)
SliceSerializer.register(slice)
RangeSerializer.register(range)
RegexSerializer.register(re.Pattern)
NoDefaultSerializer.register(NoDefault)
if pa is not None:
    ArrowBufferSerializer.register(pa.Buffer)
PlaceholderSerializer.register(Placeholder)


cdef class _SerialStackItem:
    cdef public list serialized
    cdef public list subs
    cdef public list subs_serialized

    def __cinit__(self, list serialized, list subs):
        self.serialized = serialized
        self.subs = subs
        self.subs_serialized = []


cdef class _IdContextHolder:
    cdef public unordered_map[uint64_t, uint64_t] d
    cdef public uint64_t obj_count

    def __cinit__(self):
        self.obj_count = 0


cdef tuple _serial_single(
    obj, dict context, _IdContextHolder id_context_holder
):
    """Serialize single object and return serialized tuples"""
    cdef uint64_t obj_id, ordered_id
    cdef Serializer serializer
    cdef int serializer_id
    cdef list common_header, serialized, subs

    while True:
        name = context.get("serializer")
        obj_type = type(obj) if name is None else NamedType(name, type(obj))
        serializer = _serial_dispatcher.get_handler(obj_type)
        serializer_id = serializer._serializer_id
        ret_serial = serializer.serial(obj, context)
        if type(ret_serial) is tuple:
            # object is serialized, form a common header and return
            serialized, subs, final = <tuple>ret_serial

            if type(obj) is Placeholder:
                obj_id = (<Placeholder>obj).id
                ordered_id = id_context_holder.d[obj_id]
            else:
                ordered_id = id_context_holder.obj_count
                id_context_holder.obj_count += 1
                # only need to record object ids for non-primitive types
                if serializer_id != PRIMITIVE_SERIALIZER:
                    obj_id = _fast_id(<PyObject*>obj)
                    id_context_holder.d[obj_id] = ordered_id

            # REMEMBER to change _COMMON_HEADER_LEN when content of
            # this header changed
            common_header = [
                serializer_id, ordered_id, len(subs), final
            ]
            break
        else:
            # object is converted into another (usually a Placeholder)
            obj = ret_serial
    common_header.extend(serialized)
    return common_header, subs, final


class _SerializeObjectOverflow(Exception):
    def __init__(self, list cur_serialized, int num_total_serialized):
        super(_SerializeObjectOverflow, self).__init__(cur_serialized)
        self.cur_serialized = cur_serialized
        self.num_total_serialized = num_total_serialized


cpdef object _serialize_with_stack(
    list serial_stack,
    list serialized,
    dict context,
    _IdContextHolder id_context_holder,
    list result_bufs_list,
    int64_t num_overflow = 0,
    int64_t num_total_serialized = 0,
):
    cdef _SerialStackItem stack_item
    cdef list subs
    cdef bint final
    cdef int64_t num_sub_serialized
    cdef bint is_resume = num_total_serialized > 0

    while serial_stack:
        stack_item = serial_stack[-1]
        if serialized is not None:
            # have previously-serialized results, record first
            stack_item.subs_serialized.append(serialized)

        num_sub_serialized = len(stack_item.subs_serialized)
        if len(stack_item.subs) == num_sub_serialized:
            # all subcomponents serialized, serialization of current is done
            # and we can move to the parent object
            serialized = stack_item.serialized + stack_item.subs_serialized
            num_total_serialized += 1
            serial_stack.pop()
        else:
            # serialize next subcomponent at stack top
            serialized, subs, final = _serial_single(
                stack_item.subs[num_sub_serialized], context, id_context_holder
            )
            num_total_serialized += 1
            if final or not subs:
                # the subcomponent is a leaf
                if subs:
                    result_bufs_list.extend(subs)
            else:
                # the subcomponent has its own subcomponents, we push itself
                # into stack and process its children
                stack_item = _SerialStackItem(serialized, subs)
                serial_stack.append(stack_item)
                # note that the serialized header should not be recorded
                # as we are now processing the subcomponent itself
                serialized = None
        if 0 < num_overflow < num_total_serialized:
            raise _SerializeObjectOverflow(serialized, num_total_serialized)

    # we keep an empty dict for extra metas required for other modules
    if is_resume:
        # returns num of deserialized objects when resumed
        extra_meta = {"_N": num_total_serialized}
    else:
        # otherwise does not record the number to reduce result size
        extra_meta = {}
    return [extra_meta, serialized], result_bufs_list


def serialize(obj, dict context = None):
    """
    Serialize an object and return a header and buffers.
    Buffers are intended for zero-copy data manipulation.

    Parameters
    ----------
    obj: Any
        Object to serialize
    context:
        Serialization context for instantiation of Placeholder
        objects

    Returns
    -------
    result: Tuple[Tuple, List]
        Picklable header and buffers
    """
    cdef list serial_stack = []
    cdef list result_bufs_list = []
    cdef list serialized
    cdef list subs
    cdef bint final
    cdef _IdContextHolder id_context_holder = _IdContextHolder()
    cdef tuple result

    context = context if context is not None else dict()
    serialized, subs, final = _serial_single(obj, context, id_context_holder)
    if final or not subs:
        # marked as a leaf node, return directly
        result = [{}, serialized], subs
    else:
        serial_stack.append(_SerialStackItem(serialized, subs))
        result = _serialize_with_stack(
            serial_stack, None, context, id_context_holder, result_bufs_list
        )
    result[0][0]["_PUB"] = context.get(Serializer._public_data_context_key)
    return result


async def serialize_with_spawn(
    obj, dict context = None, int spawn_threshold = 100, object executor = None
):
    """
    Serialize an object and return a header and buffers.
    Buffers are intended for zero-copy data manipulation.

    Parameters
    ----------
    obj: Any
        Object to serialize
    context: Dict
        Serialization context for instantiation of Placeholder
        objects
    spawn_threshold: int
        Threshold to spawn into a ThreadPoolExecutor
    executor: ThreadPoolExecutor
        ThreadPoolExecutor to spawn rest serialization into

    Returns
    -------
    result: Tuple[Tuple, List]
        Picklable header and buffers
    """
    cdef list serial_stack = []
    cdef list result_bufs_list = []
    cdef list serialized
    cdef list subs
    cdef bint final
    cdef _IdContextHolder id_context_holder = _IdContextHolder()
    cdef tuple result

    context = context if context is not None else dict()
    serialized, subs, final = _serial_single(obj, context, id_context_holder)
    if final or not subs:
        # marked as a leaf node, return directly
        result = [{}, serialized], subs
    else:
        serial_stack.append(_SerialStackItem(serialized, subs))

        try:
            result = _serialize_with_stack(
                serial_stack,
                None,
                context,
                id_context_holder,
                result_bufs_list,
                spawn_threshold,
            )
        except _SerializeObjectOverflow as ex:
            result = await asyncio.get_running_loop().run_in_executor(
                executor,
                _serialize_with_stack,
                serial_stack,
                ex.cur_serialized,
                context,
                id_context_holder,
                result_bufs_list,
                0,
                ex.num_total_serialized,
            )
    result[0][0]["_PUB"] = context.get(Serializer._public_data_context_key)
    return result


cdef object deserialize_impl


def deserialize(list serialized, list buffers, dict context = None):
    """
    Deserialize an object with serialized headers and buffers

    Parameters
    ----------
    serialized: List
        Serialized object header
    buffers: List
        List of buffers extracted from serialize() calls
    context: Dict
        Serialization context for replacing Placeholder
        objects

    Returns
    -------
    result: Any
        Deserialized object
    """
    global deserialize_impl

    if deserialize_impl is None:
        from .deserializer import deserialize as deserialize_impl

    return deserialize_impl(serialized, buffers, context)
