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

from libc.stdint cimport int32_t, uint64_t


cdef bint unpickle_allowed


cdef class Serializer:
    cdef int _serializer_id

    cpdef bint is_public_data_exist(self, dict context, object key)
    cpdef put_public_data(self, dict context, object key, object value)
    cpdef get_public_data(self, dict context, object key)
    cpdef serial(self, object obj, dict context)
    cpdef deserial(self, list serialized, dict context, list subs)
    cpdef on_deserial_error(
        self,
        list serialized,
        dict context,
        list subs_serialized,
        int error_index,
        object exc,
    )


cdef class Placeholder:
    """
    Placeholder object to reduce duplicated serialization

    The object records object identifier and keeps callbacks
    to replace itself in parent objects.
    """
    cdef public uint64_t id
    cdef public list callbacks


cdef Serializer get_deserializer(int32_t deserializer_id)
