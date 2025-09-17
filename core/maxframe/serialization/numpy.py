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

from typing import Any, Dict, List

import numpy as np

from .core import Serializer, buffered

_TYPE_CHAR_NP_GENERIC = "G"


class NDArraySerializer(Serializer):
    @buffered
    def serial(self, obj: np.generic, context: Dict):
        order = "C"
        if obj.flags.f_contiguous:
            order = "F"
        elif not obj.flags.c_contiguous:
            obj = np.ascontiguousarray(obj)
        try:
            desc = np.lib.format.dtype_to_descr(obj.dtype)
            dtype_new_order = None
        except ValueError:
            # for structured dtype, array[[field2, field1]] will create a view,
            # and dtype_to_desc will fail due to the order
            fields = obj.dtype.fields
            new_fields = sorted(fields, key=lambda k: fields[k][1])
            desc = np.lib.format.dtype_to_descr(obj.dtype[new_fields])
            dtype_new_order = list(fields)
        type_char = _TYPE_CHAR_NP_GENERIC if not isinstance(obj, np.ndarray) else None

        header = dict(
            type=type_char,
            descr=desc,
            dtype_new_order=dtype_new_order,
            shape=list(obj.shape),
            strides=list(obj.strides),
            order=order,
        )
        flattened = obj.ravel(order=order)
        if obj.dtype.hasobject:
            is_leaf = False
            data = flattened.tolist()
        else:
            is_leaf = True
            data = memoryview(flattened.view("uint8").data)
        return [header], [data], is_leaf

    def deserial(self, serialized: List, context: Dict, subs: List[Any]):
        header = serialized[0]
        try:
            dtype = np.lib.format.descr_to_dtype(header["descr"])
        except AttributeError:  # pragma: no cover
            # for older numpy versions, descr_to_dtype is not implemented
            dtype = np.dtype(header["descr"])

        dtype_new_order = header["dtype_new_order"]
        if dtype_new_order:
            dtype = dtype[dtype_new_order]
        if dtype.hasobject:
            shape = tuple(header["shape"])
            if shape == ():
                val = np.array(subs[0]).reshape(shape)
            else:
                # fill empty object array
                val = np.empty(shape, dtype=dtype)
                try:
                    val[(slice(None),) * len(shape)] = subs[0]
                except ValueError:
                    val[(slice(None),) * len(shape)] = np.array(
                        subs[0], dtype=dtype
                    ).reshape(shape)
        else:
            val = np.ndarray(
                shape=tuple(header["shape"]),
                dtype=dtype,
                buffer=subs[0],
                strides=tuple(header["strides"]),
                order=header["order"],
            )
        if header.get("type") == _TYPE_CHAR_NP_GENERIC:
            return np.take(val, 0)
        return val


class RandomStateSerializer(Serializer):
    def serial(self, obj: np.random.RandomState, context: Dict):
        return [], [obj.get_state()], False

    def deserial(self, serialized, context: Dict, subs: List):
        rs = np.random.RandomState()
        rs.set_state(subs[0])
        return rs


NDArraySerializer.register(np.generic)
NDArraySerializer.register(np.ndarray)
RandomStateSerializer.register(np.random.RandomState)
