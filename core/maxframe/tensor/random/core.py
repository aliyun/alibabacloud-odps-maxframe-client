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

from collections.abc import Iterable
from contextlib import contextmanager
from typing import List

import numpy as np

from ...core import EntityData
from ...serialization.serializables import FieldTypes, Int32Field, TupleField
from ...utils import on_deserialize_shape, on_serialize_shape
from ..core import TENSOR_TYPE
from ..datasource import tensor as astensor
from ..misc import broadcast_to
from ..operators import TensorMapReduceOperator, TensorOperator, TensorOperatorMixin
from ..utils import broadcast_shape


class RandomState:
    def __init__(self, seed=None):
        self._random_state = np.random.RandomState(seed=seed)

    def seed(self, seed=None):
        """
        Seed the generator.

        This method is called when `RandomState` is initialized. It can be
        called again to re-seed the generator. For details, see `RandomState`.

        Parameters
        ----------
        seed : int or 1-d array_like, optional
            Seed for `RandomState`.
            Must be convertible to 32 bit unsigned integers.

        See Also
        --------
        RandomState
        """
        self._random_state.seed(seed=seed)

    def to_numpy(self):
        return self._random_state

    @classmethod
    def from_numpy(cls, np_random_state):
        state = RandomState()
        state._random_state = np_random_state
        return state

    @classmethod
    def _handle_size(cls, size):
        if size is None:
            return size
        try:
            return tuple(int(s) for s in size)
        except TypeError:
            return (size,)


_random_state = RandomState()


def handle_array(arg):
    if not isinstance(arg, TENSOR_TYPE):
        if not isinstance(arg, Iterable):
            return arg

        arg = np.asarray(arg)
        return arg[(0,) * max(1, arg.ndim)]
    elif hasattr(arg, "op") and hasattr(arg.op, "data"):
        return arg.op.data[(0,) * max(1, arg.ndim)]

    return np.empty((0,), dtype=arg.dtype)


class TensorRandomOperatorMixin(TensorOperatorMixin):
    __slots__ = ()

    def _calc_shape(self, shapes):
        shapes = list(shapes)
        if getattr(self, "size", None) is not None:
            shapes.append(getattr(self, "size"))
        return broadcast_shape(*shapes)

    @classmethod
    def _handle_arg(cls, arg, chunk_size):
        if isinstance(arg, (list, np.ndarray)):
            arg = astensor(arg, chunk_size=chunk_size)

        return arg

    @contextmanager
    def _get_inputs_shape_by_given_fields(
        self, inputs, shape, raw_chunk_size=None, tensor=True
    ):
        fields = getattr(self, "_input_fields_", [])
        to_one_chunk_fields = set(getattr(self, "_into_one_chunk_fields_", list()))

        field_to_obj = dict()
        to_broadcast_shapes = []
        if fields:
            if getattr(self, fields[0], None) is None:
                # create from beginning
                for field, val in zip(fields, inputs):
                    if field not in to_one_chunk_fields:
                        if isinstance(val, list):
                            val = np.asarray(val)
                        if tensor:
                            val = self._handle_arg(val, raw_chunk_size)
                    if isinstance(val, TENSOR_TYPE):
                        field_to_obj[field] = val
                        if field not in to_one_chunk_fields:
                            to_broadcast_shapes.append(val.shape)
                    setattr(self, field, val)
            else:
                inputs_iter = iter(inputs)
                for field in fields:
                    if isinstance(getattr(self, field), TENSOR_TYPE):
                        field_to_obj[field] = next(inputs_iter)

        if tensor:
            if shape is None:
                shape = self._calc_shape(to_broadcast_shapes)

            for field, inp in field_to_obj.items():
                if field not in to_one_chunk_fields:
                    field_to_obj[field] = broadcast_to(inp, shape)

        yield [field_to_obj[f] for f in fields if f in field_to_obj], shape

        inputs_iter = iter(getattr(self, "_inputs"))
        for field in fields:
            if field in field_to_obj:
                setattr(self, field, next(inputs_iter))

    @classmethod
    def _get_shape(cls, kws, kw):
        if kw.get("shape") is not None:
            return kw.get("shape")
        elif kws is not None and len(kws) > 0:
            return kws[0].get("shape")

    def _new_tileables(self, inputs, kws=None, **kw):
        raw_chunk_size = kw.get("chunk_size", None)
        shape = self._get_shape(kws, kw)
        with self._get_inputs_shape_by_given_fields(
            inputs, shape, raw_chunk_size, True
        ) as (inputs, shape):
            kw["shape"] = shape
            return super()._new_tileables(inputs, kws=kws, **kw)


def _on_serialize_random_state(rs):
    return rs.get_state() if rs is not None else None


def _on_deserialize_random_state(tup):
    if tup is None:
        return None

    rs = np.random.RandomState()
    rs.set_state(tup)
    return rs


def RandomStateField(name, **kwargs):
    kwargs.update(
        dict(
            on_serialize=_on_serialize_random_state,
            on_deserialize=_on_deserialize_random_state,
        )
    )
    return TupleField(name, **kwargs)


class TensorSeedOperatorMixin:
    @property
    def seed(self):
        return getattr(self, "seed", None)

    @property
    def args(self):
        if hasattr(self, "_fields_"):
            return self._fields_
        else:
            return [
                field
                for field in self._FIELDS
                if field not in TensorRandomOperator._FIELDS
            ]

    @classmethod
    def _set_inputs(cls, op: "TensorRandomOperator", inputs: List[EntityData]):
        super()._set_inputs(op, inputs)
        fields = getattr(cls, "_input_fields_", [])
        for field, inp in zip(fields, inputs):
            setattr(op, field, inp)


class TensorRandomOperator(TensorSeedOperatorMixin, TensorOperator):
    seed = Int32Field("seed", default=None)

    def __init__(self, dtype=None, **kw):
        dtype = np.dtype(dtype) if dtype is not None else dtype
        if "state" in kw:
            kw["_state"] = kw.pop("state")
        super().__init__(dtype=dtype, **kw)


class TensorRandomMapReduceOperator(TensorSeedOperatorMixin, TensorMapReduceOperator):
    seed = Int32Field("seed", default=None)

    def __init__(self, dtype=None, **kw):
        dtype = np.dtype(dtype) if dtype is not None else dtype
        if "state" in kw:
            kw["_state"] = kw.pop("state")
        super().__init__(dtype=dtype, **kw)


class TensorDistribution(TensorRandomOperator):
    size = TupleField("size", FieldTypes.int64)


class TensorSimpleRandomData(TensorRandomOperator):
    size = TupleField(
        "size",
        FieldTypes.int64,
        default=None,
        on_serialize=on_serialize_shape,
        on_deserialize=on_deserialize_shape,
    )

    def __init__(self, size=None, **kw):
        if type(size) is int:
            size = (size,)
        super().__init__(size=size, **kw)
