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

import builtins
import operator
from collections.abc import Iterable
from functools import reduce

from ...serialization.serializables import AnyField, BoolField, Int32Field, KeyField
from ..array_utils import get_array_module
from ..core import Tensor, TensorOrder
from ..datasource import tensor as astensor
from ..operators import TensorHasInput, TensorOperatorMixin
from ..utils import check_out_param, validate_axis


def numel(x, **kwargs):
    xp = get_array_module(x)
    return xp.sum(xp.ones_like(x), **kwargs)


def nannumel(x, **kwargs):
    x_size = reduce(operator.mul, x.shape)
    xp = get_array_module(x)
    return x_size - xp.sum(xp.isnan(x), **kwargs)


class TensorReductionMixin(TensorOperatorMixin):
    __slots__ = ()

    @classmethod
    def _is_cum(cls):
        return False

    @classmethod
    def _calc_order(cls, a, out):
        return out.order if out is not None else a.order

    @classmethod
    def _is_sparse(cls, input_sparse, shape):
        return False

    def _call(self, a, out):
        a = astensor(a)
        if out is not None and not isinstance(out, Tensor):
            raise TypeError(f"out should be Tensor object, got {type(out)} instead")

        axis = getattr(self, "axis", None)
        keepdims = getattr(self, "keepdims", None)
        order = self._calc_order(a, out)

        if self._is_cum():
            if axis is None:
                a, axis = a.ravel(), 0
                setattr(self, "axis", axis)
            shape = a.shape
        else:
            axis = list(range(len(a.shape))) if axis is None else axis
            if not isinstance(axis, Iterable):
                axis = (validate_axis(a.ndim, axis),)
            axis = set(axis)

            shape = tuple(
                s if i not in axis else 1
                for i, s in enumerate(a.shape)
                if keepdims or i not in axis
            )

        self.sparse = self._is_sparse(a.issparse(), shape)
        t = self.new_tensor([a], shape, order=order)

        if out is None:
            return t

        check_out_param(out, t, "same_kind")
        out_shape, out_dtype = out.shape, out.dtype
        # if `out` is specified, use out's dtype and shape
        if out_shape != t.shape:
            if out.ndim > t.ndim:
                raise ValueError("output has too many dimensions")
            raise ValueError(f"output shape should be {t.shape}, got {out_shape}")

        setattr(self, "dtype", out_dtype)

        out.data = t.data
        return out

    def _new_tileables(self, inputs, kws=None, **kw):
        tensors = super()._new_tileables(inputs, kws=kws, **kw)
        setattr(self, "_input", getattr(self, "_inputs")[0])
        return tensors

    def __call__(self, a, out=None):
        return self._call(a, out=out)

    @classmethod
    def get_axis(cls, axis):
        return tuple(axis) if axis is not None else axis

    @classmethod
    def get_arg_axis(cls, axis, ndim):
        return None if len(axis) == ndim or ndim == 1 else axis[0]


class TensorArgReductionMixin(TensorReductionMixin):
    __slots__ = ()

    @staticmethod
    def _get_arg_axis(axis, ndim):
        if axis is None:
            axis = tuple(range(ndim))
            ravel = True
        elif isinstance(axis, int):
            axis = validate_axis(ndim, axis)
            axis = (axis,)
            ravel = ndim == 1
        else:
            raise TypeError(f"axis must be either `None` or int, got '{axis}'")
        return axis, ravel

    @staticmethod
    def _get_offset(tensor, axis, chunk, ravel):
        nsplits = tensor.nsplits
        offset = tuple(
            builtins.sum(split[:idx]) for split, idx in zip(nsplits, chunk.index)
        )
        if not ravel:
            offset = offset[axis[0]]
        return offset

    @classmethod
    def _calc_order(cls, a, out):
        return out.order if out is not None else TensorOrder.C_ORDER


class TensorCumReductionMixin(TensorReductionMixin):
    __slots__ = ()

    @classmethod
    def _is_cum(cls):
        return True


class TensorReduction(TensorHasInput):
    _input = KeyField("input")
    out = KeyField("out", default=None)
    axis = AnyField(
        "axis", default=None
    )  # can be None or int or tuple of ints, just infer the data
    keepdims = BoolField("keepdims", default=None)


class TensorCumReduction(TensorHasInput):
    _input = KeyField("input")
    axis = Int32Field("axis", default=None)
