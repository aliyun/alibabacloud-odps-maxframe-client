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

from ...serialization.serializables import (
    FieldTypes,
    Int32Field,
    Int64Field,
    StringField,
    TupleField,
)
from ..operators import TensorHasInput, TensorOperatorMixin
from ..utils import validate_axis


class TensorFFTBaseMixin(TensorOperatorMixin):
    __slots__ = ()

    @classmethod
    def _get_shape(cls, op, shape):
        raise NotImplementedError

    def __call__(self, a, order=None):
        shape = self._get_shape(self, a.shape)
        order = a.order if order is None else order
        return self.new_tensor([a], shape, order=order)


class TensorFFTMixin(TensorFFTBaseMixin):
    __slots__ = ()


class TensorComplexFFTMixin(TensorFFTMixin):
    @classmethod
    def _get_shape(cls, op, shape):
        new_shape = list(shape)
        if op.n is not None:
            new_shape[op.axis] = op.n
        return tuple(new_shape)


def validate_fft(tensor, axis=-1, norm=None):
    validate_axis(tensor.ndim, axis)
    if norm is not None and norm not in ("ortho",):
        raise ValueError(f'Invalid norm value {norm}, should be None or "ortho"')


class TensorFFTNMixin(TensorFFTBaseMixin):
    @staticmethod
    def _merge_shape(op, shape):
        new_shape = list(shape)
        if op.shape is not None:
            for ss, axis in zip(op.shape, op.axes):
                new_shape[axis] = ss
        return new_shape


class TensorComplexFFTNMixin(TensorFFTNMixin):
    @classmethod
    def _get_shape(cls, op, shape):
        return tuple(cls._merge_shape(op, shape))


class TensorRealFFTNMixin(TensorFFTNMixin):
    @classmethod
    def _get_shape(cls, op, shape):
        new_shape = cls._merge_shape(op, shape)
        new_shape[op.axes[-1]] = new_shape[op.axes[-1]] // 2 + 1
        return tuple(new_shape)


class TensorRealIFFTNMixin(TensorFFTNMixin):
    @classmethod
    def _get_shape(cls, op, shape):
        new_shape = list(shape)
        new_shape[op.axes[-1]] = 2 * (new_shape[op.axes[-1]] - 1)
        return tuple(cls._merge_shape(op, new_shape))


def validate_fftn(tensor, s=None, axes=None, norm=None):
    if axes is None:
        if s is None:
            axes = tuple(range(tensor.ndim))
        else:
            axes = tuple(range(len(s)))
    else:
        for axis in axes:
            validate_axis(tensor.ndim, axis)
        if len(set(axes)) < len(axes):
            raise ValueError("Duplicate axes not allowed")

    if norm is not None and norm not in ("ortho",):
        raise ValueError(f'Invalid norm value {norm}, should be None or "ortho"')

    return axes


class TensorFFTShiftMixin(TensorOperatorMixin):
    __slots__ = ()

    @classmethod
    def _is_inverse(cls):
        return False

    @classmethod
    def _process_axes(cls, x, axes):
        if axes is None:
            axes = tuple(range(x.ndim))
        elif isinstance(axes, Iterable):
            axes = tuple(axes)
        else:
            axes = (axes,)

        return axes


class TensorDiscreteFourierTransform(TensorHasInput):
    __slots__ = ()


class TensorBaseFFT(TensorDiscreteFourierTransform):
    norm = StringField("norm", default=None)


class TensorBaseSingleDimensionFFT(TensorBaseFFT):
    n = Int64Field("n", default=None)
    axis = Int32Field("axis", default=None)


class TensorBaseMultipleDimensionFFT(TensorBaseFFT):
    shape = TupleField("shape", FieldTypes.int64, default=None)
    axes = TupleField("axes", FieldTypes.int32, default=None)


class TensorStandardFFT(TensorBaseSingleDimensionFFT):
    pass


class TensorStandardFFTN(TensorBaseMultipleDimensionFFT):
    pass


class TensorFFTShiftBase(TensorHasInput):
    axes = TupleField("axes", FieldTypes.int32, default=None)


class TensorRealFFT(TensorBaseSingleDimensionFFT):
    pass


class TensorRealFFTN(TensorBaseMultipleDimensionFFT):
    pass


class TensorHermitianFFT(TensorBaseSingleDimensionFFT):
    pass
