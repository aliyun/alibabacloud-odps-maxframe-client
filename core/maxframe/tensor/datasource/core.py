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

import numpy as np

from ...serialization.serializables import FieldTypes, StringField, TupleField
from ..core import TensorOrder
from ..operators import TensorOperator, TensorOperatorMixin
from ..utils import normalize_shape


class TensorDataSource(TensorOperator, TensorOperatorMixin):
    """
    Tensor data source base class, provide universal tile logic,
    subclass can overwrite tile method.
    """

    __slots__ = ()


class TensorNoInput(TensorDataSource):
    """
    Tensor operator with no inputs.
    """

    def check_inputs(self, inputs):
        # no inputs
        if inputs and len(inputs) > 0:
            raise ValueError("Tensor data source has no inputs")

    def _new_tileables(self, inputs, kws=None, **kw):
        shape = kw.get("shape", None)
        self.extra_params[
            "shape"
        ] = shape  # set shape to make the operator key different
        return super()._new_tileables(inputs, kws=kws, **kw)

    def __call__(self, shape, chunk_size=None, order=None):
        shape = normalize_shape(shape)
        order = TensorOrder.C_ORDER if order is None else order
        return self.new_tensor(None, shape, raw_chunk_size=chunk_size, order=order)


class TensorHasInput(TensorDataSource):
    """
    Tensor operator with a single input.
    """

    @property
    def input(self):
        return self._input

    def check_inputs(self, inputs):
        # no inputs
        if len(inputs) != 1:
            raise ValueError("Tensor can only have 1 input")

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        self._input = self._inputs[0]

    def __call__(self, a, order=None):
        order = a.order if order is None else order
        return self.new_tensor([a], a.shape, order=order)


class TensorLike(TensorHasInput):
    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        if self.dtype is None:
            self.dtype = self.input.dtype
        if self.gpu is None:
            self.gpu = self.input.op.gpu

        # FIXME: remove when cupy supports other dtypes
        if self.gpu and self.dtype not in (np.float32, np.float64):
            raise NotImplementedError(
                "Sparse tensor on GPU only supports float32 and float64"
            )


class TensorFromHDF5Like(TensorNoInput):
    filename = StringField("filename", default=None)
    group = StringField("group", default=None)
    dataset = StringField("dataset", default=None)
    axis_offsets = TupleField("axis_offsets", FieldTypes.int64, default=None)

    @property
    def path(self):
        return self.get_path(self.group, self.dataset)

    @staticmethod
    def get_path(group, dataset):
        paths = []
        if group:
            paths.append(group)
        paths.append(dataset)
        return "/".join(paths)
