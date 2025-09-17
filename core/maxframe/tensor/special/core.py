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

import numpy as np

try:
    import scipy.special as spspecial
except ImportError:
    spspecial = None

from ... import opcodes
from ...core import ExecutableTuple
from ...serialization.serializables import StringField
from ..arithmetic.core import TensorBinOp, TensorMultiOp, TensorUnaryOp
from ..datasource import tensor as astensor

_func_name_to_special_cls = {}


def _register_special_op(cls):
    if cls._func_name is not None:
        _func_name_to_special_cls[cls._func_name] = cls
    return cls


class TensorSpecialOperatorMixin:
    _op_type_ = opcodes.SPECIAL
    _func_name = None

    def __new__(cls, *args, **kwargs):
        func_name = kwargs.get("func_name", cls._func_name)
        kwargs["func_name"] = func_name
        if func_name is not None:
            return object.__new__(_func_name_to_special_cls[func_name])
        return super().__new__(cls, *args, **kwargs)


class TensorSpecialUnaryOp(TensorSpecialOperatorMixin, TensorUnaryOp):
    func_name = StringField("func_name")


class TensorSpecialBinOp(TensorSpecialOperatorMixin, TensorBinOp):
    func_name = StringField("func_name")


class TensorSpecialMultiOp(TensorSpecialOperatorMixin, TensorMultiOp):
    func_name = StringField("func_name")


class TensorSpecialUnaryTupleOp(TensorSpecialUnaryOp):
    @property
    def output_limit(self):
        return self._n_outputs

    def __call__(self, x, out=None):
        x = astensor(x)

        if out is not None:
            if not isinstance(out, ExecutableTuple):
                raise TypeError(
                    f"out should be ExecutableTuple object, got {type(out)} instead"
                )
            if len(out) != self._n_outputs:
                raise TypeError(
                    "out should be an ExecutableTuple object with "
                    f"{self._n_outputs} elements, got {len(out)} instead"
                )

        func = getattr(spspecial, self._func_name)
        res = func(np.ones(x.shape, dtype=x.dtype))
        res_tensors = self.new_tensors(
            [x],
            kws=[
                {
                    "side": f"{self._func_name}[{i}]",
                    "dtype": output.dtype,
                    "shape": output.shape,
                }
                for i, output in enumerate(res)
            ],
        )

        if out is None:
            return ExecutableTuple(res_tensors)

        for res_tensor, out_tensor in zip(res_tensors, out):
            out_tensor.data = res_tensor.data
        return out
