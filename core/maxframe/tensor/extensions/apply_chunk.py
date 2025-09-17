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

from ... import opcodes
from ...core import ENTITY_TYPE
from ...serialization.serializables import (
    BoolField,
    DictField,
    FunctionField,
    TupleField,
)
from ...udf import BuiltinFunction
from ...utils import find_objects, quiet_stdio, replace_objects
from ..core import TensorOrder
from ..operators import TensorOperator, TensorOperatorMixin


class TensorApplyChunk(TensorOperator, TensorOperatorMixin):
    _op_type_ = opcodes.APPLY_CHUNK

    func = FunctionField("func")
    elementwise = BoolField("elementwise")
    args = TupleField("args", default=None)
    kwargs = DictField("kwargs", default=None)
    with_chunk_index = BoolField("with_chunk_index", default=False)

    @classmethod
    def _set_inputs(cls, op: "TensorApplyChunk", inputs):
        super()._set_inputs(op, inputs)
        old_inputs = find_objects(op.args, ENTITY_TYPE) + find_objects(
            op.kwargs, ENTITY_TYPE
        )
        mapping = {o: n for o, n in zip(old_inputs, op._inputs[1:])}
        op.args = replace_objects(op.args, mapping)
        op.kwargs = replace_objects(op.kwargs, mapping)

    def has_custom_code(self) -> bool:
        return not isinstance(self.func, BuiltinFunction)

    def __call__(self, t, dtype=None, shape=None, order=None):
        if dtype is None:
            try:
                kwargs = self.kwargs or dict()
                if self.with_chunk_index:
                    kwargs["chunk_index"] = (0,) * t.ndim
                with np.errstate(all="ignore"), quiet_stdio():
                    mock_result = self.func(
                        np.random.rand(2, 2).astype(t.dtype),
                        *(self.args or ()),
                        **kwargs
                    )
            except:
                raise TypeError("Cannot estimate output type of apply_chunk call")
            dtype = mock_result.dtype
            order = (
                TensorOrder.C_ORDER
                if mock_result.flags["C_CONTIGUOUS"]
                else TensorOrder.F_ORDER
            )

        if shape is not None:
            new_shape = shape
        else:
            new_shape = t.shape if self.elementwise else (np.nan,) * t.ndim
        inputs = (
            [t]
            + find_objects(self.args, ENTITY_TYPE)
            + find_objects(self.kwargs, ENTITY_TYPE)
        )
        return self.new_tensor(inputs, dtype=dtype, shape=new_shape, order=order)


def apply_chunk(t, func, args=(), **kwargs):
    """
    Apply function to each chunk.

    Parameters
    ----------
    func : function
        Function to apply to each chunk.
    args : tuple
        Positional arguments to pass to func in addition to the array.
    **kwargs
        Additional keyword arguments to pass as keywords arguments to func.

    Returns
    -------
    Tensor
        Result of applying ``func`` to each chunk of the Tensor.

    Examples
    --------
    >>> import maxframe.tensor as mt
    >>> a = mt.array([[4, 9]] * 3)
    >>> a.execute()
    array([[4, 9],
           [4, 9],
           [4, 9]])

    Output dtype will be auto inferred.

    >>> a.mf.apply_chunk(lambda c: c * 0.5).execute()
    array([[2. , 4.5],
           [2. , 4.5],
           [2. , 4.5]])

    You can specify ``dtype`` by yourself if auto infer failed.
    """
    elementwise = kwargs.pop("elementwise", None)
    dtype = np.dtype(kwargs.pop("dtype")) if "dtype" in kwargs else None
    shape = kwargs.pop("shape", None)
    order = kwargs.pop("order", None)
    sparse = kwargs.pop("sparse", t.issparse())
    with_chunk_index = kwargs.pop("with_chunk_index", False)

    op = TensorApplyChunk(
        func=func,
        args=args,
        kwargs=kwargs,
        elementwise=elementwise,
        with_chunk_index=with_chunk_index,
        sparse=sparse,
    )
    return op(t, dtype=dtype, shape=shape, order=order)
