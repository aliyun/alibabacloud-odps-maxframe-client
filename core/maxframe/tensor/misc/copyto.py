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

from typing import List

import numpy as np

from ... import opcodes
from ...core import EntityData
from ...serialization.serializables import KeyField, StringField
from ..datasource import tensor as astensor
from ..operators import TensorOperator, TensorOperatorMixin
from .broadcast_to import broadcast_to


class TensorCopyTo(TensorOperator, TensorOperatorMixin):
    _op_type_ = opcodes.COPYTO

    src = KeyField("src")
    dst = KeyField("dest")
    casting = StringField("casting")
    where = KeyField("where")

    @classmethod
    def check_inputs(cls, inputs):
        if not 2 <= len(inputs) <= 3:
            raise ValueError("inputs' length must be 2 or 3")

    @classmethod
    def _set_inputs(cls, op: "TensorCopyTo", inputs: List[EntityData]):
        super()._set_inputs(op, inputs)
        op.src = inputs[0]
        op.dst = inputs[1]
        if len(op.inputs) > 2:
            op.where = inputs[2]

    @staticmethod
    def _extract_inputs(inputs):
        if len(inputs) == 2:
            (src, dst), where = inputs, None
        else:
            src, dst, where = inputs
            if where is True:
                where = None
            else:
                where = astensor(where)

        return src, dst, where

    def __call__(self, *inputs):
        from ..core import Tensor

        src, dst, where = self._extract_inputs(inputs)

        if not isinstance(dst, Tensor):
            raise TypeError("dst has to be a Tensor")

        self.dtype = dst.dtype
        self.gpu = dst.op.gpu
        self.sparse = dst.issparse()

        if not np.can_cast(src.dtype, dst.dtype, casting=self.casting):
            raise TypeError(
                f"Cannot cast array from {src.dtype!r} to {dst.dtype!r} "
                f"according to the rule {self.casting!s}"
            )

        try:
            broadcast_to(src, dst.shape)
        except ValueError:
            raise ValueError(
                "could not broadcast input array "
                f"from shape {src.shape!r} into shape {dst.shape!r}"
            )
        if where is not None:
            try:
                broadcast_to(where, dst.shape)
            except ValueError:
                raise ValueError(
                    "could not broadcast where mask "
                    f"from shape {src.shape!r} into shape {dst.shape!r}"
                )

        inps = [src, dst]
        if where is not None:
            inps.append(where)
        ret = self.new_tensor(inps, dst.shape, order=dst.order)
        dst.data = ret.data


def copyto(dst, src, casting="same_kind", where=True):
    """
    Copies values from one array to another, broadcasting as necessary.

    Raises a TypeError if the `casting` rule is violated, and if
    `where` is provided, it selects which elements to copy.

    Parameters
    ----------
    dst : Tensor
        The tensor into which values are copied.
    src : array_like
        The tensor from which values are copied.
    casting : {'no', 'equiv', 'safe', 'same_kind', 'unsafe'}, optional
        Controls what kind of data casting may occur when copying.

          * 'no' means the data types should not be cast at all.
          * 'equiv' means only byte-order changes are allowed.
          * 'safe' means only casts which can preserve values are allowed.
          * 'same_kind' means only safe casts or casts within a kind,
            like float64 to float32, are allowed.
          * 'unsafe' means any data conversions may be done.
    where : array_like of bool, optional
        A boolean tensor which is broadcasted to match the dimensions
        of `dst`, and selects elements to copy from `src` to `dst`
        wherever it contains the value True.
    """
    op = TensorCopyTo(casting=casting)
    return op(src, dst, where)
