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

from typing import List, Optional

import numpy as np

from ... import opcodes
from ... import tensor as mt
from ...core import OutputType
from ...core.operator import Operator, TileableOperatorMixin
from ...serialization.serializables import Int16Field, ReferenceField
from ...typing_ import EntityType


class LearnCountNonzero(Operator, TileableOperatorMixin):
    _op_module_ = "learn"
    _op_type_ = opcodes.COUNT_NONZERO

    axis = Int16Field("axis")
    sample_weight = ReferenceField("sample_weight")

    @classmethod
    def _set_inputs(cls, op: "LearnCountNonzero", inputs: List[EntityType]):
        super()._set_inputs(op, inputs)
        if op.sample_weight is not None:
            op.sample_weight = inputs[-1]

    def __call__(self, x, sample_weight=None):
        self.sample_weight = sample_weight
        self._output_types = [
            OutputType.scalar if self.axis is None else OutputType.tensor
        ]
        dtype = np.dtype(int)
        inputs = [x]
        if sample_weight is not None:
            dtype = sample_weight.dtype
            inputs = [x, sample_weight]

        if self.axis is None:
            shape = ()
        else:
            shape = (x.shape[1 - self.axis],)

        return self.new_tileable(inputs, shape=shape, dtype=dtype)


def count_nonzero(X, axis: Optional[int] = None, sample_weight=None):
    """A variant of X.getnnz() with extension to weighting on axis 0

    Useful in efficiently calculating multilabel metrics.

    Parameters
    ----------
    X : CSR sparse matrix of shape (n_samples, n_labels)
        Input data.

    axis : None, 0 or 1
        The axis on which the data is aggregated.

    sample_weight : array-like of shape (n_samples,), default=None
        Weight for each row of X.
    """
    if axis == -1:
        axis = 1
    elif axis == -2:
        axis = 0
    if axis is not None and axis not in (0, 1):
        raise ValueError(f"Unsupported axis: {axis}")

    X = mt.asarray(X)
    if sample_weight is not None:
        sample_weight = mt.asarray(sample_weight)

    op = LearnCountNonzero(axis=axis)
    return op(X, sample_weight=sample_weight)
