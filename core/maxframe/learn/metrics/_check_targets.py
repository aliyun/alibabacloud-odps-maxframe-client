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
from ... import tensor as mt
from ...core import ENTITY_TYPE, ExecutableTuple, OutputType
from ...core.operator import Operator
from ...serialization.serializables import AnyField
from ...tensor.core import TensorOrder
from ...typing_ import EntityType
from ..core import LearnOperatorMixin


class CheckTargets(Operator, LearnOperatorMixin):
    _op_type_ = opcodes.CHECK_TARGETS

    y_true = AnyField("y_true")
    y_pred = AnyField("y_pred")

    @property
    def output_limit(self):
        return 3

    @classmethod
    def _set_inputs(cls, op: "CheckTargets", inputs: List[EntityType]):
        super()._set_inputs(op, inputs)
        inputs_iter = iter(op.inputs)
        if isinstance(op.y_true, ENTITY_TYPE):
            op.y_true = next(inputs_iter)
        if isinstance(op.y_pred, ENTITY_TYPE):
            op.y_pred = next(inputs_iter)

    def __call__(self, y_true, y_pred):
        # scalar(y_type), y_true, y_pred
        self.output_types = [OutputType.tensor] * 3

        inputs = []
        if isinstance(y_true, ENTITY_TYPE):
            inputs.append(y_true)
        if isinstance(y_pred, ENTITY_TYPE):
            inputs.append(y_pred)

        kws = list()
        kws.append(
            {"shape": (), "dtype": np.dtype(object), "order": TensorOrder.C_ORDER}
        )
        kws.extend([y.params for y in (mt.tensor(y_true), mt.tensor(y_pred))])
        kws[1]["shape"] = kws[2]["shape"] = (np.nan,)
        return ExecutableTuple(self.new_tileables(inputs, kws=kws))


def _check_targets(y_true, y_pred):
    """Check that y_true and y_pred belong to the same classification task

    This converts multiclass or binary types to a common shape, and raises a
    ValueError for a mix of multilabel and multiclass targets, a mix of
    multilabel formats, for the presence of continuous-valued or multioutput
    targets, or for targets of different lengths.

    Column vectors are squeezed to 1d, while multilabel formats are returned
    as CSR sparse label indicators.

    Parameters
    ----------
    y_true : array-like

    y_pred : array-like

    Returns
    -------
    type_true : one of {'multilabel-indicator', 'multiclass', 'binary'}
        The type of the true target data, as output by
        ``utils.multiclass.type_of_target``

    y_true : Tensor

    y_pred : Tensor
    """
    op = CheckTargets(y_true=y_true, y_pred=y_pred)
    return op(y_true, y_pred)
