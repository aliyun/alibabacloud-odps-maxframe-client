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

from .... import opcodes
from ....config import options
from ....core import EntityData
from ....core.operator import OperatorStage
from ....serialization.serializables import (
    AnyField,
    BoolField,
    DictField,
    Int64Field,
    KeyField,
)
from ....tensor.core import TensorOrder
from .core import PairwiseDistances


class PairwiseDistancesTopk(PairwiseDistances):
    _op_type_ = opcodes.PAIRWISE_DISTANCES_TOPK

    x = KeyField("x")
    y = KeyField("y")
    k = Int64Field("k", default=None)
    metric = AnyField("metric", default=None)
    metric_kwargs = DictField("metric_kwargs", default=None)
    return_index = BoolField("return_index", default=None)
    working_memory = AnyField("working_memory", default=None)

    @property
    def output_limit(self):
        return 2 if self.return_index else 1

    @classmethod
    def _set_inputs(cls, op: "PairwiseDistancesTopk", inputs: List[EntityData]):
        super()._set_inputs(op, inputs)
        if op.stage != OperatorStage.agg:
            op.x, op.y = inputs[:2]
        else:
            op.x = op.y = None

    def __call__(self, X, Y):
        from .pairwise import pairwise_distances

        # leverage pairwise_distances for checks
        d = pairwise_distances(X, Y, metric=self.metric, **self.metric_kwargs)

        if self.k > Y.shape[0]:
            self.k = Y.shape[0]

        X, Y = d.op.inputs

        shape_list = [X.shape[0]]
        shape_list.append(min(Y.shape[0], self.k))
        shape = tuple(shape_list)
        kws = [
            {
                "shape": shape,
                "order": TensorOrder.C_ORDER,
                "dtype": np.dtype(np.float64),
                "_type_": "distance",
            },
        ]
        if self.return_index:
            kws.append(
                {
                    "shape": shape,
                    "order": TensorOrder.C_ORDER,
                    "dtype": np.dtype(np.int64),
                    "_type_": "index",
                }
            )
            return self.new_tensors([X, Y], kws=kws)
        else:
            return self.new_tensors([X, Y], kws=kws)[0]


def pairwise_distances_topk(
    X,
    Y=None,
    k=None,
    metric="euclidean",
    return_index=True,
    axis=1,
    working_memory=None,
    **kwds
):
    if k is None:  # pragma: no cover
        raise ValueError("`k` has to be specified")

    if Y is None:
        Y = X
    if axis == 0:
        X, Y = Y, X
    if working_memory is None:
        working_memory = options.learn.working_memory
    op = PairwiseDistancesTopk(
        x=X,
        y=Y,
        k=k,
        metric=metric,
        metric_kwargs=kwds,
        return_index=return_index,
        working_memory=working_memory,
    )
    return op(X, Y)
