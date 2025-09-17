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

import pandas as pd
from pandas.api.types import is_list_like

from ...core import ENTITY_TYPE, EntityData
from ...core.operator import MapReduceOperator
from ...serialization.serializables import AnyField, KeyField, StringField
from ..operators import DataFrameOperatorMixin


class BaseDuplicateOp(MapReduceOperator, DataFrameOperatorMixin):
    _legacy_name = "DuplicateOperand"  # since 2.2.0

    input = KeyField("input")
    subset = AnyField("subset", default=None)
    keep = AnyField("keep", default="first")
    method = StringField("method", default=None)

    @classmethod
    def _set_inputs(cls, op: "BaseDuplicateOp", inputs: List[EntityData]):
        super()._set_inputs(op, inputs)
        op.input = op._inputs[0]


# keep for import compatibility
DuplicateOperand = BaseDuplicateOp


def validate_subset(df, subset):
    if subset is None:
        return subset
    if not is_list_like(subset) or isinstance(subset, ENTITY_TYPE):
        subset = [subset]
    else:
        subset = list(subset)

    for s in subset:
        if s not in df.dtypes:
            raise KeyError(pd.Index([s]))

    return subset
