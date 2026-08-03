# Copyright 1999-2026 Alibaba Group Holding Ltd.
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

from maxframe import opcodes
from maxframe.core import OutputType
from maxframe.core.operator import OperatorStage
from maxframe.liteframe.core import FrameMetadata
from maxframe.liteframe.operators.core import LiteFrameOperator, LiteFrameOperatorMixin
from maxframe.serialization.serializables import DictField, Int32Field, ReferenceField


class LiteFrameGroupByOp(LiteFrameOperatorMixin, LiteFrameOperator):
    """Internal operator for distributed groupby partitioning.

    Handles only data partitioning by group keys (hash-based shuffle).
    Does NOT perform aggregation — that is handled by LiteFrameAgg.
    """

    _op_type_ = opcodes.GROUPBY  # reuse opcode 2030
    _output_type_ = OutputType.liteframe

    stage = ReferenceField("stage", OperatorStage, default=None)
    groupby_params = DictField("groupby_params")
    index_shuffle_size = Int32Field("index_shuffle_size", default=None)
    n_reducers = Int32Field("n_reducers", default=None)

    def __call__(self, liteframe):
        hidden_columns = None
        if liteframe.frame_metadata and liteframe.frame_metadata.hidden_columns:
            hidden_columns = liteframe.frame_metadata.hidden_columns

        frame_metadata = None
        if hidden_columns:
            frame_metadata = FrameMetadata(hidden_columns=hidden_columns)

        return self.new_liteframe(
            [liteframe],
            shape=(np.nan, liteframe.shape[1]),
            physical_dtypes=liteframe._data._physical_dtypes,
            frame_metadata=frame_metadata,
        )
