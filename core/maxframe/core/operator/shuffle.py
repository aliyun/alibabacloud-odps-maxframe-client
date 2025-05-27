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

from ... import opcodes
from ...serialization.serializables import Int32Field, StringField
from .base import Operator, OperatorStage, VirtualOperator
from .core import TileableOperatorMixin


class ShuffleProxy(VirtualOperator, TileableOperatorMixin):
    _op_type_ = opcodes.SHUFFLE_PROXY
    n_mappers = Int32Field("n_mappers", default=0)
    n_reducers = Int32Field("n_reducers", default=0)


class MapReduceOperator(Operator):
    """
    An operator for shuffle execution which partitions data by the value in each record’s partition key, and
    send the partitioned data from all mappers to all reducers.
    """

    # for reducer
    reducer_id = Int32Field("reducer_id")
    # Total reducer nums, which also be shuffle blocks for single mapper.
    n_reducers = Int32Field("n_reducers")
    # preserved field to keep serialization compatible
    _mr_preserved = Int32Field("mr_preserved")
    reducer_phase = StringField("reducer_phase", default=None)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.stage == OperatorStage.reduce:
            # for reducer, we assign worker at first
            self.scheduling_hint.reassign_worker = True
