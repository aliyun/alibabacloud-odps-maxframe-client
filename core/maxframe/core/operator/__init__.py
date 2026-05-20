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

from maxframe.core.operator.base import (
    HasInput,
    LogicKeyGenerator,
    Operator,
    OperatorLogicKeyGeneratorMixin,
    OperatorStage,
    VirtualOperator,
)
from maxframe.core.operator.core import (
    TileableOperatorMixin,
    estimate_size,
    estimate_tileable_execution_size,
    execute,
)
from maxframe.core.operator.fetch import Fetch, FetchMixin, FetchShuffle
from maxframe.core.operator.objects import (
    ObjectFetch,
    ObjectOperator,
    ObjectOperatorMixin,
)
from maxframe.core.operator.shuffle import MapReduceOperator, ShuffleProxy
from maxframe.core.operator.utils import add_fetch_builder, build_fetch
