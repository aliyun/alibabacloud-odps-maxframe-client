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

import inspect

from maxframe import opcodes
from maxframe.core import OutputType
from maxframe.core.operator import ShuffleProxy
from maxframe.core.operator.base import CallPoint
from maxframe.liteframe.operators.core import LiteFrameOperatorMixin


class LiteFrameShuffleProxy(ShuffleProxy, LiteFrameOperatorMixin):
    """
    Virtual operator for coordinating shuffle operations in LiteFrame merge.

    Acts as a graph marker for shuffle stages, enabling proper chunk
    coordination between map and reduce phases during distributed merge.

    Unlike regular operators, shuffle proxy does not execute computation
    but coordinates data flow between mappers and reducers.

    Notes
    -----
    - Inherits from ShuffleProxy for proper shuffle infrastructure integration
    - Uses existing SHUFFLE_PROXY opcode for engine compatibility
    - Integrated with LiteFrameMergeAlign for shuffle map/reduce stages
    """

    _op_type_ = opcodes.SHUFFLE_PROXY

    def __init__(self, output_types=None, **kwargs):
        kwargs["call_points"] = kwargs.pop("call_points", None) or [
            CallPoint.from_frame(inspect.currentframe().f_back)
        ]
        if output_types is None:
            output_types = [OutputType.liteframe]
        super().__init__(_output_types=output_types, **kwargs)
