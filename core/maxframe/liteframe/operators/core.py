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

from maxframe.core import OutputType
from maxframe.core.operator import Operator, TileableOperatorMixin


class LiteFrameOperatorMixin(TileableOperatorMixin):
    __slots__ = ()
    _op_module_ = "liteframe"

    def new_liteframes(
        self, inputs, shape=None, physical_dtypes=None, frame_metadata=None, **kw
    ):
        setattr(self, "_output_types", [OutputType.liteframe])
        return self.new_tileables(
            inputs,
            shape=shape,
            physical_dtypes=physical_dtypes,
            frame_metadata=frame_metadata,
            **kw,
        )

    def new_liteframe(self, inputs, **kw):
        return self.new_liteframes(inputs, **kw)[0]

    def new_scalars(self, inputs, dtype=None, **kw):
        setattr(self, "_output_types", [OutputType.scalar])
        return self.new_tileables(inputs, shape=(), dtype=dtype, **kw)

    def new_scalar(self, inputs, dtype=None, **kw):
        return self.new_scalars(inputs, dtype=dtype, **kw)[0]


LiteFrameOperator = Operator
