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
import pyarrow as pa

from maxframe import opcodes
from maxframe.liteframe.core import FrameMetadata
from maxframe.liteframe.expressions import LiteFrameExpr
from maxframe.liteframe.operators.core import LiteFrameOperator, LiteFrameOperatorMixin
from maxframe.serialization.serializables import AnyField
from maxframe.utils import wrap_arrow_dtype


class LiteFrameFilter(LiteFrameOperatorMixin, LiteFrameOperator):
    _op_type_ = opcodes.LITEFRAME_FILTER

    predicate = AnyField("predicate")

    def __call__(self, liteframe):
        shape = (np.nan, liteframe.shape[1])
        # Materialize range columns: use _physical_dtypes which includes hidden columns
        new_dtypes = liteframe._data._physical_dtypes.copy()
        if liteframe.frame_metadata and liteframe.frame_metadata.range_columns:
            for rc_name in liteframe.frame_metadata.range_columns:
                new_dtypes[rc_name] = wrap_arrow_dtype(pa.int64())

        hidden_columns = None
        if liteframe.frame_metadata and liteframe.frame_metadata.hidden_columns:
            hidden_columns = liteframe.frame_metadata.hidden_columns

        new_metadata = FrameMetadata(
            range_columns={},
            hidden_columns=hidden_columns,
        )
        inputs = [liteframe]
        if not isinstance(self.predicate, LiteFrameExpr):
            # Boolean mask predicate: include as input for proper dependency tracking
            inputs.append(self.predicate)
        return self.new_liteframe(
            inputs,
            shape=shape,
            physical_dtypes=new_dtypes,
            frame_metadata=new_metadata,
        )
