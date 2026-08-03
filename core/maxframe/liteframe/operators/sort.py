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

import pandas as pd
import pyarrow as pa

from maxframe import opcodes
from maxframe.liteframe.core import FrameMetadata
from maxframe.liteframe.operators.core import LiteFrameOperator, LiteFrameOperatorMixin
from maxframe.serialization.serializables import (
    AnyField,
    FieldTypes,
    Int64Field,
    ListField,
    StringField,
)
from maxframe.utils import wrap_arrow_dtype

_VALID_KEEP_KINDS = {"head", "first", "last", "all"}


class LiteFrameSort(LiteFrameOperatorMixin, LiteFrameOperator):
    _op_type_ = opcodes.SORT_VALUES

    by = ListField("by", default=None)
    ascending = AnyField("ascending", default=True)
    kind = StringField("kind", default="quicksort")
    na_position = StringField("na_position", default="last")
    parallel_kind = StringField("parallel_kind", default="PSRS")
    psrs_kinds = ListField("psrs_kinds", FieldTypes.string, default=None)
    nrows = Int64Field("nrows", default=None)
    keep_kind = StringField("keep_kind", default="head")

    def __call__(self, liteframe):
        if self.keep_kind not in _VALID_KEEP_KINDS:
            raise ValueError(
                f"Invalid keep_kind '{self.keep_kind}', "
                f"must be one of {sorted(_VALID_KEEP_KINDS)}"
            )
        in_dtypes = liteframe._data._physical_dtypes
        in_metadata = liteframe.frame_metadata

        # If input has virtual range_columns, materialize them into dtypes.
        # Range columns go first to match the tiler's materialization order
        # (range columns first, then physical columns).
        if in_metadata is not None and in_metadata.range_columns:
            rc_dtypes = {
                rc_name: wrap_arrow_dtype(pa.int64())
                for rc_name in in_metadata.range_columns
            }
            new_dtypes = pd.concat([pd.Series(rc_dtypes), in_dtypes])
        else:
            new_dtypes = in_dtypes.copy()

        # Preserve hidden columns
        hidden_columns = None
        if in_metadata is not None and in_metadata.hidden_columns:
            hidden_columns = in_metadata.hidden_columns

        # After sort, no virtual range_columns (row order changed)
        new_metadata = FrameMetadata(
            range_columns={},
            hidden_columns=hidden_columns,
        )
        n_rows = liteframe.shape[0]
        # Apply nrows limit if set by head pushdown optimization
        if (
            self.nrows is not None
            and n_rows is not None
            and not (isinstance(n_rows, float) and n_rows != n_rows)
        ):
            n_rows = min(int(n_rows), self.nrows)
        shape = (n_rows, len(new_dtypes))

        return self.new_liteframe(
            [liteframe],
            shape=shape,
            physical_dtypes=new_dtypes,
            frame_metadata=new_metadata,
        )
