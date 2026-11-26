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
from maxframe.liteframe.operators.core import LiteFrameOperator, LiteFrameOperatorMixin
from maxframe.serialization.serializables import (
    BoolField,
    FieldTypes,
    Float64Field,
    Int64Field,
    ListField,
    StringField,
)
from maxframe.tensor.random import RandomStateField
from maxframe.utils import wrap_arrow_dtype


class LiteFrameSample(LiteFrameOperatorMixin, LiteFrameOperator):
    """Sample rows from a LiteFrame with optional global coordination.

    Shares opcode ``RAND_SAMPLE`` with the DataFrame ``DataFrameSample`` /
    ``GroupBySample`` operators; distinguished by ``_op_module_ = "liteframe"``.
    """

    _op_type_ = opcodes.RAND_SAMPLE
    _op_module_ = "liteframe"

    # --- sample parameters (set at API time) ---
    n = Int64Field("n", default=None)
    frac = Float64Field("frac", default=None)
    replace = BoolField("replace", default=False)
    exact = BoolField("exact", default=False)
    random_state = RandomStateField("random_state", default=None)
    # Column name for weights. If weights is a string, the column already
    # exists in the data. If weights is a LiteFrame, the API layer merges
    # it via concat before constructing the operator. The operator does not
    # drop the weight column -- if it was merged, the API layer adds a
    # projection to drop it after sampling.
    weights_col = StringField("weights_col", default=None)

    # --- per-chunk fields (set by tiler) ---
    # Per-chunk seed derived from global random_state via gen_random_seeds
    seed = Int64Field("seed", default=None)
    # Per-chunk sample count (multinomial distribution)
    chunk_n = Int64Field("chunk_n", default=None)
    # Path D: per-chunk selected row IDs (positions within the chunk).
    # Sorted ascending so output respects original data order.
    selected_row_ids = ListField("selected_row_ids", FieldTypes.int64, default=None)
    # Internal dispatch mode for the DPE executor: "build_id_frame" generates
    # the id frame, "exact_select" selects rows from the top-n result, or
    # None for normal sampling. FIXME: a future refactor should use a
    # dedicated operator instead of overloading this one.
    action = StringField("action", default=None)

    def __call__(self, liteframe):
        # Single input only. When weights is a separate LiteFrame, the API
        # layer calls concat to merge the weight column into the data before
        # constructing the operator. weights_col identifies the column.
        n_cols = liteframe.shape[1]
        # Materialize range columns: sampling changes the row set, so it
        # belongs to the "materialize all range columns" family (like
        # LiteFrameFilter).  Append them as physical Int64 columns so the
        # output schema and shape agree.
        new_dtypes = liteframe._data._physical_dtypes.copy()
        if liteframe.frame_metadata and liteframe.frame_metadata.range_columns:
            for rc_name in liteframe.frame_metadata.range_columns:
                new_dtypes[rc_name] = wrap_arrow_dtype(pa.int64())

        # Preserve hidden columns
        hidden_columns = None
        if liteframe.frame_metadata and liteframe.frame_metadata.hidden_columns:
            hidden_columns = liteframe.frame_metadata.hidden_columns

        new_metadata = FrameMetadata(
            range_columns={},
            hidden_columns=hidden_columns,
        )
        return self.new_liteframe(
            [liteframe],
            shape=(np.nan, n_cols),
            physical_dtypes=new_dtypes,
            frame_metadata=new_metadata,
        )
