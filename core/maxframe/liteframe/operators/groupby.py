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
import pandas as pd
import pyarrow as pa

from maxframe import opcodes
from maxframe.core import OutputType
from maxframe.core.operator import OperatorStage
from maxframe.liteframe.core import FrameMetadata
from maxframe.liteframe.operators.core import LiteFrameOperator, LiteFrameOperatorMixin
from maxframe.serialization.serializables import DictField, Int32Field, ReferenceField
from maxframe.utils import wrap_arrow_dtype

# Fixed module-level constants for the routing columns attached by the
# groupby map executor when placeback is requested.  Both the tiler and
# the map executor reference these directly -- no names are passed
# through extra_params.  If a future conflict arises (a user column
# named ``__mf_gb_ci__``) the constants can be made conflict-free via
# ``_make_unique_col_name`` later.
_GB_CI_COL = "__mf_gb_ci__"  # groupby source-chunk id
_GB_RI_COL = "__mf_gb_ri__"  # groupby per-chunk row id


def _routing_dtypes(base_dtypes: pd.Series) -> pd.Series:
    """Append the two Int64 routing columns to a dtype series.

    When ``base_dtypes`` is ``None`` (unknown output schema, e.g.
    ``skip_infer=True``), only the routing columns are returned — the
    real output columns are resolved at execution time.
    """
    int64_dt = wrap_arrow_dtype(pa.int64())
    if base_dtypes is None:
        return pd.Series({_GB_CI_COL: int64_dt, _GB_RI_COL: int64_dt})
    return pd.concat(
        [
            base_dtypes,
            pd.Series({_GB_CI_COL: int64_dt, _GB_RI_COL: int64_dt}),
        ]
    )


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

    def __call__(self, liteframe):
        existing_hidden = list(
            getattr(liteframe.frame_metadata, "hidden_columns", None) or []
        )
        # Preserve input order and append routing columns deterministically
        # so FrameMetadata.__maxframe_tokenize__ produces stable keys across
        # processes (set ordering is hash-seed-dependent).
        hidden = list(existing_hidden)

        if self.extra_params.get("placeback", False):
            # Plan-time collision check: a user column sharing a routing
            # name would be silently overwritten and then dropped.  Raise
            # rather than lose data.
            user_cols = set(liteframe._physical_dtypes.index)
            collisions = user_cols & {_GB_CI_COL, _GB_RI_COL}
            if collisions:
                raise ValueError(
                    f"Grouped place-back reserves internal column names "
                    f"{_GB_CI_COL!r} and {_GB_RI_COL!r}, but the input "
                    f"frame already has column(s): {sorted(collisions)}. "
                    f"Please rename the conflicting column(s)."
                )
            phys = _routing_dtypes(liteframe._physical_dtypes)
            for rc in (_GB_CI_COL, _GB_RI_COL):
                if rc not in hidden:
                    hidden.append(rc)
            n_cols = liteframe.shape[1] + 2
        else:
            phys = liteframe._physical_dtypes
            n_cols = liteframe.shape[1]

        frame_metadata = FrameMetadata(hidden_columns=hidden) if hidden else None

        return self.new_liteframe(
            [liteframe],
            shape=(np.nan, n_cols),
            physical_dtypes=phys,
            frame_metadata=frame_metadata,
        )
