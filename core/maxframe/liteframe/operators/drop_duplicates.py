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
from maxframe.core.operator import MapReduceOperator
from maxframe.liteframe.core import FrameMetadata
from maxframe.liteframe.operators.core import LiteFrameOperatorMixin
from maxframe.serialization.serializables import (
    AnyField,
    BoolField,
    FieldTypes,
    Int32Field,
    ListField,
    StringField,
)
from maxframe.utils import wrap_arrow_dtype

MASK_COL_NAME = "0"

_CI_BASE = "__mf_dd_ci__"
_RI_BASE = "__mf_dd_ri__"
_MASK_BASE = "__mf_dd_mask__"

_VALID_KEEP = ("first", "last", False)
_VALID_METHOD = ("auto", "tree", "subset_tree", "shuffle")


def _make_unique_col_name(existing, base):
    """Return a column name derived from *base* not colliding with *existing*.

    Mirrors the DPE helper in ``executors/liteframe/window/transform.py``:
    deterministic (no randomness), so two equivalent operations allocate the
    same transient names → identical keys/tokens. A fixed base would collide
    with a legitimate user column of the same name and silently clobber it.
    """
    cols = set(existing)
    if base not in cols:
        return base
    i = 0
    while f"{base}_{i}" in cols:
        i += 1
    return f"{base}_{i}"


def _allocate_transient_cols(existing_cols):
    """Deterministically allocate collision-free routing + mask column names.

    Returns ``(chunk_index_col, row_index_col, mask_col)``. Because allocation
    is deterministic w.r.t. the input column set, equivalent operations produce
    equal serialized keys (no randomness in key-bearing state). The mask is
    stored under *mask_col* internally and renamed to ``MASK_COL_NAME`` only at
    emit, so a real user column of the same name is never excluded from subset
    resolution or clobbered.
    """
    ci = _make_unique_col_name(existing_cols, _CI_BASE)
    ri = _make_unique_col_name(existing_cols | {ci}, _RI_BASE)
    mask = _make_unique_col_name(existing_cols | {ci, ri}, _MASK_BASE)
    return ci, ri, mask


class LiteFrameDropDuplicates(MapReduceOperator, LiteFrameOperatorMixin):
    """Deduplicate rows or mark duplicate rows in a LiteFrame.

    A single operator that unifies ``duplicated`` (boolean mask) and
    ``drop_duplicates`` (filtered values) via two switches:

    - ``return_mask``:  emit a one-column boolean mask marking duplicate rows
      (all rows preserved).
    - ``return_value``: emit deduplicated values (duplicate rows removed).

    Parameters
    ----------
    subset : list of str, optional
        Columns to consider; ``None`` means all columns.
    keep : {"first", "last", False}, default "first"
        Which occurrence is *not* a duplicate.
    return_mask : bool, default False
        Whether to output the boolean mask.
    return_value : bool, default True
        Whether to output the deduplicated values.
    method : {"auto", "tree", "subset_tree", "shuffle"}, optional
        Tiling strategy.
    """

    _op_type_ = opcodes.DROP_DUPLICATES  # 728
    _output_type_ = OutputType.liteframe

    subset = ListField("subset", FieldTypes.string, default=None)
    keep = AnyField("keep", default="first")
    return_mask = BoolField("return_mask", default=False)
    return_value = BoolField("return_value", default=True)
    method = StringField("method", default=None)
    shuffle_size = Int32Field("shuffle_size", default=None)
    hidden_columns = ListField("hidden_columns", FieldTypes.string, default=None)
    chunk_index_col = StringField("chunk_index_col", default=None)
    row_index_col = StringField("row_index_col", default=None)
    mask_col = StringField("mask_col", default=None)

    @property
    def output_limit(self):
        # __call__ rejects return_mask & return_value both True, so the
        # effective limit is always 1. Guard defensively so an inconsistent
        # state raises here rather than silently advertising a 2-output mode.
        if self.return_mask and self.return_value:
            raise ValueError(
                "return_mask and return_value cannot both be True (unsupported)"
            )
        return 1

    def __call__(self, liteframe):
        # Validate by identity, not equality: `0 == False` and `1 == True`
        # would otherwise let keep=0/1 pass and take the wrong executor path
        # (executors use `keep is False`). Only the exact singleton False and
        # the two strings are accepted.
        if not (self.keep in ("first", "last") or self.keep is False):
            raise ValueError(f"keep must be one of {_VALID_KEEP}, got {self.keep!r}")
        if self.method is not None and self.method not in _VALID_METHOD:
            raise ValueError(
                f"method must be one of {_VALID_METHOD}, got {self.method!r}"
            )
        if not self.return_mask and not self.return_value:
            raise ValueError("at least one of return_mask or return_value must be True")
        if self.return_mask and self.return_value:
            raise ValueError(
                "return_mask and return_value cannot both be True (unsupported)"
            )
        in_dtypes = liteframe._data._physical_dtypes
        in_metadata = liteframe.frame_metadata

        # If input has virtual range_columns, materialize them into dtypes.
        # Range columns go first to match the tiler's materialization order.
        if in_metadata is not None and in_metadata.range_columns:
            rc_dtypes = {
                rc_name: wrap_arrow_dtype(pa.int64())
                for rc_name in in_metadata.range_columns
            }
            new_dtypes = pd.concat([pd.Series(rc_dtypes), in_dtypes])
        else:
            new_dtypes = in_dtypes.copy()

        # Deterministically allocate collision-free transient column names
        # against the (materialized) input schema — no randomness in
        # key-bearing state, so equivalent ops produce equal keys. Range
        # columns are materialized to Int64 and live in new_dtypes, so they
        # are probed here too.
        existing_cols = set(new_dtypes.index)
        ci, ri, mask = _allocate_transient_cols(existing_cols)
        self.chunk_index_col = ci
        self.row_index_col = ri
        self.mask_col = mask

        hidden_columns = None
        if in_metadata is not None and in_metadata.hidden_columns:
            hidden_columns = list(in_metadata.hidden_columns)
        self.hidden_columns = hidden_columns

        new_metadata = FrameMetadata(range_columns={}, hidden_columns=hidden_columns)
        n_rows = liteframe.shape[0]

        bool_dtype = wrap_arrow_dtype(pa.bool_())
        kws = []
        if self.return_mask:
            kws.append(
                {
                    "shape": (n_rows, 1),
                    "physical_dtypes": pd.Series({MASK_COL_NAME: bool_dtype}),
                    "frame_metadata": FrameMetadata(range_columns={}),
                }
            )
        if self.return_value:
            kws.append(
                {
                    "shape": (np.nan, len(new_dtypes)),
                    "physical_dtypes": new_dtypes,
                    "frame_metadata": new_metadata,
                }
            )
        results = self.new_liteframes([liteframe], kws=kws)
        if len(results) == 1:
            return results[0]
        return results
