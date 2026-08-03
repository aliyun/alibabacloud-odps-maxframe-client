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

import pyarrow as pa

from maxframe.liteframe.arithmetic.core import _resolve_column_expr, _root_source_lf
from maxframe.liteframe.core import LiteFrame
from maxframe.liteframe.expressions import (
    LiteFrameAnd,
    LiteFrameInvert,
    LiteFrameIsNA,
    LiteFrameLiteral,
    LiteFrameOr,
)
from maxframe.liteframe.operators.filter import LiteFrameFilter
from maxframe.liteframe.utils import normalize_col_name
from maxframe.utils import wrap_arrow_dtype


def dropna(lf, axis=0, how="any", subset=None):
    """Drop rows or columns with missing values.

    Parameters
    ----------
    axis : {0, 1}, default 0
        0: drop rows containing NAs. 1: not yet supported.
    how : {"any", "all"}, default "any"
        "any": drop if any NA present. "all": drop only if all values are NA.
    subset : list of column names, optional
        Columns to inspect for NA values (axis=0 only).
    """
    if axis != 0:
        raise NotImplementedError("dropna with axis=1 is not yet supported")

    if how not in ("any", "all"):
        raise ValueError(
            f"invalid how option: '{how}'. Only 'any' or 'all' are allowed"
        )

    if subset is not None:
        if isinstance(subset, str):
            columns = [normalize_col_name(subset)]
        else:
            columns = [normalize_col_name(c) for c in subset]
    else:
        columns = list(lf.columns)

    bool_dtype = wrap_arrow_dtype(pa.bool_())
    isna_exprs = [
        LiteFrameIsNA(
            operand=_resolve_column_expr(lf, col, lf.dtypes[col]),
            dtype=bool_dtype,
        )
        for col in columns
    ]

    if len(isna_exprs) == 0:
        # Empty subset: how="any" → keep all rows; how="all" → drop all rows
        combined = LiteFrameLiteral(value=(how == "all"), dtype=bool_dtype)
    elif len(isna_exprs) == 1:
        combined = isna_exprs[0]
    else:
        combiner = LiteFrameOr if how == "any" else LiteFrameAnd
        combined = isna_exprs[0]
        for expr in isna_exprs[1:]:
            combined = combiner(left=combined, right=expr, dtype=bool_dtype)

    predicate = LiteFrameInvert(operand=combined, dtype=bool_dtype)

    # The fused predicate references root source columns, so the filter
    # must be applied to the root source.  If *lf* is itself a projection,
    # re-apply that projection on top of the filtered root source so the
    # output schema matches lf.
    root_source = _root_source_lf(lf)
    filter_op = LiteFrameFilter(predicate=predicate)
    filtered = filter_op(LiteFrame(root_source))
    if root_source is lf._data:
        return filtered
    return lf.op(filtered)
