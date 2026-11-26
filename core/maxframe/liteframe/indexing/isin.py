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

from maxframe.core import TILEABLE_TYPE
from maxframe.liteframe.arithmetic.core import (
    _extract_projection_exprs,
    _rewrite_column_refs,
    _root_source_lf,
)
from maxframe.liteframe.core import LiteFrame
from maxframe.liteframe.expressions import (
    _ISIN_NOT_SCALAR_MSG,
    LiteFrameBaseExpr,
    LiteFrameColumn,
    LiteFrameIsInExpr,
    LiteFrameLiteral,
    NamedExpr,
    _normalize_isin_values,
)
from maxframe.liteframe.operators.project import LiteFrameProjection
from maxframe.liteframe.utils import normalize_col_name
from maxframe.utils import wrap_arrow_dtype

_BOOL_DTYPE = wrap_arrow_dtype(pa.bool_())


def isin(lf, values):
    """Return a boolean LiteFrame showing whether each element is in *values*.

    Parameters
    ----------
    values : iterable or dict
        If an iterable, the same membership test is applied to every
        column.  If a dict, ``{column_name: values}`` applies a per-column
        membership test; columns not present in the dict are all ``False``.

    Returns
    -------
    LiteFrame
        Boolean LiteFrame with the same shape (visible columns only).

    Examples
    --------
    >>> import maxframe.liteframe as mlf
    >>> lf = mlf.LiteFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    >>> lf.isin([1, 4]).execute()
           a      b
    0   True   True
    1  False  False
    2  False  False
    """
    if isinstance(values, (TILEABLE_TYPE, LiteFrameBaseExpr)):
        raise NotImplementedError(_ISIN_NOT_SCALAR_MSG)

    # Boolean columns are produced for the visible *data* columns only.  The
    # virtual range index (e.g. the default range index) is not data: it must
    # be passed through unchanged instead of being turned into a bool column.
    meta = lf._data.frame_metadata
    range_cols = list(meta.range_columns) if (meta and meta.range_columns) else []
    range_set = set(range_cols)
    columns = [c for c in lf.columns if c not in range_set]
    existing_exprs = _extract_projection_exprs(lf)

    # Normalise to per-column values: dict gives per-column, list applies to all.
    if isinstance(values, dict):
        # Normalize string keys like column names so the case-insensitive
        # column contract (shared with select/filter) holds.  Non-string keys
        # match nothing, and unmatched keys are ignored as pandas does.
        normalized = {}
        for key, vals in values.items():
            normalized[normalize_col_name(key) if isinstance(key, str) else key] = vals
        col_values = {
            col: _normalize_isin_values(normalized[col])
            for col in columns
            if col in normalized
        }
    else:
        flat = _normalize_isin_values(values)
        col_values = {col: flat for col in columns}

    projections = []
    # Virtual range pass-throughs go first so the affine range column stays at
    # a contiguous prefix position of the output schema; the projection
    # analyzer materializes a range column that appears after a non-range
    # output.  Identity pass-through keeps it virtual and carries the metadata.
    for rc in range_cols:
        named = NamedExpr(
            name=rc,
            expr=LiteFrameColumn(name=rc, dtype=lf.dtypes[rc]),
            dtype=lf.dtypes[rc],
        )
        projections.append(_rewrite_column_refs(named, existing_exprs))

    for col in columns:
        if col in col_values:
            expr = LiteFrameIsInExpr(
                operand=LiteFrameColumn(name=col, dtype=lf.dtypes[col]),
                values=col_values[col],
                dtype=_BOOL_DTYPE,
            )
        else:
            expr = LiteFrameLiteral(value=False, dtype=_BOOL_DTYPE)
        named = NamedExpr(name=col, expr=expr, dtype=_BOOL_DTYPE)
        projections.append(_rewrite_column_refs(named, existing_exprs))

    op = LiteFrameProjection(projections=projections)
    return op(LiteFrame(_root_source_lf(lf)))
