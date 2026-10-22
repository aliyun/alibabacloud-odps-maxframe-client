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

"""``LiteFrame.struct.explode`` implementation.

Lowers through :func:`LiteFrame.transform` with ``axis=1`` (the
expression-based per-row path) and a ``@builtin_function`` UDF.
``struct.explode`` is a row-preserving (1:1) operation -- every input struct
becomes exactly one output row of field columns, including missing structs
(one all-NA row).  Lowering through ``transform(axis=1)`` builds a
:class:`LiteFrameProjection` with one per-field projection, so a downstream
``lf[["a"]]`` prunes to just that field's projection (projection pushdown),
and the row count and hidden columns are preserved (unlike ``flat_apply``,
which is 0-N and drops hidden columns).
"""

import pandas as pd

from maxframe.lib.dtypes_extension import is_struct_dtype
from maxframe.udf import builtin_function
from maxframe.utils import is_na_scalar, wrap_arrow_dtype


@builtin_function
def _explode_struct_row(row, column=None, fields=None):
    """Expand the struct in ``column`` into one row of field columns.

    Mirrors ``Series.struct.explode``: each struct becomes one row with one
    column per field; a missing struct becomes one row of all-NA fields.

    Returns a ``{field: value}`` dict per row; ``transform(axis=1)`` expands
    it to one column per field.
    """
    value = row[column]
    if is_na_scalar(value):
        return {field: pd.NA for field in fields}
    return {field: value.get(field, pd.NA) for field in fields}


def _struct_field_dtypes(col_dtype):
    """Return an ordered ``{field: dtype}`` mapping for an Arrow struct dtype."""
    pa_type = col_dtype.pyarrow_dtype
    return {
        pa_type[i].name: wrap_arrow_dtype(pa_type[i].type)
        for i in range(pa_type.num_fields)
    }


def explode_struct_column(liteframe):
    """``LiteFrame.struct.explode``: expand a struct column into field columns.

    Mirrors ``pandas.Series.struct.explode``: the single struct column is
    replaced by one column per struct field (each with its field dtype); a
    missing struct becomes one row of all-NA fields.  Row count is unchanged.
    """
    if len(liteframe.columns) != 1:
        raise ValueError("struct.explode requires a single-column LiteFrame")
    column = liteframe.columns[0]
    col_dtype = liteframe.dtypes.iloc[0]
    if not is_struct_dtype(col_dtype):
        raise AttributeError("Can only use .struct.explode with struct values")
    field_dtypes = _struct_field_dtypes(col_dtype)
    fields = list(field_dtypes)
    # transform(axis=1): the UDF returns a {field: value} dict per row,
    # expanded to one column per field.  struct.explode is 1:1 (one output
    # row per input row), so transform(axis=1) preserves the row count and
    # carries hidden columns.  Unlike apply(axis=1) (operator-based), this
    # builds a LiteFrameProjection with one per-field projection, enabling
    # projection pushdown (a downstream lf[["a"]] prunes to that field).
    return liteframe.transform(
        _explode_struct_row,
        axis=1,
        dtypes=field_dtypes,
        column=column,
        fields=fields,
    )
