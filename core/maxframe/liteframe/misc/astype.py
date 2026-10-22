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

from maxframe.liteframe.arithmetic.core import (
    _build_fused_projection,
    _resolve_column_expr,
    _resolve_named_expr,
)
from maxframe.liteframe.expressions import LiteFrameAstype, NamedExpr
from maxframe.liteframe.utils import normalize_col_name, validate_errors
from maxframe.utils import wrap_arrow_dtype

try:
    from maxframe.lib.dtypes_extension import ArrowDtype
except ImportError:  # pragma: no cover
    ArrowDtype = None

try:
    from maxframe.lib.dtypes_extension.blob import ExternalBlobDtype
except ImportError:  # pragma: no cover
    ExternalBlobDtype = None


def _normalize_dtype(dtype):
    """Normalize a dtype specification to an ArrowDtype where possible."""
    # Already an ArrowDtype — pass through unchanged
    if ArrowDtype is not None and isinstance(dtype, ArrowDtype):
        return dtype
    # pandas StringDtype("pyarrow") — pass through unchanged
    if isinstance(dtype, pd.StringDtype) and dtype.storage == "pyarrow":
        return dtype
    # Other pandas extension dtypes (including ExternalBlobDtype) — pass through unchanged
    if isinstance(dtype, pd.api.extensions.ExtensionDtype):
        return dtype
    # bytes type → ArrowDtype(pa.binary())
    if dtype is bytes:
        return wrap_arrow_dtype(pa.binary())
    # String dtype specifications
    if isinstance(dtype, str):
        dtype_lower = dtype.lower()
        if dtype_lower == "blob":
            if ExternalBlobDtype is None:
                raise ImportError("blob dtype requires maxframe blob extension")
            return ExternalBlobDtype()
        if dtype_lower == "binary":
            return wrap_arrow_dtype(pa.binary())
    # Try converting numpy dtype or string to ArrowDtype
    try:
        if isinstance(dtype, np.dtype):
            pa_type = pa.from_numpy_dtype(dtype)
        else:
            # dtype is a string like "int64", "float64", etc.
            pa_type = pa.from_numpy_dtype(np.dtype(dtype))
        return wrap_arrow_dtype(pa_type)
    except (TypeError, ValueError):
        # Types that can't be converted (e.g., datetime64) — fall back to numpy
        if isinstance(dtype, np.dtype):
            return dtype
        return np.dtype(dtype)


def astype(lf, dtype, errors="raise"):
    """Cast a LiteFrame to a specified dtype.

    Parameters
    ----------
    dtype : data type, or dict of column name -> data type
        Use a numpy.dtype or Python type to cast entire LiteFrame to
        the same type. Alternatively, use {col: dtype, ...}, where col is a
        column label and dtype is a numpy.dtype or Python type to cast one
        or more of the DataFrame's columns to column-specific types.
    errors : {'raise', 'coerce'}, default 'raise'
        Control raising of exceptions on invalid data for provided dtype.

        - ``raise`` : allow exceptions to be raised
        - ``coerce`` : uncastable values become null and the output dtype
          changes to the target type.

        ``'ignore'`` is not supported by LiteFrame astype; use ``'coerce'``
        for null-on-failure semantics.

    Returns
    -------
    LiteFrame
        Casted LiteFrame.

    See Also
    --------
    LiteFrame.eval : Evaluate a string expression to add columns.
    """
    validate_errors(errors)
    if errors == "ignore":
        raise ValueError(
            "errors='ignore' is not supported for LiteFrame astype. "
            "Use errors='coerce' for null-on-failure semantics."
        )

    if isinstance(dtype, dict):
        dtype_map = {}
        for col, dt in dtype.items():
            normalized = normalize_col_name(col)
            if normalized not in lf.columns:
                raise KeyError(
                    f"Column '{col}' not found in LiteFrame columns {list(lf.columns)}"
                )
            dtype_map[normalized] = _normalize_dtype(dt)
    else:
        dtype_map = {col: _normalize_dtype(dtype) for col in lf.columns}

    projections = []

    for col_name in lf.columns:
        col_dtype = lf.dtypes[col_name]
        if col_name in dtype_map:
            target_dtype = dtype_map[col_name]
            operand = _resolve_column_expr(lf, col_name, col_dtype)
            expr = LiteFrameAstype(
                operand=operand,
                dtype=target_dtype,
                target_dtype=target_dtype,
                errors=errors,
            )
            projections.append(NamedExpr(name=col_name, expr=expr))
        else:
            # Column not in dtype_map, keep as-is (preserve rename alias)
            projections.append(_resolve_named_expr(lf, col_name, col_dtype))
    return _build_fused_projection(lf, projections)
