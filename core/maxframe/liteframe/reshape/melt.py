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

"""Wide-to-long ``melt`` for LiteFrame, composed from existing operators."""

import numpy as np
import pandas as pd
import pyarrow as pa
from pandas.api.types import is_list_like

from maxframe.liteframe.merge.concat import concat
from maxframe.liteframe.utils import normalize_col_name
from maxframe.utils import is_arrow_dtype_supported, wrap_arrow_dtype


def _normalize_vars(value, arg_name):
    """Normalize ``id_vars`` / ``value_vars`` to a list of lower-cased names.

    ``None`` -> ``None`` (caller applies the default); a str is wrapped.
    Non-list-like, non-string elements and duplicates raise ``ValueError``
    -- stricter than pandas (opaque ``KeyError`` / silent de-dup).
    """
    if value is None:
        return None
    if isinstance(value, str):
        return [normalize_col_name(value)]
    if not is_list_like(value):
        raise ValueError(
            f"{arg_name} must be a string or a list-like of strings, "
            f"got {type(value).__name__}"
        )
    result = []
    for element in value:
        if not isinstance(element, str):
            raise ValueError(
                f"all elements of {arg_name} must be strings, "
                f"got {type(element).__name__}"
            )
        name = normalize_col_name(element)
        if name in result:
            raise ValueError(f"duplicate column in {arg_name}: {element!r}")
        result.append(name)
    return result


def _promoted_value_dtype(dtypes):
    """The promoted dtype of *dtypes*, or ``None`` if already uniform.

    Uses pandas' own promotion as the authority: the promoted dtype is
    ``pd.concat`` of one empty ``Series`` per input dtype, so any pair pandas
    would merge into an ``object`` column (e.g. int + str, bool + int,
    duration + int where supported) raises ``ValueError`` -- LiteFrame cannot
    execute a mixed ``object`` column. Non-object results are mapped to an
    executable Arrow dtype when pandas supports it (numpy only as a dtype
    converter); under pandas < 1.5 / old pyarrow (no ``pd.ArrowDtype``) the
    numpy promotion path is used instead.
    """
    if all(d == dtypes[0] for d in dtypes):
        return None
    target = _pandas_promoted_dtype(dtypes)
    if target == object:
        raise ValueError(
            "cannot melt value columns with incompatible dtypes "
            f"{[str(d) for d in dtypes]}; cast them to a common dtype first"
        )
    if is_arrow_dtype_supported():
        if isinstance(target, pd.ArrowDtype):
            return target
        try:
            return wrap_arrow_dtype(pa.from_numpy_dtype(target))
        except (TypeError, ValueError):
            pass
    return target


def _pandas_promoted_dtype(dtypes):
    """Pandas' own promotion of *dtypes* (empty-Series ``pd.concat``).

    Old pandas (< 1.5) cannot build a ``pd.Series`` from a LiteFrame's
    FakeArrowDtype columns, so fall back to the numpy promotion path.
    """
    try:
        return pd.concat([pd.Series(dtype=dt) for dt in dtypes]).dtype
    except (TypeError, AttributeError, ValueError):
        np_dtypes = [pd.api.types.pandas_dtype(d).numpy_dtype for d in dtypes]
        result_np = np.result_type(*np_dtypes)
        if result_np == object or np.issubdtype(result_np, np.str_):
            return object
        return wrap_arrow_dtype(pa.from_numpy_dtype(result_np))


def _melt_part(data, id_cols, value_col, var_name, value_name, cast_dtype):
    """Build one value var's part: ``id_cols`` + var literal + value column,
    with an optional per-part ``astype`` so ``concat`` sees one schema."""
    part = data[id_cols].assign(**{var_name: value_col, value_name: data[value_col]})
    if cast_dtype is not None:
        part = part.astype({value_name: cast_dtype})
    return part


def melt(
    data,
    id_vars=None,
    value_vars=None,
    var_name="variable",
    value_name="value",
):
    """Unpivot a LiteFrame from wide to long format.

    Mirrors ``pandas.DataFrame.melt`` for flat, index-free frames. Each
    column in ``value_vars`` is stacked under a single ``value_name`` column,
    paired with a ``var_name`` column holding the original column name. The
    ``id_vars`` columns are repeated for every value variable. Output rows are
    in **variable-major** order -- all rows of the first value variable, then
    the second, and so on -- matching ``pd.melt`` exactly for any chunk size.

    Parameters
    ----------
    id_vars : str or list of str, optional
        Column(s) to use as identifier variables; repeated for each value
        variable. If ``None``, no identifier columns are carried.
    value_vars : str or list of str, optional
        Column(s) to unpivot. If ``None``, all columns not in ``id_vars`` are
        used, in frame order.
    var_name : str, default "variable"
        Name of the variable-name column.
    value_name : str, default "value"
        Name of the value column.

    Returns
    -------
    LiteFrame
        A new lazy LiteFrame with columns ``id_vars + [var_name, value_name]``
        and ``shape[0] * len(value_vars)`` rows in variable-major order. Call
        ``.execute()`` to materialize the results.

    Raises
    ------
    ValueError
        If ``var_name``/``value_name`` is not a string, if they collide with
        each other or with existing columns, or if ``id_vars`` and
        ``value_vars`` overlap (deliberate divergence from pandas, which
        produces a degenerate empty frame). Also raised when the value columns
        have incompatible dtypes that pandas would promote to ``object``.
    KeyError
        If a column named in ``id_vars`` or ``value_vars`` is not present.

    See Also
    --------
    pivot : Reshape without aggregation (raises on duplicates).
    pivot_table : Create a spreadsheet-style pivot table.
    concat : Stack LiteFrame objects, used internally by ``melt``.

    Examples
    --------
    >>> import maxframe.liteframe as mlf
    >>> lf = mlf.LiteFrame({'a': ['x', 'y'], 'b': [1, 3], 'c': [2, 4]})
    >>> result = lf.melt(id_vars=['a'], value_vars=['b', 'c'])
    >>> result.execute()
       a variable  value
    0  x        b      1
    1  y        b      3
    2  x        c      2
    3  y        c      4
    """
    if var_name is None:
        var_name = "variable"
    if value_name is None:
        value_name = "value"
    if not isinstance(var_name, str):
        raise ValueError("var_name must be a string")
    if not isinstance(value_name, str):
        raise ValueError("value_name must be a string")
    var_name = normalize_col_name(var_name)
    value_name = normalize_col_name(value_name)

    id_cols = _normalize_vars(id_vars, "id_vars") or []
    value_cols = _normalize_vars(value_vars, "value_vars")

    all_cols = list(data.columns)

    # Rule 5: unknown columns -> KeyError (matches pandas' message). Only
    # user-specified columns are checked; the value_vars default (rule 6)
    # draws from data.columns and so cannot be "unknown".
    specified = id_cols + (value_cols or [])
    missing = [c for c in specified if c not in all_cols]
    if missing:
        raise KeyError(
            "The following id_vars or value_vars are not present in the "
            f"DataFrame: {missing}"
        )
    # Rule 7: id_vars ∩ value_vars -> ValueError (Q3 divergence from pandas).
    if value_cols is not None:
        overlap = set(id_cols) & set(value_cols)
        if overlap:
            raise ValueError(
                f"id_vars and value_vars must not overlap: {sorted(overlap)}"
            )
    # Rule 11: var_name == value_name -> ValueError.
    if var_name == value_name:
        raise ValueError("var_name and value_name must not be the same")
    # Rule 9: value_name cannot match any existing column.
    if value_name in all_cols:
        raise ValueError(
            f"value_name ({value_name}) cannot match an element in the "
            "DataFrame columns."
        )
    # Rule 10: var_name cannot match an id_vars column.
    if var_name in id_cols:
        raise ValueError(f"var_name ({var_name}) cannot match an id_vars column")
    # Rule 12: var_name matching a non-id, non-value column is allowed
    # (that column is dropped anyway) -- nothing to enforce.

    # Rule 6: default value_vars = all columns except id_vars, in frame order.
    if value_cols is None:
        id_set = set(id_cols)
        value_cols = [c for c in all_cols if c not in id_set]

    # Rule 13: explicitly empty value_vars mirrors pandas. Without any
    # id_vars pandas treats an empty value_vars list like the default and
    # melts every column; with id_vars the result is a true 0-row frame.
    # Build the id-only frame assign-then-head: on a 0-column projection
    # ``head(0).assign(...)`` can materialize one fake row at runtime while
    # lazy metadata claims 0 rows (LF#1897).
    if not value_cols:
        if not id_cols:
            value_cols = [c for c in all_cols]
        if not value_cols:
            # No columns at all (e.g. melt on an empty frame): pandas yields
            # a 0-row frame with var/value columns; assign-then-head keeps
            # lazy metadata and runtime consistent.
            return data[id_cols].assign(**{var_name: "", value_name: 0.0}).head(0)

    cast_dtype = _promoted_value_dtype([data.dtypes[c] for c in value_cols])
    parts = [
        _melt_part(data, id_cols, col, var_name, value_name, cast_dtype)
        for col in value_cols
    ]
    return concat(parts, axis=0)
