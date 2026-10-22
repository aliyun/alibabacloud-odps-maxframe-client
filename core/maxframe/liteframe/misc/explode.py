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

"""LiteFrame ``explode`` and shared list-explode primitives.

The DataFrame-level :func:`explode` and the ``list.flatten`` accessor both
build on the ``_explode_list_row`` ``@builtin_function`` UDF via
:func:`LiteFrame.flat_apply`, so the :class:`LiteFrameFunctional` operator
does not treat them as custom code (no isolation environment): a built-in
function is reloaded by name in the worker instead of being pickled.  The
struct counterpart lives in
:mod:`maxframe.liteframe.accessors.struct_.explode`.
"""

from collections import Counter

import pandas as pd

from maxframe.lib.dtypes_extension import is_list_dtype
from maxframe.liteframe.utils import normalize_col_name
from maxframe.udf import builtin_function
from maxframe.utils import is_na_scalar, wrap_arrow_dtype


@builtin_function
def _explode_list_row(row, columns=None, skip_empty=False):
    """Yield one output row per element index across ``columns``.

    Mirrors ``pandas.DataFrame.explode`` (``skip_empty=False``): each row's
    list values in ``columns`` are expanded in lock-step to one output row per
    element index.  ``flat_apply`` is called with ``subset=columns`` so the
    UDF receives only the exploded columns and emits just them; the broadcast
    (non-exploded) columns are repeated by ``flat_apply`` and inserted at
    their original positions by the column-order contract, keeping the
    exploded columns in place.  Missing or empty lists collapse to a single
    NA element; mismatched element counts across ``columns`` raise
    ``ValueError``, matching pandas.

    ``skip_empty=True`` mirrors ``Series.list.flatten`` (single column):
    empty or missing lists drop the row entirely instead of yielding an NA
    row.
    """
    base = row.to_dict()
    normalized = []
    for column in columns:
        value = base.get(column)
        if is_na_scalar(value):
            normalized.append([] if skip_empty else [pd.NA])
            continue
        elements = list(value)
        if not elements:
            normalized.append([] if skip_empty else [pd.NA])
        else:
            normalized.append(elements)
    # skip_empty (list.flatten, single column) drops empty/missing rows.
    if skip_empty and not normalized[0]:
        return
    counts = {len(elems) for elems in normalized}
    if len(counts) > 1:
        raise ValueError("columns must have matching element counts")
    count = counts.pop() if counts else 0
    for i in range(count):
        yield {column: elems[i] for column, elems in zip(columns, normalized)}


def _list_value_dtype(col_dtype):
    """Return the element dtype of an Arrow list dtype."""
    return wrap_arrow_dtype(col_dtype.pyarrow_dtype.value_type)


def _explode_list_columns(liteframe, columns, skip_empty):
    """Explode list ``columns`` via a single ``flat_apply`` (preserves order).

    ``columns`` are exploded in lock-step (one ``flat_apply`` regardless of
    how many columns are given); each exploded column's dtype is replaced with
    its list ``value_type`` and the remaining columns keep their input dtypes.
    """
    dtypes = liteframe.dtypes
    missing = [c for c in columns if c not in dtypes.index]
    if missing:
        raise KeyError(missing[0])
    # Build the function-output dtypes: only the exploded columns, each with
    # its list value_type.  The broadcast (non-exploded) columns inherit their
    # input dtypes automatically and are inserted at their original positions
    # by the flat_apply column-order contract.
    func_dtypes = {}
    for column in columns:
        col_dtype = dtypes[column]
        if not is_list_dtype(col_dtype):
            raise ValueError(
                f"column {column!r} has dtype {col_dtype!r}; explode requires a "
                f"list dtype"
            )
        func_dtypes[column] = _list_value_dtype(col_dtype)
    # subset=columns -> the UDF receives only the exploded columns and emits
    # just them.  When the exploded columns are non-contiguous in the original
    # schema, flat_apply groups them into one func-out block at the first
    # subset column's position, which would reorder non-exploded columns
    # (e.g. explode(["a","c"]) on cols [a,b,c] yields [a,c,b]).  Append a
    # fused projection that restores the original column order so explode
    # stays pandas-compatible ([a,b,c]) regardless of subset contiguity.
    result = liteframe.flat_apply(
        _explode_list_row,
        dtypes=func_dtypes,
        subset=list(columns),
        columns=list(columns),
        skip_empty=skip_empty,
    )
    original_order = list(dtypes.index)
    return (
        result.select(original_order)
        if list(result.columns) != original_order
        else result
    )


def explode(self, column):
    """Explode list column(s) into multiple rows.

    LiteFrame analogue of ``pandas.DataFrame.explode``: each list-like value
    in ``column`` is expanded to one row per element; the other columns are
    broadcast (repeated) across the expanded rows and keep their original
    order.  An empty list or missing value produces a single NA row.

    When multiple columns are given they are exploded in lock-step (one output
    row per element index across all of them), matching pandas; their element
    counts must agree per row.  The expansion is a single ``flat_apply``.

    Parameters
    ----------
    column : str or list of str
        The list column(s) to explode.

    Returns
    -------
    LiteFrame
        A new LiteFrame with the same columns; each exploded column now holds
        its element dtype (the list's ``value_type``).

    See Also
    --------
    flat_apply : Apply a function per row, expanding to 0-N rows.
    LiteFrame.list.flatten : Flatten a list column.

    Examples
    --------
    >>> import maxframe.liteframe as mlf
    >>> import pandas as pd
    >>> import pyarrow as pa
    >>> from maxframe.utils import wrap_arrow_dtype
    >>> _list = wrap_arrow_dtype(pa.list_(pa.int64()))
    >>> raw = pd.DataFrame({'b': [4, 5]})
    >>> raw['a'] = pd.Series([[1, 2], [3]]).astype(_list)
    >>> lf = mlf.LiteFrame(raw)
    >>> result = lf.explode('a')
    >>> result.execute()
       b  a
    0  4  1
    0  4  2
    1  5  3
    """
    columns = [column] if isinstance(column, str) else list(column)
    if not columns:
        raise ValueError("explode requires at least one column")
    # LiteFrame lowercases stored column names; normalize the caller's names so
    # case-insensitive selection (lf["Val"]) stays consistent with explode, and
    # reject duplicate names after normalization (mirrors pandas / LiteFrame's
    # other selection APIs).  An empty column sequence would otherwise silently
    # yield an empty frame, and duplicates would surface later as a duplicate
    # projection.
    columns = [normalize_col_name(c) for c in columns]
    dupes = sorted({c for c, n in Counter(columns).items() if n > 1})
    if dupes:
        raise ValueError(
            f"explode columns must be unique after normalization; duplicates: "
            f"{dupes}"
        )
    return _explode_list_columns(self, columns, skip_empty=False)
