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

from pandas.api.types import is_list_like

from maxframe.liteframe.operators.reshape.pivot_table import LiteFramePivotTable
from maxframe.liteframe.reduction.custom_agg import CustomAgg
from maxframe.liteframe.utils import is_numeric_dtype, normalize_col_name
from maxframe.udf import builtin_function
from maxframe.utils import no_default


@builtin_function
class PivotUniqueReduction(CustomAgg):
    """Custom aggregation for ``pivot()``: collapse each group to one row.

    Used as the internal aggfunc when ``pivot()`` delegates to
    ``pivot_table()``, so each ``(index, columns)`` combo collapses to a
    single row.  Raises ``ValueError`` when any combo has more than one
    row — matching pandas ``pivot()`` which rejects any duplicate
    ``(index, columns)`` pair, even when the values are identical.

    Associativity across tree-reduction stages is achieved with a
    ``_mf_pivot_row_count`` column injected by ``pre``: each input row gets
    count 1 at map stage, counts are summed at combine/agg stages, and
    ``post`` checks for duplicates and strips the column.
    """

    _COUNT_COL = "_mf_pivot_row_count"

    def __init__(self):
        super().__init__(name="pivot_unique")

    def pre(self, df):
        import polars as pl

        # Inject a count column (1 per row) so the agg stage can detect
        # duplicates associatively via sum.
        return df.with_columns(pl.lit(1).alias(self._COUNT_COL))

    def agg(self, df):
        import polars as pl

        # df is a per-group polars DataFrame (input_frame_type='polars').
        # Sum the count column across rows in this group, keep the first row.
        total = df[self._COUNT_COL].sum()
        return df.head(1).with_columns(pl.lit(total).alias(self._COUNT_COL))

    def post(self, df):
        # After full tree reduction, each group has one row with the total
        # count.  Any count > 1 means the (index, columns) key was duplicated.
        if (df[self._COUNT_COL] > 1).any():
            raise ValueError("Index contains duplicate entries, cannot reshape")
        return df.drop(self._COUNT_COL)


def _normalize_and_validate(data, values, index, columns):
    """Promote str params to lists and validate column existence.

    LiteFrame lowercases all column names, so user-provided names are
    normalized before checking against ``data.dtypes.index``.
    """
    name_to_val = {"values": values, "index": index, "columns": columns}
    for key, val in name_to_val.items():
        if val is None:
            continue
        if not is_list_like(val):
            raise ValueError(f"Need to specify {key} as a list-like object.")
        non_exist_key = next(
            (c for c in val if normalize_col_name(c) not in data.dtypes.index),
            no_default,
        )
        if non_exist_key is not no_default:
            raise ValueError(
                f"Column {non_exist_key} specified in {key} " "is not a valid column."
            )


def pivot_table(
    data,
    values=None,
    index=None,
    columns=None,
    aggfunc="mean",
    fill_value=None,
    dropna=True,
    sort=True,
):
    """Create a spreadsheet-style pivot table as a LiteFrame.

    Aggregate ``data`` by ``index`` columns and spread ``columns`` values
    into wide format.  The result is a flat LiteFrame — column names are
    plain strings (no MultiIndex).

    Parameters
    ----------
    data : LiteFrame
        Source LiteFrame.
    values : str or list of str, optional
        Column(s) to aggregate.  If ``None``, all non-index/non-columns
        columns are aggregated.
    index : str or list of str, optional
        Column(s) to group by on the pivot table index (rows).  At least
        one of ``index`` or ``columns`` must be specified.
    columns : str or list of str, optional
        Column(s) whose distinct values become the new columns.  At least
        one of ``index`` or ``columns`` must be specified.
    aggfunc : str, default 'mean'
        Aggregation function.  A string names a built-in reduction
        (``"sum"``, ``"mean"``, ``"min"``, ``"max"``, ``"count"``,
        ``"median"``, etc.).
    fill_value : scalar, default None
        Value to replace missing values with in the resulting pivot table.
    dropna : bool, default True
        Do not include columns whose entries are all NaN.
    sort : bool, default True
        Sort the result by the ``columns`` values.

    Returns
    -------
    LiteFrame
        Pivot table as a flat LiteFrame.  Index columns appear first,
        followed by one column per distinct ``columns`` value.

    See Also
    --------
    pivot : Reshape without aggregation (raises on duplicates).
    groupby : Group-by aggregation without spreading columns.

    Examples
    --------
    >>> import maxframe.liteframe as mlf
    >>> lf = mlf.LiteFrame({
    ...     'city': ['A', 'A', 'B', 'B'],
    ...     'item': ['x', 'y', 'x', 'y'],
    ...     'sales': [10, 20, 30, 40],
    ... })
    >>> result = lf.pivot_table(values='sales', index='city', columns='item',
    ...                         aggfunc='sum')
    >>> result.execute()
      city   x   y
    0    A  10  20
    1    B  30  40
    """
    if index is None and columns is None:
        raise ValueError(
            "No group keys passed, need to specify at least one of " "index or columns"
        )

    values_list = [values] if values and not is_list_like(values) else values
    index_list = [index] if index and not is_list_like(index) else index
    columns_list = [columns] if columns and not is_list_like(columns) else columns

    _normalize_and_validate(data, values_list, index_list, columns_list)

    # LiteFrame lowercases all column names; normalize user-provided names
    # so the operator and downstream tiler/executor see stored columns.
    norm_values = [normalize_col_name(c) for c in values_list] if values_list else None
    norm_index = [normalize_col_name(c) for c in index_list] if index_list else index
    norm_columns = (
        [normalize_col_name(c) for c in columns_list] if columns_list else columns
    )

    # Fallback: columns=None → plain groupby + agg (no pivot needed).
    # Matches pandas behavior when margins=False.  Uses normalized names
    # so user-provided mixed-case input works.
    if norm_columns is None:
        gb = data.groupby(by=norm_index, sort=sort)
        if norm_values is not None:
            gb = gb[norm_values]
        return gb.agg(aggfunc)
    # Resolve values=None to numeric-only columns here (where
    # ``data.dtypes`` is available) so the operator, tiler, and executor
    # all see the same set.  Matches pandas, which ignores non-numeric
    # columns.  ``pivot()`` resolves its own values before calling us.
    if norm_values is None:
        key_cols = set(norm_index or []) | set(norm_columns)
        resolved = [
            normalize_col_name(c)
            for c in data.dtypes.index
            if normalize_col_name(c) not in key_cols
            and is_numeric_dtype(data.dtypes[c])
        ]
        if not resolved:
            raise ValueError(
                "No numeric columns found for aggregation. "
                "Specify values explicitly."
            )
        norm_values = resolved

    # index=None is handled by the executor (adds a synthetic constant
    # column for polars pivot, then drops it).
    op = LiteFramePivotTable(
        values=norm_values,
        index=norm_index,
        columns=norm_columns,
        aggfunc=aggfunc,
        fill_value=fill_value,
        dropna=dropna,
        sort=sort,
    )
    return op(data)


def pivot(data, *, columns, index=None, values=None):
    """Return reshaped LiteFrame organized by given index / column values.

    Reshape data based on column values.  Uses unique values from
    specified ``index`` / ``columns`` to form axes of the resulting
    LiteFrame.  This function does not support data aggregation --
    multiple values for an index/column pair raise ``ValueError``.

    Parameters
    ----------
    data : LiteFrame
        Source LiteFrame.
    columns : str or list of str
        Column(s) whose distinct values become the new columns.
    index : str or list of str, optional
        Column(s) to use as the row index of the result.
    values : str or list of str, optional
        Column(s) to populate the new frame's values.  If ``None``, all
        non-index/non-columns columns are used.

    Returns
    -------
    LiteFrame
        Reshaped LiteFrame.  Index columns appear first, followed by one
        column per distinct ``columns`` value.

    Raises
    ------
    ValueError
        When there are any ``index`` / ``columns`` combinations with
        multiple values.  Use ``pivot_table`` when you need to aggregate.

    See Also
    --------
    pivot_table : Pivot table with aggregation.

    Examples
    --------
    >>> import maxframe.liteframe as mlf
    >>> lf = mlf.LiteFrame({
    ...     'city': ['A', 'A', 'B', 'B'],
    ...     'item': ['x', 'y', 'x', 'y'],
    ...     'sales': [10, 20, 30, 40],
    ... })
    >>> result = lf.pivot(index='city', columns='item', values='sales')
    >>> result.execute()
      city   x   y
    0    A  10  20
    """
    # pivot() carries ALL non-key columns (no aggregation), unlike
    # pivot_table() which selects numeric-only.  Resolve here so the shared
    # pivot_table() path does not need to distinguish aggfunc types.
    if values is None:
        values_list = [index] if index and not is_list_like(index) else index
        columns_list = [columns] if columns and not is_list_like(columns) else columns
        _normalize_and_validate(data, None, values_list, columns_list)
        norm_index = (
            [normalize_col_name(c) for c in values_list] if values_list else index
        )
        norm_columns = (
            [normalize_col_name(c) for c in columns_list] if columns_list else columns
        )
        key_cols = set(norm_index or []) | set(norm_columns or [])
        values = [
            normalize_col_name(c)
            for c in data.dtypes.index
            if normalize_col_name(c) not in key_cols
        ]

    return pivot_table(
        data,
        values=values,
        index=index,
        columns=columns,
        aggfunc=PivotUniqueReduction,
        dropna=False,
        sort=True,
    )
