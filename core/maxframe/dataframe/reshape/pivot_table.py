# Copyright 1999-2025 Alibaba Group Holding Ltd.
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

from typing import List

import numpy as np
import pandas as pd
from pandas.api.types import is_list_like

from ... import opcodes
from ...core import EntityData, OutputType
from ...serialization.serializables import AnyField, BoolField, KeyField, StringField
from ...utils import no_default
from ..operators import DataFrameOperator, DataFrameOperatorMixin
from ..utils import build_df, make_column_list, parse_index


class DataFramePivotTable(DataFrameOperator, DataFrameOperatorMixin):
    _op_type_ = opcodes.PIVOT_TABLE

    values = AnyField("values", default=None)
    index = AnyField("index", default=None)
    columns = AnyField("columns", default=None)
    aggfunc = AnyField("aggfunc", default="mean")
    fill_value = AnyField("fill_value", default=None)
    margins = BoolField("margins", default=False)
    dropna = BoolField("dropna", default=True)
    margins_name = StringField("margins_name", default=None)
    sort = BoolField("sort", default=False)

    # generate intermediate agg result to infer value of columns
    agg_results = KeyField("agg_results", default=None)

    def __init__(self, aggfunc=None, **kw):
        if aggfunc is None:
            aggfunc = "mean"
        super().__init__(aggfunc=aggfunc, **kw)
        self._output_types = [OutputType.dataframe]

    @classmethod
    def _set_inputs(cls, op: "DataFramePivotTable", inputs: List[EntityData]):
        super()._set_inputs(op, inputs)
        if op.agg_results is not None:  # pragma: no branch
            op.agg_results = inputs[-1]

    def __call__(self, df):
        index_list = make_column_list(self.index, df.dtypes)
        columns_list = make_column_list(self.columns, df.dtypes)
        values_list = make_column_list(self.values, df.dtypes)

        if not index_list:
            index_data = pd.Index([])
        elif len(index_list) == 1:
            index_data = pd.Index(
                [], dtype=df.dtypes[index_list[0]], name=index_list[0]
            )
        else:
            index_data = pd.MultiIndex.from_frame(build_df(df[index_list]))
        index_value = parse_index(index_data, df)

        if columns_list is None:  # output columns can be determined
            sel_df = df
            groupby_obj = sel_df.groupby(index_list)
            if values_list:
                groupby_obj = groupby_obj[values_list]
            aggregated_df = groupby_obj.agg(self.aggfunc)
            index_value = aggregated_df.index_value
            columns_value = aggregated_df.columns_value
            dtypes = aggregated_df.dtypes
        else:
            self.agg_results = df[columns_list].drop_duplicates()
            columns_value = dtypes = None

        inputs = [df]
        if self.agg_results is not None:
            inputs.append(self.agg_results)
        return self.new_dataframe(
            inputs,
            shape=(np.nan, np.nan),
            dtypes=dtypes,
            columns_value=columns_value,
            index_value=index_value,
        )


def pivot_table(
    data,
    values=None,
    index=None,
    columns=None,
    aggfunc="mean",
    fill_value=None,
    margins=False,
    dropna=True,
    margins_name="All",
    sort=True,
):
    """
    Create a spreadsheet-style pivot table as a DataFrame.

    The levels in the pivot table will be stored in MultiIndex objects
    (hierarchical indexes) on the index and columns of the result DataFrame.

    Parameters
    ----------
    values : column to aggregate, optional
    index : column, Grouper, array, or list of the previous
        If an array is passed, it must be the same length as the data. The
        list can contain any of the other types (except list).
        Keys to group by on the pivot table index.  If an array is passed,
        it is being used as the same manner as column values.
    columns : column, Grouper, array, or list of the previous
        If an array is passed, it must be the same length as the data. The
        list can contain any of the other types (except list).
        Keys to group by on the pivot table column.  If an array is passed,
        it is being used as the same manner as column values.
    aggfunc : function, list of functions, dict, default numpy.mean
        If list of functions passed, the resulting pivot table will have
        hierarchical columns whose top level are the function names
        (inferred from the function objects themselves)
        If dict is passed, the key is column to aggregate and value
        is function or list of functions.
    fill_value : scalar, default None
        Value to replace missing values with (in the resulting pivot table,
        after aggregation).
    margins : bool, default False
        Add all row / columns (e.g. for subtotal / grand totals).
    dropna : bool, default True
        Do not include columns whose entries are all NaN.
    margins_name : str, default 'All'
        Name of the row / column that will contain the totals
        when margins is True.
    sort : bool, default True
        Specifies if the result should be sorted.

    Returns
    -------
    DataFrame
        An Excel style pivot table.

    See Also
    --------
    DataFrame.pivot : Pivot without aggregation that can handle
        non-numeric data.
    DataFrame.melt: Unpivot a DataFrame from wide to long format,
        optionally leaving identifiers set.
    wide_to_long : Wide panel to long format. Less flexible but more
        user-friendly than melt.

    Examples
    --------
    >>> import numpy as np
    >>> import maxframe.dataframe as md
    >>> df = md.DataFrame({"A": ["foo", "foo", "foo", "foo", "foo",
    ...                          "bar", "bar", "bar", "bar"],
    ...                    "B": ["one", "one", "one", "two", "two",
    ...                          "one", "one", "two", "two"],
    ...                    "C": ["small", "large", "large", "small",
    ...                          "small", "large", "small", "small",
    ...                          "large"],
    ...                    "D": [1, 2, 2, 3, 3, 4, 5, 6, 7],
    ...                    "E": [2, 4, 5, 5, 6, 6, 8, 9, 9]})
    >>> df.execute()
         A    B      C  D  E
    0  foo  one  small  1  2
    1  foo  one  large  2  4
    2  foo  one  large  2  5
    3  foo  two  small  3  5
    4  foo  two  small  3  6
    5  bar  one  large  4  6
    6  bar  one  small  5  8
    7  bar  two  small  6  9
    8  bar  two  large  7  9

    This first example aggregates values by taking the sum.

    >>> table = md.pivot_table(df, values='D', index=['A', 'B'],
    ...                        columns=['C'], aggfunc=np.sum)
    >>> table.execute()
    C        large  small
    A   B
    bar one    4.0    5.0
        two    7.0    6.0
    foo one    4.0    1.0
        two    NaN    6.0

    We can also fill missing values using the `fill_value` parameter.

    >>> table = md.pivot_table(df, values='D', index=['A', 'B'],
    ...                        columns=['C'], aggfunc=np.sum, fill_value=0)
    >>> table.execute()
    C        large  small
    A   B
    bar one      4      5
        two      7      6
    foo one      4      1
        two      0      6

    The next example aggregates by taking the mean across multiple columns.

    >>> table = md.pivot_table(df, values=['D', 'E'], index=['A', 'C'],
    ...                        aggfunc={'D': np.mean,
    ...                                 'E': np.mean})
    >>> table.execute()
                    D         E
    A   C
    bar large  5.500000  7.500000
        small  5.500000  8.500000
    foo large  2.000000  4.500000
        small  2.333333  4.333333

    We can also calculate multiple types of aggregations for any given
    value column.

    >>> table = md.pivot_table(df, values=['D', 'E'], index=['A', 'C'],
    ...                        aggfunc={'D': np.mean,
    ...                                 'E': [min, max, np.mean]})
    >>> table.execute()
                    D    E
                mean  max      mean  min
    A   C
    bar large  5.500000  9.0  7.500000  6.0
        small  5.500000  9.0  8.500000  8.0
    foo large  2.000000  5.0  4.500000  4.0
        small  2.333333  6.0  4.333333  2.0
    """
    if index is None and columns is None:
        raise ValueError(
            "No group keys passed, need to specify at least one of index or columns"
        )

    values_list = make_column_list(values, data.dtypes)
    index_list = make_column_list(index, data.dtypes)
    columns_list = make_column_list(columns, data.dtypes)

    name_to_attr = {"values": values_list, "index": index_list, "columns": columns_list}
    for key, val in name_to_attr.items():
        if val is None:
            continue
        if not is_list_like(val):
            raise ValueError(f"Need to specify {key} as a list-like object.")
        non_exist_key = next((c for c in val if c not in data.dtypes.index), no_default)
        if non_exist_key is not no_default:
            raise ValueError(
                f"Column {non_exist_key} specified in {key} is not a valid column."
            )

    if columns is None and not margins:
        if values_list:
            data = data[index_list + values_list]
        return data.groupby(index, sort=sort).agg(aggfunc)

    op = DataFramePivotTable(
        values=values,
        index=index,
        columns=columns,
        aggfunc=aggfunc,
        fill_value=fill_value,
        margins=margins,
        dropna=dropna,
        margins_name=margins_name,
        sort=sort,
    )
    return op(data)
