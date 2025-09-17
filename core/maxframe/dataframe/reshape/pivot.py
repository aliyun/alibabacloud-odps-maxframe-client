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
from ...serialization.serializables import AnyField, KeyField
from ...utils import no_default
from ..operators import DataFrameOperator, DataFrameOperatorMixin
from ..utils import build_df, make_column_list, parse_index


class DataFramePivot(DataFrameOperator, DataFrameOperatorMixin):
    _op_type_ = opcodes.PIVOT

    values = AnyField("values", default=None)
    index = AnyField("index", default=None)
    columns = AnyField("columns", default=None)

    agg_results = KeyField("agg_results", default=None)
    fill_value = AnyField("fill_value", default=None)

    def __init__(self, aggfunc=None, **kw):
        if aggfunc is None:
            aggfunc = "mean"
        super().__init__(aggfunc=aggfunc, **kw)
        self._output_types = [OutputType.dataframe]

    @classmethod
    def _set_inputs(cls, op: "DataFramePivot", inputs: List[EntityData]):
        super()._set_inputs(op, inputs)
        if op.agg_results is not None:  # pragma: no branch
            op.agg_results = inputs[-1]

    def __call__(self, df):
        index_list = make_column_list(self.index, df.dtypes)
        columns_list = make_column_list(self.columns, df.dtypes)

        if not index_list:
            index_data = pd.Index([])
        elif len(index_list) == 1:
            index_data = pd.Index(
                [], dtype=df.dtypes[index_list[0]], name=index_list[0]
            )
        else:
            index_data = pd.MultiIndex.from_frame(build_df(df[index_list]))
        index_value = parse_index(index_data, df)

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


def pivot(data, columns, index=None, values=None):
    """
    Return reshaped DataFrame organized by given index / column values.

    Reshape data (produce a "pivot" table) based on column values. Uses
    unique values from specified `index` / `columns` to form axes of the
    resulting DataFrame. This function does not support data
    aggregation, multiple values will result in a MultiIndex in the
    columns. See the :ref:`User Guide <reshaping>` for more on reshaping.

    Parameters
    ----------
    index : str or object or a list of str, optional
        Column to use to make new frame's index. If None, uses
        existing index.

    columns : str or object or a list of str
        Column to use to make new frame's columns.

    values : str, object or a list of the previous, optional
        Column(s) to use for populating new frame's values. If not
        specified, all remaining columns will be used and the result will
        have hierarchically indexed columns.

    Returns
    -------
    DataFrame
        Returns reshaped DataFrame.

    Raises
    ------
    ValueError:
        When there are any `index`, `columns` combinations with multiple
        values. `DataFrame.pivot_table` when you need to aggregate.

    See Also
    --------
    DataFrame.pivot_table : Generalization of pivot that can handle
        duplicate values for one index/column pair.
    DataFrame.unstack : Pivot based on the index values instead of a
        column.
    wide_to_long : Wide panel to long format. Less flexible but more
        user-friendly than melt.

    Notes
    -----
    For finer-tuned control, see hierarchical indexing documentation along
    with the related stack/unstack methods.

    Examples
    --------
    >>> import maxframe.dataframe as md
    >>> df = md.DataFrame({'foo': ['one', 'one', 'one', 'two', 'two',
    ...                            'two'],
    ...                    'bar': ['A', 'B', 'C', 'A', 'B', 'C'],
    ...                    'baz': [1, 2, 3, 4, 5, 6],
    ...                    'zoo': ['x', 'y', 'z', 'q', 'w', 't']})
    >>> df.execute()
        foo   bar  baz  zoo
    0   one   A    1    x
    1   one   B    2    y
    2   one   C    3    z
    3   two   A    4    q
    4   two   B    5    w
    5   two   C    6    t

    >>> df.pivot(index='foo', columns='bar', values='baz').execute()
    bar  A   B   C
    foo
    one  1   2   3
    two  4   5   6

    >>> df.pivot(index='foo', columns='bar', values=['baz', 'zoo']).execute()
          baz       zoo
    bar   A  B  C   A  B  C
    foo
    one   1  2  3   x  y  z
    two   4  5  6   q  w  t

    You could also assign a list of column names or a list of index names.

    >>> df = md.DataFrame({
    ...        "lev1": [1, 1, 1, 2, 2, 2],
    ...        "lev2": [1, 1, 2, 1, 1, 2],
    ...        "lev3": [1, 2, 1, 2, 1, 2],
    ...        "lev4": [1, 2, 3, 4, 5, 6],
    ...        "values": [0, 1, 2, 3, 4, 5]})
    >>> df.execute()
        lev1 lev2 lev3 lev4 values
    0   1    1    1    1    0
    1   1    1    2    2    1
    2   1    2    1    3    2
    3   2    1    2    4    3
    4   2    1    1    5    4
    5   2    2    2    6    5

    >>> df.pivot(index="lev1", columns=["lev2", "lev3"], values="values").execute()
    lev2    1         2
    lev3    1    2    1    2
    lev1
    1     0.0  1.0  2.0  NaN
    2     4.0  3.0  NaN  5.0

    >>> df.pivot(index=["lev1", "lev2"], columns=["lev3"], values="values").execute()
          lev3    1    2
    lev1  lev2
       1     1  0.0  1.0
             2  2.0  NaN
       2     1  4.0  3.0
             2  NaN  5.0

    A ValueError is raised if there are any duplicates.

    >>> df = md.DataFrame({"foo": ['one', 'one', 'two', 'two'],
    ...                    "bar": ['A', 'A', 'B', 'C'],
    ...                    "baz": [1, 2, 3, 4]})
    >>> df.execute()
       foo bar  baz
    0  one   A    1
    1  one   A    2
    2  two   B    3
    3  two   C    4

    Notice that the first two rows are the same for our `index`
    and `columns` arguments.

    >>> df.pivot(index='foo', columns='bar', values='baz').execute()
    Traceback (most recent call last):
       ...
    ValueError: Index contains duplicate entries, cannot reshape
    """
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

    op = DataFramePivot(
        values=values,
        index=index,
        columns=columns,
    )
    return op(data)
