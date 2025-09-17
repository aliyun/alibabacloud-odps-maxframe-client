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

from typing import Any, Dict

from ... import opcodes
from ...serialization.serializables import BoolField, FieldTypes, ListField, StringField
from ..utils import make_column_list
from .core import BaseGroupByWindowOp


class GroupByRank(BaseGroupByWindowOp):
    _op_type_ = opcodes.RANK

    method = StringField("method", default=None)
    ascending = ListField("ascending", FieldTypes.bool, default=True)
    na_option = StringField("na_option", default=None)
    pct = BoolField("pct", default=False)

    def _calc_mock_result_df(self, mock_groupby):
        return mock_groupby.rank(
            method=self.method,
            ascending=self.ascending,
            na_option=self.na_option,
            pct=self.pct,
        )

    def get_sort_cols_to_asc(self) -> Dict[Any, bool]:
        if self.inputs[0].ndim == 1:
            return {self.inputs[0].name: self.ascending}
        selections = make_column_list(
            self.groupby_params.get("selection", []), self.inputs[0].dtypes
        )
        if not selections:
            by_cols = set(
                make_column_list(
                    self.groupby_params.get("by") or [], self.inputs[0].dtypes
                )
            )
            selections = [c for c in self.inputs[0].dtypes.index if c not in by_cols]

        if len(self.ascending) < len(selections):
            ascending = [self.ascending[0]] * len(selections)
        else:
            ascending = self.ascending
        return dict(zip(selections, ascending))


def rank(groupby, method="average", ascending=True, na_option="keep", pct=False):
    """
    Provide the rank of values within each group.

    Parameters
    ----------
    method : {'average', 'min', 'max', 'first', 'dense'}, default 'average'
        * average: average rank of group.
        * min: lowest rank in group.
        * max: highest rank in group.
        * first: ranks assigned in order they appear in the array.
        * dense: like 'min', but rank always increases by 1 between groups.
    ascending : bool, default True
        False for ranks by high (1) to low (N).
    na_option : {'keep', 'top', 'bottom'}, default 'keep'
        * keep: leave NA values where they are.
        * top: smallest rank if ascending.
        * bottom: smallest rank if descending.
    pct : bool, default False
        Compute percentage rank of data within each group.

    Returns
    -------
    DataFrame with ranking of values within each group

    See Also
    --------
    Series.groupby : Apply a function groupby to a Series.
    DataFrame.groupby : Apply a function groupby
        to each row or column of a DataFrame.

    Examples
    --------
    >>> import maxframe.dataframe as md
    >>> df = md.DataFrame(
    ...     {
    ...         "group": ["a", "a", "a", "a", "a", "b", "b", "b", "b", "b"],
    ...         "value": [2, 4, 2, 3, 5, 1, 2, 4, 1, 5],
    ...     }
    ... )
    >>> df.execute()
      group  value
    0     a      2
    1     a      4
    2     a      2
    3     a      3
    4     a      5
    5     b      1
    6     b      2
    7     b      4
    8     b      1
    9     b      5
    >>> for method in ['average', 'min', 'max', 'dense', 'first']:
    ...     df[f'{method}_rank'] = df.groupby('group')['value'].rank(method)
    >>> df.execute()
      group  value  average_rank  min_rank  max_rank  dense_rank  first_rank
    0     a      2           1.5       1.0       2.0         1.0         1.0
    1     a      4           4.0       4.0       4.0         3.0         4.0
    2     a      2           1.5       1.0       2.0         1.0         2.0
    3     a      3           3.0       3.0       3.0         2.0         3.0
    4     a      5           5.0       5.0       5.0         4.0         5.0
    5     b      1           1.5       1.0       2.0         1.0         1.0
    6     b      2           3.0       3.0       3.0         2.0         3.0
    7     b      4           4.0       4.0       4.0         3.0         4.0
    8     b      1           1.5       1.0       2.0         1.0         2.0
    9     b      5           5.0       5.0       5.0         4.0         5.0
    """
    if not isinstance(ascending, list):
        ascending = [ascending]
    op = GroupByRank(
        method=method,
        ascending=ascending,
        na_option=na_option,
        pct=pct,
        groupby_params=groupby.op.groupby_params,
    )
    return op(groupby)
