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

import pandas as pd

from ... import opcodes
from ...serialization.serializables import AnyField, DictField
from ..window.expanding import Expanding
from .core import BaseGroupByWindowOp

_supported_funcs = {
    "sum",
    "prod",
    "mean",
    "std",
    "var",
    "median",
    "min",
    "max",
    "count",
}


class GroupByExpandingAgg(BaseGroupByWindowOp):
    _op_type_ = opcodes.EXPANDING_AGG

    func = AnyField("func", default=None)
    kwargs = DictField("kwargs", default_factory=dict)

    def _calc_mock_result_df(self, mock_groupby):
        expanding_args = self.window_params.copy()
        # exclude MF-specific args
        for key in Expanding._mf_specific_fields:
            expanding_args.pop(key, None)

        def apply_func(frame, **_):
            func = self.func
            if self.func == "prod":
                func = lambda x: x.prod()
            return frame.expanding(**expanding_args).agg(func)

        is_cumcount = self.kwargs.get("cumcount", False)
        # special handling for cumcount which returns a series
        if is_cumcount:
            return mock_groupby.cumcount(ascending=self.kwargs.get("ascending", True))
        if self.func in ["sum", "max", "min", "prod"]:
            return getattr(mock_groupby, f"cum{self.func}")()
        return mock_groupby.apply(apply_func, include_groups=False)

    def get_sort_cols_to_asc(self) -> Dict[Any, bool]:
        res = super().get_sort_cols_to_asc()
        if self.func == "count" and self.kwargs.get("ascending") is not None:
            res = {k: not v for k, v in res.items()}
        return res


class ExpandingGroupby(Expanding):
    def aggregate(self, func, **kwargs):
        if func not in _supported_funcs:
            raise NotImplementedError(f"func {func} is not supported")
        op = GroupByExpandingAgg(
            func=func,
            groupby_params=self.input.op.groupby_params,
            window_params=self.params,
            kwargs=kwargs,
        )
        return op(self.input)

    agg = aggregate


def expanding(
    groupby,
    min_periods=1,
    *,
    shift=0,
    reverse_range=False,
    order_cols=None,
    ascending=True,
):
    """
    Return an expanding grouper, providing expanding
    functionality per group.

    Parameters
    ----------
    min_periods : int, default None
        Minimum number of observations in window required to have a value;
        otherwise, result is ``np.nan``.

    shift : int, default 0
        If specified, the window will be shifted by `shift` rows (or data will be
        shifted by `-shift` rows) before computing window function.

    reverse_range : bool, default False
        If True, the window for current row is expanded from the last row to
        the current instead of the first row.

    Returns
    -------
    maxframe.dataframe.groupby.ExpandingGroupby

    See Also
    --------
    Series.groupby : Apply a function groupby to a Series.
    DataFrame.groupby : Apply a function groupby
        to each row or column of a DataFrame.
    """
    return ExpandingGroupby(
        input=groupby,
        min_periods=min_periods,
        shift=shift,
        reverse_range=reverse_range,
        order_cols=order_cols,
        ascending=ascending,
    )


_cum_doc_template = """
Cumulative %(func_name)s for each group.

Returns
-------
Series or DataFrame

See Also
--------
Series.groupby : Apply a function groupby to a Series.
DataFrame.groupby : Apply a function groupby
    to each row or column of a DataFrame.

Examples
--------
For SeriesGroupBy:

>>> import maxframe.dataframe as md
>>> lst = ['a', 'a', 'b']
>>> ser = md.Series([6, 2, 0], index=lst)
>>> ser.execute()
a    6
a    2
b    0
dtype: int64
>>> ser.groupby(level=0).cum%(func_name)s().execute()
%(series_result)s

For DataFrameGroupBy:

>>> data = [[1, 8, 2], [1, 2, 5], [2, 6, 9]]
>>> df = md.DataFrame(data, columns=["a", "b", "c"],
...                   index=["fox", "gorilla", "lion"])
>>> df.execute()
          a   b   c
fox       1   8   2
gorilla   1   2   5
lion      2   6   9
>>> df.groupby("a").groups.execute()
{1: ['fox', 'gorilla'], 2: ['lion']}
>>> df.groupby("a").cum%(func_name)s().execute()
%(df_result)s
"""


def _build_cum_docs(func_name):
    ser = pd.Series([6, 2, 0], index=list("aab"))
    data = [[1, 8, 2], [1, 2, 5], [2, 6, 9]]
    df = pd.DataFrame(data, columns=["a", "b", "c"], index=["fox", "gorilla", "lion"])
    cum_func_name = f"cum{func_name}"
    return _cum_doc_template % dict(
        func_name=func_name,
        series_result=getattr(ser.groupby(level=0), cum_func_name)().to_string(),
        df_result=getattr(df.groupby("a"), cum_func_name)().to_string(),
    )


def cumcount(groupby, ascending: bool = True):
    """
    Number each item in each group from 0 to the length of that group - 1.

    Essentially this is equivalent to

    .. code-block:: python

        self.apply(lambda x: pd.Series(np.arange(len(x)), x.index))

    Parameters
    ----------
    ascending : bool, default True
        If False, number in reverse, from length of group - 1 to 0.

    Returns
    -------
    Series
        Sequence number of each element within each group.

    See Also
    --------
    .ngroup : Number the groups themselves.

    Examples
    --------
    >>> import maxframe.tensor as mt
    >>> import maxframe.dataframe as md
    >>> df = md.DataFrame([['a'], ['a'], ['a'], ['b'], ['b'], ['a']],
    ...                   columns=['A'])
    >>> df.execute()
       A
    0  a
    1  a
    2  a
    3  b
    4  b
    5  a
    >>> df.groupby('A').cumcount().execute()
    0    0
    1    1
    2    2
    3    0
    4    1
    5    3
    dtype: int64
    >>> df.groupby('A').cumcount(ascending=False).execute()
    0    3
    1    2
    2    1
    3    1
    4    0
    5    0
    dtype: int64
    """
    return groupby.expanding().agg("count", ascending=ascending, cumcount=True)


def cummin(groupby):
    return groupby.expanding().min()


def cummax(groupby):
    return groupby.expanding().max()


def cumprod(groupby):
    return groupby.expanding().prod()


def cumsum(groupby):
    return groupby.expanding().sum()


for _func in (cummin, cummax, cumprod, cumsum):
    _func.__doc__ = _build_cum_docs(_func.__name__[3:])
