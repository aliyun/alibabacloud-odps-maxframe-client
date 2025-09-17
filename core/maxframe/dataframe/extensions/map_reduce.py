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

import inspect
from typing import Any, Callable, List, Optional, Union

import pandas as pd


def _has_end_arg(func) -> bool:
    f_args = inspect.getfullargspec(func)
    return "end" in f_args.args or "end" in f_args.kwonlyargs


def _gen_combined_mapper(
    mapper: Callable,
    combiner: Callable,
    group_cols: List[Any],
    order_cols: List[Any],
    ascending: Union[bool, List[bool]] = True,
):
    class CombinedMapper:
        def __init__(self):
            if isinstance(mapper, type):
                self.f = mapper()
            else:
                self.f = mapper

            if isinstance(combiner, type):
                self.combiner = combiner()
            else:
                self.combiner = combiner

        def _combine_mapper_result(self, mapper_result, end=False):
            if mapper_result is None:
                return None
            res = mapper_result
            if order_cols:
                res = mapper_result.sort_values(order_cols, ascending=ascending)

            kw = {"end": end} if _has_end_arg(self.combiner) else {}
            gcols = group_cols or list(res.columns)
            return res.groupby(gcols, group_keys=False)[list(res.columns)].apply(
                self.combiner, **kw
            )

        def __call__(self, batch, end=False):
            kw = {"end": end} if _has_end_arg(self.f) else {}
            f_ret = self.f(batch, **kw)
            return self._combine_mapper_result(f_ret, end=end)

        def close(self) -> None:
            if hasattr(self.f, "close"):
                self.f.close()
            if hasattr(self.combiner, "close"):
                self.combiner.close()

    return CombinedMapper


def map_reduce(
    df,
    mapper: Optional[Callable] = None,
    reducer: Optional[Callable] = None,
    group_cols: Optional[List[Any]] = None,
    *,
    order_cols: List[Any] = None,
    ascending: Union[bool, List[bool]] = True,
    combiner: Callable = None,
    batch_rows: Optional[int] = 1024,
    mapper_dtypes: pd.Series = None,
    mapper_index: pd.Index = None,
    mapper_batch_rows: Optional[int] = None,
    reducer_dtypes: pd.Series = None,
    reducer_index: pd.Index = None,
    reducer_batch_rows: Optional[int] = None,
    ignore_index: bool = False,
):
    """
    Map-reduce API over certain DataFrames. This function is roughly
    a shortcut for

    .. code-block:: python

        df.mf.apply_chunk(mapper).groupby(group_keys).mf.apply_chunk(reducer)

    Parameters
    ----------
    mapper : function or type
        Mapper function or class.
    reducer : function or type
        Reducer function or class.
    group_cols : str or list[str]
        The keys to group after mapper. If absent, all columns in the mapped
        DataFrame will be used.
    order_cols : str or list[str]
        The columns to sort after groupby.
    ascending : bool or list[bool] or None
        Whether columns should be in ascending order or not, only effective when
        `order_cols` are specified. If a list of booleans are passed, orders of
        every column in `order_cols` are specified.
    combiner : function or class
        Combiner function or class. Should accept and returns the same schema
        of mapper outputs.
    batch_rows : int or None
        Rows in batches for mappers and reducers. Ignored if `mapper_batch_rows`
        specified for mappers or `reducer_batch_rows` specified for reducers.
        1024 by default.
    mapper_dtypes : pd.Series or dict or None
        Output dtypes of mapper stage.
    mapper_index : pd.Index or None
        Index of DataFrame returned by mappers.
    mapper_batch_rows : int or None
        Rows in batches for mappers. If specified, `batch_rows` will be ignored
        for mappers.
    reducer_dtypes : pd.Series or dict or None
        Output dtypes of reducer stage.
    reducer_index : pd.Index or None
        Index of DataFrame returned by reducers.
    reducer_batch_rows : int or None
        Rows in batches for mappers. If specified, `batch_rows` will be ignored
        for reducers.
    ignore_index : bool
        If true, indexes generated at mapper or reducer functions will be ignored.

    Returns
    -------
    output: DataFrame
        Result DataFrame after map and reduce.

    Examples
    --------

    We first define a DataFrame with a column of several words.

    >>> from collections import defaultdict
    >>> import maxframe.dataframe as md
    >>> from maxframe.udf import with_running_options
    >>> df = pd.DataFrame(
    >>>     {
    >>>         "name": ["name key", "name", "key", "name", "key name"],
    >>>         "id": [4, 2, 4, 3, 3],
    >>>         "fid": [5.3, 3.5, 4.2, 2.2, 4.1],
    >>>     }
    >>> )

    Then we write a mapper function which accepts batches in the DataFrame
    and returns counts of words in every row.

    >>> def mapper(batch):
    >>>     word_to_count = defaultdict(lambda: 0)
    >>>     for words in batch["name"]:
    >>>         for w in words.split():
    >>>             word_to_count[w] += 1
    >>>     return pd.DataFrame(
    >>>         [list(tp) for tp in word_to_count.items()], columns=["word", "count"]
    >>>     )

    After that we write a reducer function which aggregates records with
    the same word. Running options such as CPU specifications can be supplied
    as well.

    >>> @with_running_options(cpu=2)
    >>> class TestReducer:
    >>>     def __init__(self):
    >>>         self._word_to_count = defaultdict(lambda: 0)
    >>>
    >>>     def __call__(self, batch, end=False):
    >>>         word = None
    >>>         for _, row in batch.iterrows():
    >>>             word = row.iloc[0]
    >>>             self._word_to_count[row.iloc[0]] += row.iloc[1]
    >>>         if end:
    >>>             return pd.DataFrame(
    >>>                 [[word, self._word_to_count[word]]], columns=["word", "count"]
    >>>             )
    >>>
    >>>     def close(self):
    >>>         # you can do several cleanups here
    >>>         print("close")

    Finally we can call `map_reduce` with mappers and reducers specified above.

    >>> res = df.mf.map_reduce(
    >>>     mapper,
    >>>     TestReducer,
    >>>     group_cols=["word"],
    >>>     mapper_dtypes={"word": "str", "count": "int"},
    >>>     mapper_index=pd.Index([0]),
    >>>     reducer_dtypes={"word": "str", "count": "int"},
    >>>     reducer_index=pd.Index([0]),
    >>>     ignore_index=True,
    >>> )
    >>> res.execute().fetch()
       word  count
    0   key      3
    1  name      4

    See Also
    --------
    DataFrame.mf.apply_chunk, DataFrame.groupby.mf.apply_chunk
    """
    mapper_batch_rows = mapper_batch_rows or batch_rows
    reducer_batch_rows = reducer_batch_rows or batch_rows

    def check_arg(arg_type, locals_):
        if locals_.get(arg_type) is not None:
            return
        for suffix in ("dtypes", "index"):
            arg_name = f"{arg_type}_{suffix}"
            if locals_.get(arg_name) is not None:
                raise ValueError(f"Cannot specify {arg_name} when {arg_type} is None")

    if mapper is None:
        check_arg("mapper", locals())
        mapped = df
        group_cols = group_cols or df.dtypes.index
        if combiner is not None:
            raise ValueError("Combiner cannot be set when mapper is None")
    else:
        if combiner is not None:
            mapper = _gen_combined_mapper(
                mapper, combiner, group_cols, order_cols, ascending=ascending
            )
        mapped = df.mf.apply_chunk(
            mapper,
            batch_rows=mapper_batch_rows,
            dtypes=mapper_dtypes,
            output_type="dataframe",
            index=mapper_index,
        )
        group_cols = group_cols or list(df.dtypes.index)

    if reducer is None:
        check_arg("reducer", locals())
        res = mapped
    else:
        res = mapped.groupby(group_cols, group_keys=False)[
            list(mapped.dtypes.index)
        ].mf.apply_chunk(
            reducer,
            batch_rows=reducer_batch_rows,
            dtypes=reducer_dtypes,
            output_type="dataframe",
            index=reducer_index,
            order_cols=order_cols,
            ascending=ascending,
        )

    if ignore_index:
        return res.reset_index(drop=True)
    return res
