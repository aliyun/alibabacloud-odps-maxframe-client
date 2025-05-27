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

import logging
from abc import abstractmethod
from collections import namedtuple
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
from pandas import Index

from ... import opcodes
from ...core import OutputType
from ...serialization.serializables import (
    AnyField,
    BoolField,
    DictField,
    Int32Field,
    NamedTupleField,
    Serializable,
    StringField,
    TupleField,
)
from ...utils import lazy_import
from ..core import DataFrame, Series
from ..operators import DataFrameOperator, DataFrameOperatorMixin
from ..utils import build_df, parse_index

logger = logging.getLogger(__name__)
DEFAULT_BLOOM_FILTER_CHUNK_THRESHOLD = 10
# use bloom filter to filter large DataFrame
BLOOM_FILTER_OPTIONS = [
    "max_elements",
    "error_rate",
    "apply_chunk_size_threshold",
    "filter",
]
BLOOM_FILTER_ON_OPTIONS = ["large", "small", "both"]
DEFAULT_BLOOM_FILTER_ON = "large"

cudf = lazy_import("cudf")

MergeSplitInfo = namedtuple("MergeSplitInfo", "split_side, split_index, nsplits")


class JoinHint(Serializable):
    @abstractmethod
    def verify_params(
        self,
        hint_on_df: Union[DataFrame, Series],
        on: str,
        is_on_index: bool,
        how: str,
        is_hint_for_left: bool,
    ):
        pass

    @abstractmethod
    def verify_can_work_with(self, other: "JoinHint"):
        pass


class MapJoinHint(JoinHint):
    def verify_params(
        self,
        hint_on_df: Union[DataFrame, Series],
        on: str,
        is_on_index: bool,
        how: str,
        is_hint_for_left: bool,
    ):
        if how in ("cross", "outer"):
            raise ValueError(
                "Invalid join hint, MapJoinHint is not support in cross and outer join"
            )

    def verify_can_work_with(self, other: JoinHint):
        if isinstance(other, SkewJoinHint):
            raise ValueError(
                "Invalid join hint, SkewJoinHint cannot work with MapJoinHint"
            )


class DistributedMapJoinHint(JoinHint):
    shard_count = Int32Field("shard_count")
    replica_count = Int32Field("replica_count", default=1)

    def verify_params(
        self,
        hint_on_df: Union[DataFrame, Series],
        on: str,
        is_on_index: bool,
        how: str,
        is_hint_for_left: bool,
    ):
        if how in ("cross", "outer"):
            raise ValueError(
                "Invalid join hint, DistributedMapJoinHint is not support in cross and outer join"
            )
        if not hasattr(self, "shard_count"):
            raise ValueError(
                "Invalid DistributedMapJoinHint, shard_count must be specified"
            )
        if self.shard_count <= 0 or self.replica_count <= 0:
            raise ValueError(
                "Invalid DistributedMapJoinHint, shard_count and replica_count must be greater than 0"
            )

    def verify_can_work_with(self, other: JoinHint):
        pass


class SkewJoinHint(JoinHint):
    columns = AnyField("columns", default=None)

    @staticmethod
    def _check_index_levels(index, level_list):
        selected_levels = set()
        valid_levels = set(range(index.nlevels))
        valid_level_names = set(index.names)

        for item in level_list:
            if isinstance(item, int):
                if item not in valid_levels:
                    raise ValueError(f"Level {item} is not a valid index level")
                if item in selected_levels:
                    raise ValueError(f"Level {item} is selected multiple times")
                selected_levels.add(item)
            elif isinstance(item, str):
                if item not in valid_level_names:
                    raise ValueError(f"'{item}' is not a valid index level name")
                level = index.names.index(item)
                if level in selected_levels:
                    raise ValueError(
                        f"'{item}' (Level {level}) is selected multiple times"
                    )
                selected_levels.add(level)
            else:
                raise ValueError(f"Invalid input type: {type(item)}")

    @staticmethod
    def _check_columns(join_on_columns, column_list):
        selected_columns = set()
        valid_columns = set(join_on_columns)

        for item in column_list:
            if isinstance(item, int):
                if item < 0 or item >= len(join_on_columns):
                    raise ValueError(f"Column index {item} is out of range")
                col_name = join_on_columns[item]
                if col_name in selected_columns:
                    raise ValueError(
                        f"Column '{col_name}' (index {item}) is selected multiple times"
                    )
                selected_columns.add(col_name)
            elif isinstance(item, str):
                if item not in valid_columns:
                    raise ValueError(f"'{item}' is not a valid column name")
                if item in selected_columns:
                    raise ValueError(f"Column '{item}' is selected multiple times")
                selected_columns.add(item)
            else:
                raise ValueError(f"Invalid input type: {type(item)}")

    def verify_params(
        self,
        hint_on_df: Union[DataFrame, Series],
        on: str,
        is_on_index: bool,
        how: str,
        is_hint_for_left: bool,
    ):
        if how in ("cross", "outer"):
            raise ValueError(
                "Invalid join hint, map join is not support in cross and outer join"
            )
        if is_hint_for_left and how == "right":
            raise ValueError(
                "Invalid join hint, right join can only use SkewJoinHint on right frame"
            )
        elif not is_hint_for_left and how == "left":
            raise ValueError(
                "Invalid join hint, left join can only use SkewJoinHint on left frame"
            )

        # check columns
        if self.columns is None:
            return

        if not isinstance(self.columns, list):
            raise TypeError("Invalid SkewJoinHint, `columns` must be a list")

        if all(isinstance(item, (int, str)) for item in self.columns):
            # if elements are int (levels) or str (index names or column names)
            self._verify_valid_index_or_columns(
                self.columns, hint_on_df.index_value.to_pandas(), on, is_on_index
            )
        elif all(isinstance(c, dict) for c in self.columns):
            # dict with column names and values
            cols_set = set(self.columns[0].keys())
            if any(cols_set != set(c.keys()) for c in self.columns):
                raise ValueError(
                    "Invalid SkewJoinHint, all values in `columns` need to have same columns"
                )

            self._verify_valid_index_or_columns(
                cols_set, hint_on_df.index_value.to_pandas(), on, is_on_index
            )
        else:
            raise TypeError("Invalid SkewJoinHint, annot accept `columns` type")

    def verify_can_work_with(self, other: JoinHint):
        if isinstance(other, SkewJoinHint):
            raise ValueError(
                "Invalid join hint, SkewJoinHint cannot work with MapJoinHint"
            )

    @staticmethod
    def _verify_valid_index_or_columns(
        skew_join_columns: Iterable[Union[int, str]],
        frame_index: Index,
        on: Union[str, List[str]],
        is_on_index: bool,
    ):
        if isinstance(on, str):
            on = [on]
        on_columns = set(frame_index.names if is_on_index else on)
        for col in skew_join_columns:
            if isinstance(col, int):
                if col < 0 or col >= len(on_columns):
                    raise ValueError(
                        f"Invalid, SkeJoinHint, `{col}` is out of join on columns range"
                    )
            else:
                if col not in on_columns:
                    raise ValueError(
                        f"Invalid, SkeJoinHint, '{col}' is not a valid column name"
                    )


class DataFrameMerge(DataFrameOperator, DataFrameOperatorMixin):
    _op_type_ = opcodes.DATAFRAME_MERGE

    # workaround for new field since v1.0.0rc2
    # todo remove this when all versions below v1.0.0rc1 is eliminated
    _legacy_new_non_primitives = ["left_hint", "right_hint"]

    how = StringField("how")
    on = AnyField("on")
    left_on = AnyField("left_on")
    right_on = AnyField("right_on")
    left_index = BoolField("left_index")
    right_index = BoolField("right_index")
    sort = BoolField("sort")
    suffixes = TupleField("suffixes")
    copy_ = BoolField("copy_")
    indicator = BoolField("indicator")
    validate = AnyField("validate")
    method = StringField("method")
    auto_merge = StringField("auto_merge")
    auto_merge_threshold = Int32Field("auto_merge_threshold")
    bloom_filter = AnyField("bloom_filter")
    bloom_filter_options = DictField("bloom_filter_options")

    # only for broadcast merge
    split_info = NamedTupleField("split_info")
    left_hint = AnyField("left_hint", default=None)
    right_hint = AnyField("right_hint", default=None)

    def __init__(self, copy=None, output_types=None, **kwargs):
        super().__init__(copy_=copy, _output_types=output_types, **kwargs)

    def __call__(self, left, right):
        empty_left, empty_right = build_df(left), build_df(right)

        # validate arguments.
        merged = empty_left.merge(
            empty_right,
            how=self.how,
            on=self.on,
            left_on=self.left_on,
            right_on=self.right_on,
            left_index=self.left_index,
            right_index=self.right_index,
            sort=self.sort,
            suffixes=self.suffixes,
            copy=self.copy_,
            indicator=self.indicator,
            validate=self.validate,
        )

        # update default values.
        if self.on is None and self.left_on is None and self.right_on is None:
            if not self.left_index or not self.right_index:
                # use the common columns
                left_cols = empty_left.columns
                right_cols = empty_right.columns
                common_cols = left_cols.intersection(right_cols)
                self.left_on = self.right_on = list(common_cols)

        # the `index_value` doesn't matter.
        index_tokenize_objects = [
            left,
            right,
            self.how,
            self.left_on,
            self.right_on,
            self.left_index,
            self.right_index,
        ]
        return self.new_dataframe(
            [left, right],
            shape=(np.nan, merged.shape[1]),
            dtypes=merged.dtypes,
            index_value=parse_index(merged.index, *index_tokenize_objects),
            columns_value=parse_index(merged.columns, store_data=True),
        )


def merge(
    df: Union[DataFrame, Series],
    right: Union[DataFrame, Series],
    how: str = "inner",
    on: Union[str, List[str]] = None,
    left_on: str = None,
    right_on: str = None,
    left_index: bool = False,
    right_index: bool = False,
    sort: bool = False,
    suffixes: Tuple[Optional[str], Optional[str]] = ("_x", "_y"),
    copy: bool = True,
    indicator: bool = False,
    validate: str = None,
    method: str = "auto",
    auto_merge: str = "both",
    auto_merge_threshold: int = 8,
    bloom_filter: Union[bool, str] = "auto",
    bloom_filter_options: Dict[str, Any] = None,
    left_hint: JoinHint = None,
    right_hint: JoinHint = None,
) -> DataFrame:
    """
    Merge DataFrame or named Series objects with a database-style join.

    A named Series object is treated as a DataFrame with a single named column.

    The join is done on columns or indexes. If joining columns on
    columns, the DataFrame indexes *will be ignored*. Otherwise if joining indexes
    on indexes or indexes on a column or columns, the index will be passed on.
    When performing a cross merge, no column specifications to merge on are
    allowed.

    Parameters
    ----------
    right : DataFrame or named Series
        Object to merge with.
    how : {'left', 'right', 'outer', 'inner'}, default 'inner'
        Type of merge to be performed.

        * left: use only keys from left frame, similar to a SQL left outer join;
          preserve key order.
        * right: use only keys from right frame, similar to a SQL right outer join;
          preserve key order.
        * outer: use union of keys from both frames, similar to a SQL full outer
          join; sort keys lexicographically.
        * inner: use intersection of keys from both frames, similar to a SQL inner
          join; preserve the order of the left keys.

    on : label or list
        Column or index level names to join on. These must be found in both
        DataFrames. If `on` is None and not merging on indexes then this defaults
        to the intersection of the columns in both DataFrames.
    left_on : label or list, or array-like
        Column or index level names to join on in the left DataFrame. Can also
        be an array or list of arrays of the length of the left DataFrame.
        These arrays are treated as if they are columns.
    right_on : label or list, or array-like
        Column or index level names to join on in the right DataFrame. Can also
        be an array or list of arrays of the length of the right DataFrame.
        These arrays are treated as if they are columns.
    left_index : bool, default False
        Use the index from the left DataFrame as the join key(s). If it is a
        MultiIndex, the number of keys in the other DataFrame (either the index
        or a number of columns) must match the number of levels.
    right_index : bool, default False
        Use the index from the right DataFrame as the join key. Same caveats as
        left_index.
    sort : bool, default False
        Sort the join keys lexicographically in the result DataFrame. If False,
        the order of the join keys depends on the join type (how keyword).
    suffixes : list-like, default is ("_x", "_y")
        A length-2 sequence where each element is optionally a string
        indicating the suffix to add to overlapping column names in
        `left` and `right` respectively. Pass a value of `None` instead
        of a string to indicate that the column name from `left` or
        `right` should be left as-is, with no suffix. At least one of the
        values must not be None.
    copy : bool, default True
        If False, avoid copy if possible.
    indicator : bool or str, default False
        If True, adds a column to the output DataFrame called "_merge" with
        information on the source of each row. The column can be given a different
        name by providing a string argument. The column will have a Categorical
        type with the value of "left_only" for observations whose merge key only
        appears in the left DataFrame, "right_only" for observations
        whose merge key only appears in the right DataFrame, and "both"
        if the observation's merge key is found in both DataFrames.
    validate : str, optional
        If specified, checks if merge is of specified type.

        * "one_to_one" or "1:1": check if merge keys are unique in both
          left and right datasets.
        * "one_to_many" or "1:m": check if merge keys are unique in left
          dataset.
        * "many_to_one" or "m:1": check if merge keys are unique in right
          dataset.
        * "many_to_many" or "m:m": allowed, but does not result in checks.
    method : {"auto", "shuffle", "broadcast"}, default auto
        "broadcast" is recommended when one DataFrame is much smaller than the other,
        otherwise, "shuffle" will be a better choice. By default, we choose method
        according to actual data size.
    auto_merge : {"both", "none", "before", "after"}, default both
        Auto merge small chunks before or after merge

        * "both": auto merge small chunks before and after,
        * "none": do not merge small chunks
        * "before": only merge small chunks before merge
        * "after": only merge small chunks after merge
    auto_merge_threshold : int, default 8
        When how is "inner", merged result could be much smaller than original DataFrame,
        if the number of chunks is greater than the threshold,
        it will merge small chunks automatically.
    bloom_filter: bool, str, default "auto"
        Use bloom filter to optimize merge
    bloom_filter_options: dict
        * "max_elements": max elements in bloom filter,
          default value is the max size of all input chunks
        * "error_rate": error raite, default 0.1.
        * "apply_chunk_size_threshold": min chunk size of input chunks to apply bloom filter, default 10
          when chunk size of left and right is greater than this threshold, apply bloom filter
        * "filter": "large", "small", "both", default "large"
          decides to filter on large, small or both DataFrames.
    left_hint: JoinHint, default None
        Join strategy to use for left frame. When data skew occurs, consider these strategies to avoid long-tail issues,
        but use them cautiously to prevent OOM and unnecessary overhead.
    right_hint: JoinHint, default None
        Join strategy to use for right frame.


    Returns
    -------
    DataFrame
        A DataFrame of the two merged objects.

    Examples
    --------
    >>> import maxframe.dataframe as md
    >>> df1 = md.DataFrame({'lkey': ['foo', 'bar', 'baz', 'foo'],
    ...                     'value': [1, 2, 3, 5]})
    >>> df2 = md.DataFrame({'rkey': ['foo', 'bar', 'baz', 'foo'],
    ...                     'value': [5, 6, 7, 8]})
    >>> df1.execute()
        lkey value
    0   foo      1
    1   bar      2
    2   baz      3
    3   foo      5
    >>> df2.execute()
        rkey value
    0   foo      5
    1   bar      6
    2   baz      7
    3   foo      8

    Merge df1 and df2 on the lkey and rkey columns. The value columns have
    the default suffixes, _x and _y, appended.

    >>> df1.merge(df2, left_on='lkey', right_on='rkey').execute()
      lkey  value_x rkey  value_y
    0  foo        1  foo        5
    1  foo        1  foo        8
    2  foo        5  foo        5
    3  foo        5  foo        8
    4  bar        2  bar        6
    5  baz        3  baz        7

    Merge DataFrames df1 and df2 with specified left and right suffixes
    appended to any overlapping columns.

    >>> df1.merge(df2, left_on='lkey', right_on='rkey',
    ...           suffixes=('_left', '_right')).execute()
      lkey  value_left rkey  value_right
    0  foo           1  foo            5
    1  foo           1  foo            8
    2  foo           5  foo            5
    3  foo           5  foo            8
    4  bar           2  bar            6
    5  baz           3  baz            7

    Merge DataFrames df1 and df2, but raise an exception if the DataFrames have
    any overlapping columns.

    >>> df1.merge(df2, left_on='lkey', right_on='rkey', suffixes=(False, False)).execute()
    Traceback (most recent call last):
    ...
    ValueError: columns overlap but no suffix specified:
        Index(['value'], dtype='object')

    >>> df1 = md.DataFrame({'a': ['foo', 'bar'], 'b': [1, 2]})
    >>> df2 = md.DataFrame({'a': ['foo', 'baz'], 'c': [3, 4]})
    >>> df1.execute()
          a  b
    0   foo  1
    1   bar  2
    >>> df2.execute()
          a  c
    0   foo  3
    1   baz  4

    >>> df1.merge(df2, how='inner', on='a').execute()
          a  b  c
    0   foo  1  3

    >>> df1.merge(df2, how='left', on='a').execute()
          a  b  c
    0   foo  1  3.0
    1   bar  2  NaN
    """
    if (isinstance(df, Series) and df.name is None) or (
        isinstance(right, Series) and right.name is None
    ):
        raise ValueError("Cannot merge a Series without a name")
    if method is None:
        method = "auto"
    if method not in [
        "auto",
        "shuffle",
        "broadcast",
    ]:  # pragma: no cover
        raise NotImplementedError(f"{method} merge is not supported")
    if auto_merge not in ["both", "none", "before", "after"]:  # pragma: no cover
        raise ValueError(
            f"auto_merge can only be `both`, `none`, `before` or `after`, got {auto_merge}"
        )
    if bloom_filter not in [True, False, "auto"]:
        raise ValueError(
            f'bloom_filter can only be True, False, or "auto", got {bloom_filter}'
        )
    if bloom_filter_options:
        if not isinstance(bloom_filter_options, dict):
            raise TypeError(
                f"bloom_filter_options must be a dict, got {type(bloom_filter_options)}"
            )
        for k, v in bloom_filter_options.items():
            if k not in BLOOM_FILTER_OPTIONS:
                raise ValueError(
                    f"Invalid bloom filter option {k}, available: {BLOOM_FILTER_OPTIONS}"
                )
            if k == "filter" and v not in BLOOM_FILTER_ON_OPTIONS:
                raise ValueError(
                    f"Invalid filter {k}, available: {BLOOM_FILTER_ON_OPTIONS}"
                )

    if left_hint:
        if not isinstance(left_hint, JoinHint):
            raise TypeError(f"left_hint must be a JoinHint, got {type(left_hint)}")
        left_hint.verify_can_work_with(right_hint)
        left_hint.verify_params(df, on or left_on, left_index, how, True)

    if right_hint:
        if not isinstance(right_hint, JoinHint):
            raise TypeError(f"right_hint must be a JoinHint, got {type(right_hint)}")
        right_hint.verify_params(right, on or right_on, right_index, how, False)

    op = DataFrameMerge(
        how=how,
        on=on,
        left_on=left_on,
        right_on=right_on,
        left_index=left_index,
        right_index=right_index,
        sort=sort,
        suffixes=suffixes,
        copy=copy,
        indicator=indicator,
        validate=validate,
        method=method,
        auto_merge=auto_merge,
        auto_merge_threshold=auto_merge_threshold,
        bloom_filter=bloom_filter,
        bloom_filter_options=bloom_filter_options,
        output_types=[OutputType.dataframe],
        left_hint=left_hint,
        right_hint=right_hint,
    )
    return op(df, right)


def join(
    df: Union[DataFrame, Series],
    other: Union[DataFrame, Series],
    on: str = None,
    how: str = "left",
    lsuffix: str = "",
    rsuffix: str = "",
    sort: bool = False,
    method: str = None,
    auto_merge: str = "both",
    auto_merge_threshold: int = 8,
    bloom_filter: Union[bool, Dict] = True,
    bloom_filter_options: Dict[str, Any] = None,
    left_hint: JoinHint = None,
    right_hint: JoinHint = None,
) -> DataFrame:
    """
    Join columns of another DataFrame.

    Join columns with `other` DataFrame either on index or on a key
    column. Efficiently join multiple DataFrame objects by index at once by
    passing a list.

    Parameters
    ----------
    other : DataFrame, Series, or list of DataFrame
        Index should be similar to one of the columns in this one. If a
        Series is passed, its name attribute must be set, and that will be
        used as the column name in the resulting joined DataFrame.
    on : str, list of str, or array-like, optional
        Column or index level name(s) in the caller to join on the index
        in `other`, otherwise joins index-on-index. If multiple
        values given, the `other` DataFrame must have a MultiIndex. Can
        pass an array as the join key if it is not already contained in
        the calling DataFrame. Like an Excel VLOOKUP operation.
    how : {'left', 'right', 'outer', 'inner'}, default 'left'
        How to handle the operation of the two objects.

        * left: use calling frame's index (or column if on is specified)
        * right: use `other`'s index.
        * outer: form union of calling frame's index (or column if on is
          specified) with `other`'s index, and sort it.
          lexicographically.
        * inner: form intersection of calling frame's index (or column if
          on is specified) with `other`'s index, preserving the order
          of the calling's one.

    lsuffix : str, default ''
        Suffix to use from left frame's overlapping columns.
    rsuffix : str, default ''
        Suffix to use from right frame's overlapping columns.
    sort : bool, default False
        Order result DataFrame lexicographically by the join key. If False,
        the order of the join key depends on the join type (how keyword).
    method : {"shuffle", "broadcast"}, default None
        "broadcast" is recommended when one DataFrame is much smaller than the other,
        otherwise, "shuffle" will be a better choice. By default, we choose method
        according to actual data size.
    auto_merge : {"both", "none", "before", "after"}, default both
        Auto merge small chunks before or after merge

        * "both": auto merge small chunks before and after,
        * "none": do not merge small chunks
        * "before": only merge small chunks before merge
        * "after": only merge small chunks after merge
    auto_merge_threshold : int, default 8
        When how is "inner", merged result could be much smaller than original DataFrame,
        if the number of chunks is greater than the threshold,
        it will merge small chunks automatically.
    bloom_filter: bool, str, default "auto"
        Use bloom filter to optimize merge
    bloom_filter_options: dict
        * "max_elements": max elements in bloom filter,
          default value is the max size of all input chunks
        * "error_rate": error raite, default 0.1.
        * "apply_chunk_size_threshold": min chunk size of input chunks to apply bloom filter, default 10
          when chunk size of left and right is greater than this threshold, apply bloom filter
        * "filter": "large", "small", "both", default "large"
          decides to filter on large, small or both DataFrames.
    left_hint: JoinHint, default None
        Join strategy to use for left frame. When data skew occurs, consider these strategies to avoid long-tail issues,
        but use them cautiously to prevent OOM and unnecessary overhead.
    right_hint: JoinHint, default None
        Join strategy to use for right frame.

    Returns
    -------
    DataFrame
        A dataframe containing columns from both the caller and `other`.

    See Also
    --------
    DataFrame.merge : For column(s)-on-column(s) operations.

    Examples
    --------
    >>> import maxframe.dataframe as md
    >>> df = md.DataFrame({'key': ['K0', 'K1', 'K2', 'K3', 'K4', 'K5'],
    ...                    'A': ['A0', 'A1', 'A2', 'A3', 'A4', 'A5']})

    >>> df.execute()
      key   A
    0  K0  A0
    1  K1  A1
    2  K2  A2
    3  K3  A3
    4  K4  A4
    5  K5  A5

    >>> other = md.DataFrame({'key': ['K0', 'K1', 'K2'],
    ...                       'B': ['B0', 'B1', 'B2']})

    >>> other.execute()
      key   B
    0  K0  B0
    1  K1  B1
    2  K2  B2

    Join DataFrames using their indexes.

    >>> df.join(other, lsuffix='_caller', rsuffix='_other').execute()
      key_caller   A key_other    B
    0         K0  A0        K0   B0
    1         K1  A1        K1   B1
    2         K2  A2        K2   B2
    3         K3  A3       NaN  NaN
    4         K4  A4       NaN  NaN
    5         K5  A5       NaN  NaN

    If we want to join using the key columns, we need to set key to be
    the index in both `df` and `other`. The joined DataFrame will have
    key as its index.

    >>> df.set_index('key').join(other.set_index('key')).execute()
          A    B
    key
    K0   A0   B0
    K1   A1   B1
    K2   A2   B2
    K3   A3  NaN
    K4   A4  NaN
    K5   A5  NaN

    Another option to join using the key columns is to use the `on`
    parameter. DataFrame.join always uses `other`'s index but we can use
    any column in `df`. This method preserves the original DataFrame's
    index in the result.

    >>> df.join(other.set_index('key'), on='key').execute()
      key   A    B
    0  K0  A0   B0
    1  K1  A1   B1
    2  K2  A2   B2
    3  K3  A3  NaN
    4  K4  A4  NaN
    5  K5  A5  NaN

    Using non-unique key values shows how they are matched.

    >>> df = md.DataFrame({'key': ['K0', 'K1', 'K1', 'K3', 'K0', 'K1'],
    ...                    'A': ['A0', 'A1', 'A2', 'A3', 'A4', 'A5']})

    >>> df.execute()
      key   A
    0  K0  A0
    1  K1  A1
    2  K1  A2
    3  K3  A3
    4  K0  A4
    5  K1  A5

    >>> df.join(other.set_index('key'), on='key').execute()
      key   A    B
    0  K0  A0   B0
    1  K1  A1   B1
    2  K1  A2   B1
    3  K3  A3  NaN
    4  K0  A4   B0
    5  K1  A5   B1
    """
    return merge(
        df,
        other,
        left_on=on,
        how=how,
        left_index=on is None,
        right_index=True,
        suffixes=(lsuffix, rsuffix),
        sort=sort,
        method=method,
        auto_merge=auto_merge,
        auto_merge_threshold=auto_merge_threshold,
        bloom_filter=bloom_filter,
        bloom_filter_options=bloom_filter_options,
        left_hint=left_hint,
        right_hint=right_hint,
    )
