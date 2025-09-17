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

from typing import List, Union

import pandas as pd

from ... import opcodes
from ...core import ENTITY_TYPE, EntityData, OutputType
from ...serialization.serializables import (
    AnyField,
    BoolField,
    FieldTypes,
    ListField,
    StringField,
)
from ...utils import lazy_import
from ..core import DataFrame, Series
from ..operators import SERIES_TYPE, DataFrameOperator, DataFrameOperatorMixin
from ..utils import build_empty_df, build_empty_series, parse_index, validate_axis

cudf = lazy_import("cudf")


class DataFrameConcat(DataFrameOperator, DataFrameOperatorMixin):
    _op_type_ = opcodes.CONCATENATE

    axis = AnyField("axis", default=None)
    join = StringField("join", default=None)
    ignore_index = BoolField("ignore_index", default=None)
    keys = ListField("keys", default=None)
    levels = ListField("levels", default=None)
    names = ListField("names", default=None)
    verify_integrity = BoolField("verify_integrity", default=None)
    sort = BoolField("sort", default=None)
    copy_ = BoolField("copy", default=None)

    def __init__(self, copy=None, output_types=None, **kw):
        super().__init__(copy_=copy, _output_types=output_types, **kw)

    @property
    def level(self):
        return self.levels

    @property
    def name(self):
        return self.names

    @classmethod
    def _concat_index(cls, df_or_series_list: Union[List[DataFrame], List[Series]]):
        concat_index = None
        all_indexes_have_value = all(
            input.index_value.has_value() for input in df_or_series_list
        )

        def _concat(prev_index: pd.Index, cur_index: pd.Index):
            if prev_index is None:
                return cur_index

            if (
                all_indexes_have_value
                and isinstance(prev_index, pd.RangeIndex)
                and isinstance(cur_index, pd.RangeIndex)
            ):
                # handle RangeIndex that append may generate huge amount of data
                # e.g. pd.RangeIndex(10_000) and pd.RangeIndex(10_000)
                # will generate a Int64Index full of data
                # for details see GH#1647
                prev_stop = prev_index.start + prev_index.size * prev_index.step
                cur_start = cur_index.start
                if prev_stop == cur_start and prev_index.step == cur_index.step:
                    # continuous RangeIndex, still return RangeIndex
                    return prev_index.append(cur_index)
                else:
                    # otherwise, return an empty index
                    return pd.Index([], dtype=prev_index.dtype)
            elif isinstance(prev_index, pd.RangeIndex):
                return pd.Index([], prev_index.dtype).append(cur_index)
            elif isinstance(cur_index, pd.RangeIndex):
                return prev_index.append(pd.Index([], cur_index.dtype))
            return prev_index.append(cur_index)

        for input in df_or_series_list:
            concat_index = _concat(concat_index, input.index_value.to_pandas())

        return concat_index

    def _call_series(self, objs):
        if self.axis == 0:
            row_length = 0
            for series in objs:
                row_length += series.shape[0]
            if self.ignore_index:
                idx_length = 0 if pd.isna(row_length) else row_length
                index_value = parse_index(pd.RangeIndex(idx_length))
            else:
                index = self._concat_index(objs)
                index_value = parse_index(index, objs)
            obj_names = {obj.name for obj in objs}
            return self.new_series(
                objs,
                shape=(row_length,),
                dtype=objs[0].dtype,
                index_value=index_value,
                name=objs[0].name if len(obj_names) == 1 else None,
            )
        else:
            col_length = 0
            columns = []
            dtypes = dict()
            undefined_name = 0
            for series in objs:
                if series.name is None:
                    dtypes[undefined_name] = series.dtype
                    undefined_name += 1
                    columns.append(undefined_name)
                else:
                    dtypes[series.name] = series.dtype
                    columns.append(series.name)
                col_length += 1
            if self.ignore_index or undefined_name == len(objs):
                columns_value = parse_index(pd.RangeIndex(col_length))
            else:
                columns_value = parse_index(pd.Index(columns), store_data=True)

            shape = (objs[0].shape[0], col_length)
            return self.new_dataframe(
                objs,
                shape=shape,
                dtypes=pd.Series(dtypes),
                index_value=objs[0].index_value,
                columns_value=columns_value,
            )

    def _call_dataframes(self, objs):
        if self.axis == 0:
            row_length = 0
            empty_dfs = []
            for df in objs:
                row_length += df.shape[0]
                if df.ndim == 2:
                    empty_dfs.append(build_empty_df(df.dtypes))
                else:
                    empty_dfs.append(build_empty_series(df.dtype, name=df.name))

            emtpy_result = pd.concat(empty_dfs, join=self.join, sort=self.sort)
            shape = (row_length, emtpy_result.shape[1])
            columns_value = parse_index(emtpy_result.columns, store_data=True)

            if self.join == "inner":
                objs = [o[list(emtpy_result.columns)] for o in objs]

            if self.ignore_index:
                idx_length = 0 if pd.isna(row_length) else row_length
                index_value = parse_index(pd.RangeIndex(idx_length))
            else:
                index = self._concat_index(objs)
                index_value = parse_index(index, objs)

            new_objs = []
            for obj in objs:
                if obj.ndim != 2:
                    # series
                    new_obj = obj.to_frame().reindex(columns=emtpy_result.dtypes.index)
                else:
                    # dataframe
                    if list(obj.dtypes.index) != list(emtpy_result.dtypes.index):
                        new_obj = obj.reindex(columns=emtpy_result.dtypes.index)
                    else:
                        new_obj = obj
                new_objs.append(new_obj)

            return self.new_dataframe(
                new_objs,
                shape=shape,
                dtypes=emtpy_result.dtypes,
                index_value=index_value,
                columns_value=columns_value,
            )
        else:
            col_length = 0
            empty_dfs = []
            for df in objs:
                if df.ndim == 2:
                    # DataFrame
                    col_length += df.shape[1]
                    empty_dfs.append(build_empty_df(df.dtypes))
                else:
                    # Series
                    col_length += 1
                    empty_dfs.append(build_empty_series(df.dtype, name=df.name))

            emtpy_result = pd.concat(empty_dfs, join=self.join, axis=1, sort=True)
            if self.ignore_index:
                columns_value = parse_index(pd.RangeIndex(col_length))
            else:
                columns_value = parse_index(
                    pd.Index(emtpy_result.columns), store_data=True
                )

            if self.ignore_index or len({o.index_value.key for o in objs}) == 1:
                new_objs = [obj if obj.ndim == 2 else obj.to_frame() for obj in objs]
            else:  # pragma: no cover
                raise NotImplementedError(
                    "Does not support concat dataframes which has different index"
                )

            shape = (objs[0].shape[0], col_length)
            return self.new_dataframe(
                new_objs,
                shape=shape,
                dtypes=emtpy_result.dtypes,
                index_value=objs[0].index_value,
                columns_value=columns_value,
            )

    def __call__(self, objs):
        if all(isinstance(obj, SERIES_TYPE) for obj in objs):
            self.output_types = [OutputType.series]
            return self._call_series(objs)
        else:
            self.output_types = [OutputType.dataframe]
            return self._call_dataframes(objs)


class GroupByConcat(DataFrameOperator, DataFrameOperatorMixin):
    _op_type_ = opcodes.GROUPBY_CONCAT

    _groups = ListField("groups", FieldTypes.key)
    _groupby_params = AnyField("groupby_params")

    def __init__(self, groups=None, groupby_params=None, output_types=None, **kw):
        super().__init__(
            _groups=groups,
            _groupby_params=groupby_params,
            _output_types=output_types,
            **kw
        )

    @property
    def groups(self):
        return self._groups

    @property
    def groupby_params(self):
        return self._groupby_params

    @classmethod
    def _set_inputs(cls, op: "GroupByConcat", inputs: List[EntityData]):
        super()._set_inputs(op, inputs)
        inputs_iter = iter(op._inputs)

        new_groups = []
        for _ in op._groups:
            new_groups.append(next(inputs_iter))
        op._groups = new_groups

        if isinstance(op._groupby_params["by"], list):
            by = []
            for v in op._groupby_params["by"]:
                if isinstance(v, ENTITY_TYPE):
                    by.append(next(inputs_iter))
                else:
                    by.append(v)
            op._groupby_params["by"] = by


def concat(
    objs,
    axis=0,
    join="outer",
    ignore_index=False,
    keys=None,
    levels=None,
    names=None,
    verify_integrity=False,
    sort=False,
    copy=True,
):
    """
    Concatenate dataframe objects along a particular axis with optional set logic
    along the other axes.

    Can also add a layer of hierarchical indexing on the concatenation axis,
    which may be useful if the labels are the same (or overlapping) on
    the passed axis number.

    Parameters
    ----------
    objs : a sequence or mapping of Series or DataFrame objects
        If a mapping is passed, the sorted keys will be used as the `keys`
        argument, unless it is passed, in which case the values will be
        selected (see below). Any None objects will be dropped silently unless
        they are all None in which case a ValueError will be raised.
    axis : {0/'index', 1/'columns'}, default 0
        The axis to concatenate along.
    join : {'inner', 'outer'}, default 'outer'
        How to handle indexes on other axis (or axes).
    ignore_index : bool, default False
        If True, do not use the index values along the concatenation axis. The
        resulting axis will be labeled 0, ..., n - 1. This is useful if you are
        concatenating objects where the concatenation axis does not have
        meaningful indexing information. Note the index values on the other
        axes are still respected in the join.
    keys : sequence, default None
        If multiple levels passed, should contain tuples. Construct
        hierarchical index using the passed keys as the outermost level.
    levels : list of sequences, default None
        Specific levels (unique values) to use for constructing a
        MultiIndex. Otherwise they will be inferred from the keys.
    names : list, default None
        Names for the levels in the resulting hierarchical index.
    verify_integrity : bool, default False
        Check whether the new concatenated axis contains duplicates. This can
        be very expensive relative to the actual data concatenation.
    sort : bool, default False
        Sort non-concatenation axis if it is not already aligned when `join`
        is 'outer'.
        This has no effect when ``join='inner'``, which already preserves
        the order of the non-concatenation axis.
    copy : bool, default True
        If False, do not copy data unnecessarily.

    Returns
    -------
    object, type of objs
        When concatenating all ``Series`` along the index (axis=0), a
        ``Series`` is returned. When ``objs`` contains at least one
        ``DataFrame``, a ``DataFrame`` is returned. When concatenating along
        the columns (axis=1), a ``DataFrame`` is returned.

    See Also
    --------
    Series.append : Concatenate Series.
    DataFrame.append : Concatenate DataFrames.
    DataFrame.join : Join DataFrames using indexes.
    DataFrame.merge : Merge DataFrames by indexes or columns.

    Notes
    -----
    The keys, levels, and names arguments are all optional.

    A walkthrough of how this method fits in with other tools for combining
    pandas objects can be found `here
    <https://pandas.pydata.org/pandas-docs/stable/user_guide/merging.html>`__.

    Examples
    --------
    Combine two ``Series``.

    >>> import maxframe.dataframe as md
    >>> s1 = md.Series(['a', 'b'])
    >>> s2 = md.Series(['c', 'd'])
    >>> md.concat([s1, s2]).execute()
    0    a
    1    b
    0    c
    1    d
    dtype: object

    Clear the existing index and reset it in the result
    by setting the ``ignore_index`` option to ``True``.

    >>> md.concat([s1, s2], ignore_index=True).execute()
    0    a
    1    b
    2    c
    3    d
    dtype: object

    Add a hierarchical index at the outermost level of
    the data with the ``keys`` option.

    >>> md.concat([s1, s2], keys=['s1', 's2']).execute()
    s1  0    a
        1    b
    s2  0    c
        1    d
    dtype: object

    Label the index keys you create with the ``names`` option.

    >>> md.concat([s1, s2], keys=['s1', 's2'],
    ...           names=['Series name', 'Row ID']).execute()
    Series name  Row ID
    s1           0         a
                 1         b
    s2           0         c
                 1         d
    dtype: object

    Combine two ``DataFrame`` objects with identical columns.

    >>> df1 = md.DataFrame([['a', 1], ['b', 2]],
    ...                    columns=['letter', 'number'])
    >>> df1.execute()
      letter  number
    0      a       1
    1      b       2
    >>> df2 = md.DataFrame([['c', 3], ['d', 4]],
    ...                    columns=['letter', 'number'])
    >>> df2.execute()
      letter  number
    0      c       3
    1      d       4
    >>> md.concat([df1, df2]).execute()
      letter  number
    0      a       1
    1      b       2
    0      c       3
    1      d       4

    Combine ``DataFrame`` objects with overlapping columns
    and return everything. Columns outside the intersection will
    be filled with ``NaN`` values.

    >>> df3 = md.DataFrame([['c', 3, 'cat'], ['d', 4, 'dog']],
    ...                    columns=['letter', 'number', 'animal'])
    >>> df3.execute()
      letter  number animal
    0      c       3    cat
    1      d       4    dog
    >>> md.concat([df1, df3], sort=False).execute()
      letter  number animal
    0      a       1    NaN
    1      b       2    NaN
    0      c       3    cat
    1      d       4    dog

    Combine ``DataFrame`` objects with overlapping columns
    and return only those that are shared by passing ``inner`` to
    the ``join`` keyword argument.

    >>> md.concat([df1, df3], join="inner").execute()
      letter  number
    0      a       1
    1      b       2
    0      c       3
    1      d       4

    Combine ``DataFrame`` objects horizontally along the x axis by
    passing in ``axis=1``.

    >>> df4 = md.DataFrame([['bird', 'polly'], ['monkey', 'george']],
    ...                    columns=['animal', 'name'])
    >>> md.concat([df1, df4], axis=1).execute()
      letter  number  animal    name
    0      a       1    bird   polly
    1      b       2  monkey  george

    Prevent the result from including duplicate index values with the
    ``verify_integrity`` option.

    >>> df5 = md.DataFrame([1], index=['a'])
    >>> df5.execute()
       0
    a  1
    >>> df6 = md.DataFrame([2], index=['a'])
    >>> df6.execute()
       0
    a  2
    """
    if not isinstance(objs, (list, tuple)):  # pragma: no cover
        raise TypeError(
            "first argument must be an iterable of dataframe or series objects"
        )
    axis = validate_axis(axis)
    if isinstance(objs, dict):  # pragma: no cover
        keys = objs.keys()
        objs = objs.values()
    if axis == 1 and join == "inner":  # pragma: no cover
        raise NotImplementedError("inner join is not support when specify `axis=1`")
    if verify_integrity or sort or keys:  # pragma: no cover
        raise NotImplementedError(
            "verify_integrity, sort, keys arguments are not supported now"
        )
    op = DataFrameConcat(
        axis=axis,
        join=join,
        ignore_index=ignore_index,
        keys=keys,
        levels=levels,
        names=names,
        verify_integrity=verify_integrity,
        sort=sort,
        copy=copy,
    )

    return op(objs)
