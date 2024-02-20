# Copyright 1999-2024 Alibaba Group Holding Ltd.
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

import pandas as pd

from ... import opcodes
from ...core import ENTITY_TYPE, OutputType
from ...serialization.serializables import (
    AnyField,
    BoolField,
    FieldTypes,
    ListField,
    StringField,
)
from ...utils import lazy_import
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
    def _concat_index(cls, prev_index: pd.Index, cur_index: pd.Index):
        if isinstance(prev_index, pd.RangeIndex) and isinstance(
            cur_index, pd.RangeIndex
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

    def _call_series(self, objs):
        if self.axis == 0:
            row_length = 0
            index = None
            for series in objs:
                if index is None:
                    index = series.index_value.to_pandas()
                else:
                    index = self._concat_index(index, series.index_value.to_pandas())
                row_length += series.shape[0]
            if self.ignore_index:  # pragma: no cover
                index_value = parse_index(pd.RangeIndex(row_length))
            else:
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
            index = None
            empty_dfs = []
            for df in objs:
                if index is None:
                    index = df.index_value.to_pandas()
                else:
                    index = self._concat_index(index, df.index_value.to_pandas())
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

            if self.ignore_index:  # pragma: no cover
                index_value = parse_index(pd.RangeIndex(row_length))
            else:
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

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        inputs_iter = iter(self._inputs)

        new_groups = []
        for _ in self._groups:
            new_groups.append(next(inputs_iter))
        self._groups = new_groups

        if isinstance(self._groupby_params["by"], list):
            by = []
            for v in self._groupby_params["by"]:
                if isinstance(v, ENTITY_TYPE):
                    by.append(next(inputs_iter))
                else:
                    by.append(v)
            self._groupby_params["by"] = by

    @classmethod
    def execute(cls, ctx, op):
        input_data = [ctx[input.key] for input in op.groups]
        obj = pd.concat([d.obj for d in input_data])

        params = op.groupby_params.copy()
        if isinstance(params["by"], list):
            by = []
            for v in params["by"]:
                if isinstance(v, ENTITY_TYPE):
                    by.append(ctx[v.key])
                else:
                    by.append(v)
            params["by"] = by
        selection = params.pop("selection", None)

        result = obj.groupby(**params)
        if selection:
            result = result[selection]

        ctx[op.outputs[0].key] = result


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
