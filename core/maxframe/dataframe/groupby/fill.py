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

import pandas as pd

from maxframe import opcodes
from maxframe.core import OutputType
from maxframe.dataframe.operators import DataFrameOperator, DataFrameOperatorMixin
from maxframe.dataframe.utils import find_input_of_groupby, parse_index
from maxframe.serialization.serializables import (
    AnyField,
    DictField,
    Int64Field,
    StringField,
)


class GroupByFill(DataFrameOperator, DataFrameOperatorMixin):
    _op_module_ = "dataframe.groupby"
    _legacy_name = "GroupByFillOperator"  # since v2.0.0

    value = AnyField("value", default=None)
    method = StringField("method", default=None)
    axis = AnyField("axis", default=0)
    limit = Int64Field("limit", default=None)
    downcast = DictField("downcast", default=None)

    def _calc_out_dtypes(self, in_groupby):
        mock_groupby = in_groupby.op.build_mock_groupby()
        func_name = getattr(self, "_func_name")

        if self.method in ("ffill", "pad"):
            func_name = "ffilll"
        elif self.method in ("bfill", "backfill"):
            func_name = "bfill"

        if func_name == "fillna":
            kw = {}
            if self.axis is not None:
                kw["axis"] = self.axis
            try:
                result_df = mock_groupby.fillna(
                    value=self.value, limit=self.limit, **kw
                )
            except AttributeError:
                result_df = mock_groupby.apply(
                    lambda d: d.fillna(value=self.value, limit=self.limit, **kw)
                )
        else:
            result_df = getattr(mock_groupby, func_name)(limit=self.limit)

        if isinstance(result_df, pd.DataFrame):
            self.output_types = [OutputType.dataframe]
            return result_df.dtypes
        else:
            self.output_types = [OutputType.series]
            return result_df.name, result_df.dtype

    def __call__(self, groupby):
        in_df = find_input_of_groupby(groupby)
        out_dtypes = self._calc_out_dtypes(groupby)

        kw = in_df.params.copy()
        kw["index_value"] = parse_index(pd.RangeIndex(-1), groupby.key)
        if self.output_types[0] == OutputType.dataframe:
            kw.update(
                dict(
                    columns_value=parse_index(out_dtypes.index, store_data=True),
                    dtypes=out_dtypes,
                    shape=(groupby.shape[0], len(out_dtypes)),
                )
            )
        else:
            name, dtype = out_dtypes
            kw.update(dtype=dtype, name=name, shape=(groupby.shape[0],))
        return self.new_tileable([groupby], **kw)


class GroupByFFill(GroupByFill):
    _op_type_ = opcodes.FILL_NA
    _func_name = "ffill"


class GroupByBFill(GroupByFill):
    _op_type = opcodes.FILL_NA
    _func_name = "bfill"


class GroupByFillNa(GroupByFill):
    _op_type = opcodes.FILL_NA
    _func_name = "fillna"


# keep for import compatibility
GroupByFillOperator = GroupByFill


def ffill(groupby, limit=None):
    """
    Forward fill the values.

    limit:  int, default None
            Limit number of values to fill

    return: Series or DataFrame
    """
    op = GroupByFFill(limit=limit)
    return op(groupby)


def bfill(groupby, limit=None):
    """
    Backward fill the values.

    limit:  int, default None
            Limit number of values to fill

    return: Series or DataFrame
    """
    op = GroupByBFill(limit=limit)
    return op(groupby)


def fillna(groupby, value=None, method=None, axis=None, limit=None, downcast=None):
    """
    Fill NA/NaN values using the specified method

    value:  scalar, dict, Series, or DataFrame
            Value to use to fill holes (e.g. 0), alternately a dict/Series/DataFrame
            of values specifying which value to use for each index (for a Series) or
            column (for a DataFrame). Values not in the dict/Series/DataFrame
            will not be filled. This value cannot be a list.
    method: {'backfill','bfill','ffill',None}, default None
    axis:   {0 or 'index', 1 or 'column'}
    limit:  int, default None
            If method is specified, this is the maximum number of consecutive
            NaN values to forward/backward fill
    downcast:   dict, default None
                A dict of item->dtype of what to downcast if possible,
                or the string ‘infer’ which will try to downcast to an appropriate equal type

    return: DataFrame or None
    """
    op = GroupByFillNa(
        value=value, method=method, axis=axis, limit=limit, downcast=downcast
    )
    return op(groupby)
