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

from ... import opcodes
from ...core import ENTITY_TYPE, EntityData, get_output_types
from ...serialization.serializables import AnyField, Int8Field, KeyField
from ..operators import DataFrameOperator, DataFrameOperatorMixin
from ..utils import parse_index, validate_axis


class DataFrameSetAxis(DataFrameOperator, DataFrameOperatorMixin):
    _op_type_ = opcodes.DATAFRAME_SET_AXIS

    input = KeyField("input", default=None)
    axis = Int8Field("axis", default=None)
    value = AnyField("value", default=None)

    @classmethod
    def _set_inputs(cls, op: "DataFrameSetAxis", inputs: List[EntityData]):
        super()._set_inputs(op, inputs)
        op.input = inputs[0]
        if isinstance(op.value, ENTITY_TYPE):
            op.value = inputs[-1]

    def __call__(self, df_or_series):
        new_size = self.value.shape[0]
        expect_size = df_or_series.axes[self.axis].shape[0]
        if (
            not np.isnan(new_size)
            and not np.isnan(expect_size)
            and new_size != expect_size
        ):
            raise ValueError(
                f"Length mismatch: Expected axis has {expect_size} elements, "
                f"new values have {new_size} elements"
            )

        params = df_or_series.params
        if self.axis == 0:
            params["index_value"] = (
                parse_index(self.value)
                if isinstance(self.value, pd.Index)
                else self.value.index_value
            )
        else:
            params["columns_value"] = (
                parse_index(self.value, store_data=True)
                if isinstance(self.value, pd.Index)
                else self.value.index_value
            )
            pd_columns = (
                self.value.index_value.to_pandas()
                if isinstance(self.value, ENTITY_TYPE)
                else self.value
            )
            params["dtypes"] = params["dtypes"].set_axis(pd_columns)

        self._output_types = get_output_types(df_or_series)
        inputs = [df_or_series]
        if isinstance(self.value, ENTITY_TYPE):
            inputs += [self.value]
        return self.new_tileable(inputs, **params)


def _set_axis(df_or_axis, labels, axis=0, inplace=False):
    axis = validate_axis(axis, df_or_axis)
    if not isinstance(labels, ENTITY_TYPE) and not isinstance(labels, pd.Index):
        labels = pd.Index(labels)

    op = DataFrameSetAxis(value=labels, axis=axis)
    result = op(df_or_axis)
    if inplace:
        df_or_axis.data = result.data
    else:
        return result


def df_set_axis(df, labels, axis=0, inplace=False):
    """
    Assign desired index to given axis.

    Indexes for column or row labels can be changed by assigning
    a list-like or Index.

    Parameters
    ----------
    labels : list-like, Index
        The values for the new index.

    axis : {0 or 'index', 1 or 'columns'}, default 0
        The axis to update. The value 0 identifies the rows, and 1 identifies the columns.

    inplace : bool, default False
        Whether to return a new DataFrame instance.

    Returns
    -------
    renamed : DataFrame or None
        An object of type DataFrame or None if ``inplace=True``.

    See Also
    --------
    DataFrame.rename_axis : Alter the name of the index or columns.

    Examples
    --------
    >>> import maxframe.dataframe as md
    >>> df = md.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})

    Change the row labels.

    >>> df.set_axis(['a', 'b', 'c'], axis='index').execute()
       A  B
    a  1  4
    b  2  5
    c  3  6

    Change the column labels.

    >>> df.set_axis(['I', 'II'], axis='columns').execute()
       I  II
    0  1   4
    1  2   5
    2  3   6

    Now, update the labels inplace.

    >>> df.set_axis(['i', 'ii'], axis='columns', inplace=True)
    >>> df.execute()
       i  ii
    0  1   4
    1  2   5
    2  3   6
    """
    return _set_axis(df, labels, axis=axis, inplace=inplace)


def series_set_axis(series, labels, axis=0, inplace=False):
    """
    Assign desired index to given axis.

    Indexes for row labels can be changed by assigning
    a list-like or Index.

    Parameters
    ----------
    labels : list-like, Index
        The values for the new index.

    axis : {0 or 'index'}, default 0
        The axis to update. The value 0 identifies the rows.

    inplace : bool, default False
        Whether to return a new Series instance.

    Returns
    -------
    renamed : Series or None
        An object of type Series or None if ``inplace=True``.

    See Also
    --------
    Series.rename_axis : Alter the name of the index.

    Examples
    --------
    >>> import maxframe.dataframe as md
    >>> s = md.Series([1, 2, 3])
    >>> s.execute()
    0    1
    1    2
    2    3
    dtype: int64

    >>> s.set_axis(['a', 'b', 'c'], axis=0).execute()
    a    1
    b    2
    c    3
    dtype: int64
    """
    return _set_axis(series, labels, axis=axis, inplace=inplace)
