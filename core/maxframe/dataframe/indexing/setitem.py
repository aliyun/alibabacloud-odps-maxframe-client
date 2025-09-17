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
from ...tensor.core import TENSOR_TYPE
from ...utils import pd_release_version
from ..core import DATAFRAME_TYPE, ENTITY_TYPE, SERIES_TYPE, DataFrame
from ..initializer import DataFrame as asframe
from ..initializer import Series as asseries
from ..operators import DataFrameOperator, DataFrameOperatorMixin
from ..utils import parse_index

# in pandas 1.0.x, __setitem__ with a list with missing items are not allowed
_allow_set_missing_list = pd_release_version[:2] >= (1, 1)


class DataFrameSetitem(DataFrameOperator, DataFrameOperatorMixin):
    _op_type_ = opcodes.INDEXSETVALUE

    target = KeyField("target")
    indexes = AnyField("indexes", default=None)
    value = AnyField("value", default=None)

    def __init__(self, output_types=None, **kw):
        super().__init__(_output_types=output_types, **kw)
        if self.output_types is None:
            self.output_types = [OutputType.dataframe]

    @classmethod
    def _set_inputs(cls, op: "DataFrameSetitem", inputs: List[EntityData]):
        super()._set_inputs(op, inputs)
        op.target = op._inputs[0]
        if len(inputs) > 1:
            op.value = op._inputs[-1]

    @staticmethod
    def _is_scalar_tensor(t):
        return isinstance(t, TENSOR_TYPE) and t.ndim == 0

    def __call__(self, target: DataFrame, value):
        raw_target = target

        inputs = [target]
        if np.isscalar(value):
            value_dtype = np.array(value).dtype
        elif self._is_scalar_tensor(value):
            inputs.append(value)
            value_dtype = value.dtype
        else:
            if isinstance(value, (pd.Series, SERIES_TYPE)):
                value = asseries(value)
                value_dtype = value.dtype
            elif isinstance(value, (pd.DataFrame, DATAFRAME_TYPE)):
                if len(self.indexes) != value.shape[1]:  # pragma: no cover
                    raise ValueError("Columns must be same length as key")

                value = asframe(value)
                value_dtype = pd.Series(list(value.dtypes), index=self.indexes)
            elif is_list_like(value) or isinstance(value, TENSOR_TYPE):
                # convert to numpy to get actual dim and shape
                if is_list_like(value) and not isinstance(value, ENTITY_TYPE):
                    value = np.array(value)

                if value.ndim == 1:
                    value = asseries(value, index=target.index)
                    value_dtype = value.dtype
                else:
                    if len(self.indexes) != value.shape[1]:  # pragma: no cover
                        raise ValueError("Columns must be same length as key")

                    value = asframe(value, index=target.index)
                    value_dtype = pd.Series(list(value.dtypes), index=self.indexes)
            else:  # pragma: no cover
                raise TypeError(
                    "Wrong value type, could be one of scalar, Series or tensor"
                )

            if target.shape[0] == 0:
                # target empty, reindex target first
                target = target.reindex(value.index)
                inputs[0] = target
            elif value.index_value.key != target.index_value.key:
                # need reindex when target df is not empty and index different
                value = value.reindex(target.index)
            inputs.append(value)

        index_value = target.index_value
        dtypes = target.dtypes.copy(deep=True)

        try:
            dtypes.loc[self.indexes] = value_dtype
        except (AttributeError, KeyError):
            # when some index not exist, try update one by one
            if isinstance(value_dtype, pd.Series):
                for idx in self.indexes:
                    dtypes.loc[idx] = value_dtype.loc[idx]
            else:
                for idx in self.indexes:
                    dtypes.loc[idx] = value_dtype

        columns_value = parse_index(dtypes.index, store_data=True)
        ret = self.new_dataframe(
            inputs,
            shape=(target.shape[0], len(dtypes)),
            dtypes=dtypes,
            index_value=index_value,
            columns_value=columns_value,
        )
        raw_target.data = ret.data


def dataframe_setitem(df, col, value):
    op = DataFrameSetitem(target=df, indexes=col, value=value)
    return op(df, value)
