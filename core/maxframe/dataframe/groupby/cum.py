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

import pandas as pd

from ... import opcodes
from ...core import OutputType
from ...serialization.serializables import AnyField, BoolField
from ...utils import lazy_import
from ..operators import DataFrameOperator, DataFrameOperatorMixin
from ..utils import parse_index, validate_axis

cudf = lazy_import("cudf")


class GroupByCumReductionOperator(DataFrameOperatorMixin, DataFrameOperator):
    """
    NOTE: this operator has been deprecated and merged with GroupByExpandingAgg.
    """

    _op_module_ = "dataframe.groupby"

    axis = AnyField("axis", default=None)
    ascending = BoolField("ascending", default=None)

    def __init__(self, output_types=None, **kw):
        super().__init__(_output_types=output_types, **kw)

    def _calc_out_dtypes(self, in_groupby):
        mock_groupby = in_groupby.op.build_mock_groupby()
        func_name = getattr(self, "_func_name")

        if func_name == "cumcount":
            result_df = mock_groupby.cumcount(ascending=self.ascending)
        else:
            result_df = getattr(mock_groupby, func_name)(axis=self.axis)

        if isinstance(result_df, pd.DataFrame):
            self.output_types = [OutputType.dataframe]
            return result_df.dtypes
        else:
            self.output_types = [OutputType.series]
            return result_df.name, result_df.dtype

    def __call__(self, groupby):
        in_df = groupby
        while in_df.op.output_types[0] not in (OutputType.dataframe, OutputType.series):
            in_df = in_df.inputs[0]

        self.axis = validate_axis(self.axis or 0, in_df)

        out_dtypes = self._calc_out_dtypes(groupby)

        kw = in_df.params.copy()
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


class GroupByCummin(GroupByCumReductionOperator):
    _op_type_ = opcodes.CUMMIN
    _func_name = "cummin"


class GroupByCummax(GroupByCumReductionOperator):
    _op_type_ = opcodes.CUMMAX
    _func_name = "cummax"


class GroupByCumsum(GroupByCumReductionOperator):
    _op_type_ = opcodes.CUMSUM
    _func_name = "cumsum"


class GroupByCumprod(GroupByCumReductionOperator):
    _op_type_ = opcodes.CUMPROD
    _func_name = "cumprod"


class GroupByCumcount(GroupByCumReductionOperator):
    _op_type_ = opcodes.CUMCOUNT
    _func_name = "cumcount"
