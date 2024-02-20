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

import numpy as np
import pandas as pd

from ... import opcodes as OperandDef
from ...serialization.serializables import AnyField, FieldTypes, KeyField, ListField
from ..core import SERIES_TYPE
from ..operators import DataFrameOperator, DataFrameOperatorMixin
from ..utils import build_empty_df, parse_index


class DataFrameDescribe(DataFrameOperator, DataFrameOperatorMixin):
    _op_type_ = OperandDef.DESCRIBE

    input = KeyField("input", default=None)
    percentiles = ListField("percentiles", FieldTypes.float64, default=None)
    include = AnyField("include", default=None)
    exclude = AnyField("exclude", default=None)

    def __init__(self, output_types=None, **kw):
        super().__init__(_output_types=output_types, **kw)

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        self.input = self._inputs[0]

    def __call__(self, df_or_series):
        if isinstance(df_or_series, SERIES_TYPE):
            if not np.issubdtype(df_or_series.dtype, np.number):
                raise NotImplementedError("non-numeric type is not supported for now")
            test_series = pd.Series([], dtype=df_or_series.dtype).describe(
                percentiles=self.percentiles,
                include=self.include,
                exclude=self.exclude,
            )
            return self.new_series(
                [df_or_series],
                shape=(len(test_series),),
                dtype=test_series.dtype,
                index_value=parse_index(test_series.index, store_data=True),
            )
        else:
            test_inp_df = build_empty_df(df_or_series.dtypes)
            test_df = test_inp_df.describe(
                percentiles=self.percentiles,
                include=self.include,
                exclude=self.exclude,
            )
            if len(self.percentiles) == 0:
                # specify percentiles=False
                # Note: unlike pandas that False is illegal value for percentiles,
                # MaxFrame DataFrame allows user to specify percentiles=False
                # to skip computation about percentiles
                test_df.drop(["50%"], axis=0, inplace=True)
            for dtype in test_df.dtypes:
                if not np.issubdtype(dtype, np.number):
                    raise NotImplementedError(
                        "non-numeric type is not supported for now"
                    )
            return self.new_dataframe(
                [df_or_series],
                shape=test_df.shape,
                dtypes=test_df.dtypes,
                index_value=parse_index(test_df.index, store_data=True),
                columns_value=parse_index(test_df.columns, store_data=True),
            )


def describe(df_or_series, percentiles=None, include=None, exclude=None):
    if percentiles is False:
        percentiles = []
    elif percentiles is None:
        percentiles = [0.25, 0.5, 0.75]
    else:
        percentiles = list(percentiles)
        if percentiles is not None:
            for p in percentiles:
                if p < 0 or p > 1:
                    raise ValueError(
                        "percentiles should all be in the interval [0, 1]. "
                        "Try [{0:.3f}] instead.".format(p / 100)
                    )
        # median should always be included
        if 0.5 not in percentiles:
            percentiles.append(0.5)
        percentiles = np.asarray(percentiles)

        # sort and check for duplicates
        unique_pcts = np.unique(percentiles)
        if len(unique_pcts) < len(percentiles):
            raise ValueError("percentiles cannot contain duplicates")
        percentiles = unique_pcts.tolist()

    op = DataFrameDescribe(percentiles=percentiles, include=include, exclude=exclude)
    return op(df_or_series)
