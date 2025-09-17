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

import numpy as np

from ... import opcodes
from ...core import ENTITY_TYPE, OutputType
from ...serialization.serializables import BoolField
from .core import DataFrameReduction, DataFrameReductionMixin, ReductionCallable


class KurtosisReductionCallable(ReductionCallable):
    def __call__(self, value):
        from .aggregation import where_function

        skipna = self.kwargs["skipna"]
        bias = self.kwargs["bias"]
        fisher = self.kwargs["fisher"]

        cnt = value.count()
        mean = value.mean(skipna=skipna)
        divided = (
            (value**4).mean(skipna=skipna)
            - 4 * (value**3).mean(skipna=skipna) * mean
            + 6 * (value**2).mean(skipna=skipna) * mean**2
            - 3 * mean**4
        )
        var = value.var(skipna=skipna, ddof=0)
        if isinstance(var, ENTITY_TYPE) or var > 0:
            val = where_function(var > 0, divided / var**2, np.nan)
        else:
            val = np.nan
        if not bias:
            val = where_function(
                (var > 0) & (cnt > 3),
                (val * (cnt**2 - 1) - 3 * (cnt - 1) ** 2) / (cnt - 2) / (cnt - 3),
                np.nan,
            )
        if not fisher:
            val += 3
        return val


class DataFrameKurtosis(DataFrameReduction, DataFrameReductionMixin):
    _op_type_ = opcodes.KURTOSIS
    _func_name = "kurt"

    bias = BoolField("bias", default=None)
    fisher = BoolField("fisher", default=None)

    @classmethod
    def get_reduction_callable(cls, op):
        skipna, bias, fisher = op.skipna, op.bias, op.fisher
        return KurtosisReductionCallable(
            func_name="kurt", kwargs=dict(skipna=skipna, bias=bias, fisher=fisher)
        )


def kurt_series(
    df,
    axis=None,
    skipna=True,
    level=None,
    bias=False,
    fisher=True,
    method=None,
):
    op = DataFrameKurtosis(
        axis=axis,
        skipna=skipna,
        level=level,
        bias=bias,
        fisher=fisher,
        output_types=[OutputType.scalar],
        method=method,
    )
    return op(df)


def kurt_dataframe(
    df,
    axis=None,
    skipna=True,
    level=None,
    numeric_only=None,
    bias=False,
    fisher=True,
    method=None,
):
    op = DataFrameKurtosis(
        axis=axis,
        skipna=skipna,
        level=level,
        numeric_only=numeric_only,
        bias=bias,
        fisher=fisher,
        output_types=[OutputType.series],
        method=method,
    )
    return op(df)
