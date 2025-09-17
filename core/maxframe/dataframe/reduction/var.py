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

from ... import opcodes
from ...core import OutputType
from ...serialization.serializables import Int32Field
from .core import DataFrameReduction, DataFrameReductionMixin, ReductionCallable


class VarReductionCallable(ReductionCallable):
    def __call__(self, value):
        skipna, ddof = self.kwargs["skipna"], self.kwargs["ddof"]
        cnt = value.count()
        if ddof == 0:
            return (value**2).mean(skipna=skipna) - (value.mean(skipna=skipna)) ** 2
        return (
            (value**2).sum(skipna=skipna) - value.sum(skipna=skipna) ** 2 / cnt
        ) / (cnt - ddof)


class DataFrameVar(DataFrameReduction, DataFrameReductionMixin):
    _op_type_ = opcodes.VAR
    _func_name = "var"

    ddof = Int32Field("ddof", default=None)

    @classmethod
    def get_reduction_callable(cls, op: "DataFrameVar"):
        skipna, ddof = op.skipna, op.ddof
        return VarReductionCallable(
            func_name="var", kwargs={"skipna": skipna, "ddof": ddof}
        )


def var_series(series, axis=None, skipna=True, level=None, ddof=1, method=None):
    op = DataFrameVar(
        axis=axis,
        skipna=skipna,
        level=level,
        ddof=ddof,
        output_types=[OutputType.scalar],
        method=method,
    )
    return op(series)


def var_dataframe(
    df,
    axis=None,
    skipna=True,
    level=None,
    ddof=1,
    numeric_only=None,
    method=None,
):
    op = DataFrameVar(
        axis=axis,
        skipna=skipna,
        level=level,
        ddof=ddof,
        numeric_only=numeric_only,
        output_types=[OutputType.series],
        method=method,
    )
    return op(df)
