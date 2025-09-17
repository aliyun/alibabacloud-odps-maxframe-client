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
from .core import DataFrameReduction, DataFrameReductionMixin, ReductionCallable


class MeanReductionCallable(ReductionCallable):
    def __call__(self, value):
        skipna = self.kwargs["skipna"]
        return value.sum(skipna=skipna) / value.count()


class DataFrameMean(DataFrameReduction, DataFrameReductionMixin):
    _op_type_ = opcodes.MEAN
    _func_name = "mean"

    @classmethod
    def get_reduction_callable(cls, op):
        skipna = op.skipna
        return MeanReductionCallable(func_name="mean", kwargs=dict(skipna=skipna))


def mean_series(df, axis=None, skipna=True, level=None, method=None):
    op = DataFrameMean(
        axis=axis,
        skipna=skipna,
        level=level,
        output_types=[OutputType.scalar],
        method=method,
    )
    return op(df)


def mean_dataframe(
    df,
    axis=None,
    skipna=True,
    level=None,
    numeric_only=None,
    method=None,
):
    op = DataFrameMean(
        axis=axis,
        skipna=skipna,
        level=level,
        numeric_only=numeric_only,
        output_types=[OutputType.series],
        method=method,
    )
    return op(df)
