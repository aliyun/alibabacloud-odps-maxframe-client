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


class CountReductionCallable(ReductionCallable):
    def __call__(self, value):
        skipna, numeric_only = self.kwargs["skipna"], self.kwargs["numeric_only"]
        if value.ndim == 1:
            return value.count()
        return value.count(skipna=skipna, numeric_only=numeric_only)


class DataFrameCount(DataFrameReduction, DataFrameReductionMixin):
    _op_type_ = opcodes.COUNT
    _func_name = "count"

    @property
    def is_atomic(self):
        return True

    @classmethod
    def get_reduction_callable(cls, op):
        skipna, numeric_only = op.skipna, op.numeric_only
        return CountReductionCallable(
            func_name="count", kwargs={"skipna": skipna, "numeric_only": numeric_only}
        )


def count_series(series, level=None, **kw):
    method = kw.pop("method", None)
    op = DataFrameCount(
        level=level,
        output_types=[OutputType.scalar],
        method=method,
    )
    return op(series)


def count_dataframe(df, axis=0, level=None, numeric_only=False, **kw):
    method = kw.pop("method", None)
    op = DataFrameCount(
        axis=axis,
        level=level,
        numeric_only=numeric_only,
        output_types=[OutputType.series],
        method=method,
    )
    return op(df)
