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
from .core import DataFrameReductionMixin, DataFrameReductionOperator


class DataFrameMax(DataFrameReductionOperator, DataFrameReductionMixin):
    _op_type_ = opcodes.MAX
    _func_name = "max"

    @property
    def is_atomic(self):
        return True


def max_series(df, axis=None, skipna=True, level=None, method=None):
    op = DataFrameMax(
        axis=axis,
        skipna=skipna,
        level=level,
        output_types=[OutputType.scalar],
        method=method,
    )
    return op(df)


def max_dataframe(
    df,
    axis=None,
    skipna=True,
    level=None,
    numeric_only=None,
    method=None,
):
    op = DataFrameMax(
        axis=axis,
        skipna=skipna,
        level=level,
        numeric_only=numeric_only,
        output_types=[OutputType.series],
        method=method,
    )
    return op(df)


def max_index(df, axis=None, skipna=True):
    op = DataFrameMax(
        axis=axis,
        skipna=skipna,
        output_types=[OutputType.scalar],
    )
    return op(df)
