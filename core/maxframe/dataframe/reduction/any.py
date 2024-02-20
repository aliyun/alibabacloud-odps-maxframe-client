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

from ... import opcodes
from ...core import OutputType
from .core import DATAFRAME_TYPE, DataFrameReductionMixin, DataFrameReductionOperator


class DataFrameAny(DataFrameReductionOperator, DataFrameReductionMixin):
    _op_type_ = opcodes.ANY
    _func_name = "any"

    @property
    def is_atomic(self):
        return True

    def __call__(self, df):
        if self.axis is None and isinstance(df, DATAFRAME_TYPE):
            return self.new_scalar([df], np.dtype("bool"))
        else:
            return super().__call__(df)


def any_series(
    series,
    axis=0,
    bool_only=None,
    skipna=True,
    level=None,
    method=None,
):
    op = DataFrameAny(
        axis=axis,
        skipna=skipna,
        level=level,
        bool_only=bool_only,
        output_types=[OutputType.scalar],
        method=method,
    )
    return op(series)


def any_dataframe(
    df,
    axis=0,
    bool_only=None,
    skipna=True,
    level=None,
    method=None,
):
    output_types = [OutputType.series] if axis is not None else [OutputType.scalar]
    op = DataFrameAny(
        axis=axis,
        skipna=skipna,
        level=level,
        bool_only=bool_only,
        output_types=output_types,
        method=method,
    )
    return op(df)


def any_index(index):
    op = DataFrameAny(output_types=[OutputType.scalar])
    return op(index)
