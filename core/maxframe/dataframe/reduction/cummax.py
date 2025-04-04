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
from .core import DataFrameCumReductionMixin, DataFrameCumReductionOperator


class DataFrameCummax(DataFrameCumReductionOperator, DataFrameCumReductionMixin):
    _op_type_ = opcodes.CUMMAX
    _func_name = "cummax"


def cummax(df, axis=None, skipna=True):
    op = DataFrameCummax(
        axis=axis,
        skipna=skipna,
        output_types=df.op.output_types,
    )
    return op(df)
