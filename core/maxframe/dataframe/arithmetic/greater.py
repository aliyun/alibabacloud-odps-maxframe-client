# Copyright 1999-2026 Alibaba Group Holding Ltd.
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

from maxframe import opcodes
from maxframe.dataframe.arithmetic.core import DataFrameBinopUfunc
from maxframe.dataframe.arithmetic.docstring import bin_compare_doc
from maxframe.utils import classproperty


class DataFrameGreater(DataFrameBinopUfunc):
    _op_type_ = opcodes.GT

    _func_name = "gt"
    _rfunc_name = "lt"

    return_dtype = np.dtype(bool)

    @classproperty
    def _operator(self):
        return lambda lhs, rhs: lhs.gt(rhs)

    @classproperty
    def tensor_op_type(self):
        from maxframe.tensor.arithmetic import TensorGreaterThan

        return TensorGreaterThan


_gt_example = """
>>> a.gt(b, fill_value=0).execute()
a     True
b    False
c    False
d    False
e     True
f    False
dtype: bool
"""


@bin_compare_doc("Greater than", equiv=">", series_example=_gt_example)
def gt(df, other, axis="columns", level=None, fill_value=None):
    op = DataFrameGreater(
        axis=axis, level=level, lhs=df, rhs=other, fill_value=fill_value
    )
    return op(df, other)
