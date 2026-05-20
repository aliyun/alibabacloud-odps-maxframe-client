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


class DataFrameLess(DataFrameBinopUfunc):
    _op_type_ = opcodes.LT

    _func_name = "lt"
    _rfunc_name = "gt"

    return_dtype = np.dtype(bool)

    @classproperty
    def _operator(self):
        return lambda lhs, rhs: lhs.lt(rhs)

    @classproperty
    def tensor_op_type(self):
        from maxframe.tensor.arithmetic import TensorLessThan

        return TensorLessThan


_lt_example = """
>>> a.lt(b, fill_value=0).execute()
a    False
b    False
c     True
d    False
e    False
f     True
dtype: bool
"""


@bin_compare_doc("Less than", equiv="<", series_example=_lt_example)
def lt(df, other, axis="columns", level=None, fill_value=None):
    op = DataFrameLess(axis=axis, level=level, lhs=df, rhs=other, fill_value=fill_value)
    return op(df, other)
