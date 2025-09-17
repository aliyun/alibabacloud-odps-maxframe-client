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
from ...serialization.serializables import StringField
from .core import DataFrameReduction, DataFrameReductionMixin, ReductionCallable


class StrLenReductionCallable(ReductionCallable):
    def __call__(self, value):
        return build_str_concat_object(value, **self.kwargs)


class DataFrameStrConcat(DataFrameReduction, DataFrameReductionMixin):
    _op_type_ = opcodes.STR_CONCAT
    _func_name = "str_concat"

    sep = StringField("sep", default=None)
    na_rep = StringField("na_rep", default=None)

    def get_reduction_args(self, axis=None):
        return dict(sep=self.sep, na_rep=self.na_rep)

    @property
    def is_atomic(self):
        return True

    @classmethod
    def get_reduction_callable(cls, op: "DataFrameStrConcat"):
        sep, na_rep = op.sep, op.na_rep
        return StrLenReductionCallable(
            func_name="str_concat", kwargs=dict(sep=sep, na_rep=na_rep)
        )


def build_str_concat_object(df, sep=None, na_rep=None):
    output_type = OutputType.series if df.ndim == 2 else OutputType.scalar
    op = DataFrameStrConcat(sep=sep, na_rep=na_rep, output_types=[output_type])
    return op(df)
