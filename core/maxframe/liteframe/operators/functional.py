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

from maxframe import opcodes
from maxframe.core import OutputType
from maxframe.liteframe.operators.core import LiteFrameOperator, LiteFrameOperatorMixin
from maxframe.serialization.serializables import (
    AnyField,
    DictField,
    FunctionField,
    Int32Field,
    StringField,
    TupleField,
)
from maxframe.udf import BuiltinFunction, copy_func_scheduling_hints


class LiteFrameFunctional(LiteFrameOperatorMixin, LiteFrameOperator):
    _op_type_ = opcodes.LITEFRAME_FUNCTIONAL
    _op_module_ = "liteframe"
    _output_type_ = OutputType.liteframe

    func = FunctionField("func")
    axis = Int32Field("axis", default=0)
    mode = StringField("mode")  # apply/transform/apply_chunk/transform_chunk/flat_apply
    result_type = AnyField(
        "result_type", default=None
    )  # apply(axis=1) only: expand/reduce/broadcast
    args = TupleField("args", default=())
    kwargs = DictField("kwargs", default=None)

    # GroupBy support
    groupby_params = DictField("groupby_params", default=None)

    # Frame type control
    input_frame_type = StringField("input_frame_type", default="pandas")

    # Output alignment
    check_output_dtypes = StringField("check_output_dtypes", default="raises")

    # flat_apply: columns passed to the UDF (None = all visible columns).  The
    # remaining visible columns are broadcast (repeated) across the expanded
    # output rows.  Carried to the executor because DPE chunks do not
    # propagate frame metadata.
    subset = AnyField("subset", default=None)

    # Declared output schemas (list of pd.Series, one per output), carried to
    # the executor because DPE chunks do not propagate physical_dtypes.  Used
    # by per-row/per-column apply to name/align output columns.
    output_dtypes = AnyField("output_dtypes", default=None)

    # Multiple outputs support
    n_output = Int32Field("n_output", default=None)

    @property
    def output_limit(self):
        return self.n_output or 1

    def __init__(self, **kw):
        super().__init__(**kw)
        if hasattr(self, "func"):
            copy_func_scheduling_hints(self.func, self)

    def has_custom_code(self) -> bool:
        return not isinstance(self.func, BuiltinFunction)
