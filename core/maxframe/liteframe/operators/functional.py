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
from maxframe.core.operator.base import WorkLoadType
from maxframe.liteframe.operators.core import LiteFrameOperator, LiteFrameOperatorMixin
from maxframe.serialization import PickleContainer
from maxframe.serialization.serializables import (
    AnyField,
    BoolField,
    DictField,
    FunctionField,
    Int32Field,
    StringField,
    TupleField,
)
from maxframe.udf import BuiltinFunction, copy_func_scheduling_hints
from maxframe.utils import find_objects


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

    # When True (default), row-preserving transform modes restore the input
    # row order and chunking (axis=0 re-split, grouped place-back) so the
    # output inherits the input's ``alignment_id``.  When False the output
    # keeps the cheaper single-chunk / reducer-order layout with a fresh
    # id.  Ignored for non-row-preserving modes (apply/apply_chunk/...
    # axis=0) and for groupby apply/apply_chunk.  apply(axis=1) is always
    # 1:1 chunk-preserving and always inherits regardless of this flag.
    placeback = BoolField("placeback", default=True)

    @property
    def output_limit(self):
        return self.n_output or 1

    def __init__(self, **kw):
        super().__init__(**kw)
        if hasattr(self, "func"):
            copy_func_scheduling_hints(self.func, self)
        # User UDFs run GILHeavy unless copy_func_scheduling_hints set GPU.
        # _parse_kwargs guarantees a private (fresh or copied) hint here.
        if self.scheduling_hint.work_load_type is None and not self.scheduling_hint.gpu:
            self.scheduling_hint.work_load_type = WorkLoadType.GILHeavy

    def has_custom_code(self) -> bool:
        if not isinstance(self.func, BuiltinFunction):
            return True
        # A user callable may still ride in ``args``/``kwargs`` behind a
        # builtin wrapper (e.g. the ``groupby.filter`` predicate).  Plain
        # fields carry it as a ``PickleContainer``, so it has to be detected
        # here: the op must stay on the custom-code path, otherwise the
        # worker neither unpacks the container nor isolates the user code.
        for value in (self.args, self.kwargs):
            for found in find_objects(value, types=PickleContainer, checker=callable):
                if not isinstance(found, BuiltinFunction):
                    return True
        return False

    def _inherit_alignment_id(self, inputs):
        if not self.placeback:
            return None
        # Transform modes preserve the row count; apply(axis=1) is a 1:1
        # per-row mapping over 1:1 chunks, so it is always row-aligned with
        # the input and inherits regardless of the placeback flag (the flag
        # is not exposed on the apply API).
        if self.mode in ("transform", "transform_chunk"):
            return self._inherit_single_input_alignment_id(inputs)
        if self.mode == "apply" and (self.axis or 0) == 1:
            return self._inherit_single_input_alignment_id(inputs)
        return None
