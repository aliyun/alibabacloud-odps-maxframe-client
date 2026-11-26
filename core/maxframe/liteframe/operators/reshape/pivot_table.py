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
from maxframe.core import OutputType
from maxframe.dataframe.utils import make_column_list
from maxframe.liteframe.core import FrameMetadata
from maxframe.liteframe.operators.core import LiteFrameOperator, LiteFrameOperatorMixin
from maxframe.serialization.serializables import AnyField, BoolField, KeyField


class LiteFramePivotTable(LiteFrameOperator, LiteFrameOperatorMixin):
    """Operator for both ``pivot_table()`` and ``pivot()`` on LiteFrame.

    ``pivot()`` delegates to ``pivot_table()`` with
    ``aggfunc=PivotUniqueReduction``, so a single operator, tiler, and
    executor serve both APIs.
    """

    _op_type_ = opcodes.PIVOT_TABLE  # 744
    _output_type_ = OutputType.liteframe
    _supports_dynamic_schema = True

    values = AnyField("values", default=None)
    index = AnyField("index", default=None)
    columns = AnyField("columns", default=None)
    aggfunc = AnyField("aggfunc", default="mean")
    fill_value = AnyField("fill_value", default=None)
    dropna = BoolField("dropna", default=True)
    sort = BoolField("sort", default=True)

    # Runtime-resolved distinct values of the ``columns`` parameter.
    # Set in ``__call__`` via ``input[columns_list].drop_duplicates()``.
    # Becomes ``inputs[1]`` after ``_set_inputs``.
    distinct_values = KeyField("distinct_values", default=None)

    def __call__(self, liteframe):
        dtypes = liteframe.dtypes
        columns_list = make_column_list(self.columns, dtypes)

        # Column-value discovery: project to columns cols, then drop_duplicates.
        # After rechunk (in the tiler) this is a single-chunk broadcast input.
        self.distinct_values = liteframe[columns_list].drop_duplicates()

        # Schema-dynamic output: column count/names unknown until execution.
        return self.new_liteframe(
            [liteframe, self.distinct_values],
            shape=(np.nan, np.nan),
            physical_dtypes=None,
            frame_metadata=FrameMetadata(range_columns={}),
        )

    @classmethod
    def _set_inputs(cls, op, inputs):
        super()._set_inputs(op, inputs)
        if op.distinct_values is not None:  # pragma: no branch
            op.distinct_values = inputs[1]
