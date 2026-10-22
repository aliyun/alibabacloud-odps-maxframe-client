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

import pandas as pd
import pyarrow as pa

from maxframe.codegen.spe.core import SPECodeContext
from maxframe.codegen.spe.liteframe.filter import LiteFrameFilterAdapter
from maxframe.liteframe.expressions import (
    LiteFrameColumn,
    LiteFrameGt,
    LiteFrameLiteral,
)
from maxframe.liteframe.operators.filter import LiteFrameFilter
from maxframe.utils import wrap_arrow_dtype


def test_filter_adapter_with_expr_predicate(make_liteframe):
    """LiteFrameFilterAdapter should generate _run_filter call."""
    df = pd.DataFrame({"a": [1, 2, 3, 4]})
    data = make_liteframe(df)
    col_a = LiteFrameColumn(name="a", dtype=wrap_arrow_dtype(pa.int64()))
    lit = LiteFrameLiteral(value=2, dtype=wrap_arrow_dtype(pa.int64()))
    predicate = LiteFrameGt(left=col_a, right=lit, dtype=wrap_arrow_dtype(pa.bool_()))
    op = LiteFrameFilter(predicate=predicate)
    out = op(data)
    op = out.op
    adapter = LiteFrameFilterAdapter()
    context = SPECodeContext()
    result = adapter.generate_code(op, context)
    assert len(result) == 1
    assert "LiteFrameFilterAdapter._run_filter" in result[0]


def test_filter_run_filter_with_expr():
    """Test _run_filter with an expression predicate."""
    df = pd.DataFrame({"a": [1, 2, 3, 4]})
    col_a = LiteFrameColumn(name="a", dtype=wrap_arrow_dtype(pa.int64()))
    lit = LiteFrameLiteral(value=2, dtype=wrap_arrow_dtype(pa.int64()))
    predicate = LiteFrameGt(left=col_a, right=lit, dtype=wrap_arrow_dtype(pa.bool_()))
    result = LiteFrameFilterAdapter._run_filter(df, predicate)
    assert len(result) == 2
    assert list(result["a"]) == [3, 4]


def test_filter_run_filter_with_mask():
    """Test _run_filter with a boolean mask DataFrame."""
    mask = pd.DataFrame({"mask": [True, False, True, False]})
    df = pd.DataFrame({"a": [1, 2, 3, 4]})
    result = LiteFrameFilterAdapter._run_filter(df, None, mask_df=mask)
    assert len(result) == 2
    assert list(result["a"]) == [1, 3]
