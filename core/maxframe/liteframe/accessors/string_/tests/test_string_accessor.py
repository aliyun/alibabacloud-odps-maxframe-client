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

import pyarrow as pa
import pytest

from maxframe.liteframe import LiteFrame
from maxframe.liteframe.accessors.string_.core import LiteFrameStrExpr
from maxframe.liteframe.expressions import NamedExpr
from maxframe.liteframe.operators.project import LiteFrameProjection
from maxframe.utils import wrap_arrow_dtype


def test_str_accessor_single_column_validation():
    """Test accessor only works on single-column LiteFrame."""
    df = LiteFrame({"a": ["hello"], "b": ["world"]})
    with pytest.raises(ValueError, match="multi-column"):
        df.str.upper()


@pytest.mark.parametrize(
    "method,args,expected_dtype",
    [
        ("upper", [], wrap_arrow_dtype(pa.string())),
        ("contains", ["hello"], wrap_arrow_dtype(pa.bool_())),
        ("len", [], wrap_arrow_dtype(pa.int64())),
    ],
)
def test_str_methods_dtype_inference(method, args, expected_dtype):
    """Test str methods: dtype inference without execution."""
    df = LiteFrame({"col": ["hello", "world"]})
    result = getattr(df.str, method)(*args)

    # Verify result is LiteFrame
    assert isinstance(result, LiteFrame)

    # Verify operator structure
    op = result.data.op
    assert isinstance(op, LiteFrameProjection)
    assert len(op.projections) == 1

    # Verify projection structure
    projection = op.projections[0]
    assert isinstance(projection, NamedExpr)
    assert projection.name == "col"
    assert isinstance(projection.expr, LiteFrameStrExpr)
    assert projection.expr.method == method
    assert projection.expr.args == tuple(args)

    # Verify dtype inference
    assert result.dtypes.iloc[0] == expected_dtype
    assert projection.dtype == expected_dtype


def test_string_accessor_after_projection_fuses():
    """String accessor on a projection-produced column should fuse."""
    from maxframe.liteframe.expressions import LiteFrameColumn
    from maxframe.liteframe.initializer import LiteFrame as LF

    lf = LF({"s": ["abc", "de"]})
    lf2 = lf.select("s")
    result = lf2.str.upper()

    op = result.data.op
    assert isinstance(op, LiteFrameProjection)
    # The operand should be a LiteFrameColumn, not another projection layer
    expr = op.projections[0].expr
    assert hasattr(expr, "operand")
    assert isinstance(expr.operand, LiteFrameColumn)
    # Input should be the root source
    assert result.data.inputs[0] is lf._data
