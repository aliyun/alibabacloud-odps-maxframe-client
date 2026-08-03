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
from maxframe.liteframe.accessors.struct_.core import LiteFrameStructExpr
from maxframe.liteframe.expressions import NamedExpr
from maxframe.liteframe.operators.project import LiteFrameProjection
from maxframe.utils import wrap_arrow_dtype

_STRUCT_DTYPE = wrap_arrow_dtype(pa.struct([("a", pa.int64()), ("b", pa.string())]))
_RAW = [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]


def test_struct_accessor_single_column_validation():
    """Test accessor only works on single-column LiteFrame."""
    df = LiteFrame({"a": _RAW, "b": _RAW}, dtype=_STRUCT_DTYPE)
    with pytest.raises(ValueError, match="multi-column"):
        df.struct.field("a")


def test_struct_accessor_dtype_validation():
    """Test accessor raises on non-struct dtype."""
    df = LiteFrame({"col": [1, 2, 3]})
    with pytest.raises(AttributeError, match="struct values"):
        df.struct.field("a")


@pytest.mark.parametrize(
    "args,out_name,expected_dtype",
    [
        (("a",), "a", wrap_arrow_dtype(pa.int64())),
        ((0,), "a", wrap_arrow_dtype(pa.int64())),
        (("b",), "b", wrap_arrow_dtype(pa.string())),
    ],
)
def test_struct_field_dtype_inference(args, out_name, expected_dtype):
    """Test struct.field: dtype inference and output column naming."""
    df = LiteFrame({"s": _RAW}, dtype=_STRUCT_DTYPE)
    result = df.struct.field(*args)

    # Verify result is LiteFrame
    assert isinstance(result, LiteFrame)

    # Verify operator structure
    op = result.data.op
    assert isinstance(op, LiteFrameProjection)
    assert len(op.projections) == 1

    # Verify projection structure — field() renames output to field name
    projection = op.projections[0]
    assert isinstance(projection, NamedExpr)
    assert projection.name == out_name
    assert isinstance(projection.expr, LiteFrameStructExpr)
    assert projection.expr.method == "field"
    assert projection.expr.args == tuple(args)

    # Verify dtype inference
    assert result.dtypes.iloc[0] == expected_dtype
    assert projection.dtype == expected_dtype


def test_struct_nested_field_dtype_inference():
    """Test struct.field with nested list path resolves to inner field dtype/name."""
    nested_dtype = wrap_arrow_dtype(
        pa.struct([("inner", pa.struct([("x", pa.int64()), ("y", pa.string())]))])
    )
    df = LiteFrame(
        {"s": [{"inner": {"x": 1, "y": "a"}}, {"inner": {"x": 2, "y": "b"}}]},
        dtype=nested_dtype,
    )
    result = df.struct.field(["inner", "x"])

    op = result.data.op
    assert isinstance(op, LiteFrameProjection)
    projection = op.projections[0]
    assert isinstance(projection, NamedExpr)
    assert projection.name == "x"
    assert isinstance(projection.expr, LiteFrameStructExpr)
    assert projection.expr.args == (["inner", "x"],)
    assert result.dtypes.iloc[0] == wrap_arrow_dtype(pa.int64())


def test_struct_dtypes_property():
    """struct.dtypes returns a plain pandas.Series of field dtypes."""
    df = LiteFrame({"s": _RAW}, dtype=_STRUCT_DTYPE)
    result = df.struct.dtypes
    assert list(result.index) == ["a", "b"]
    assert result["a"] == wrap_arrow_dtype(pa.int64())
    assert result["b"] == wrap_arrow_dtype(pa.string())
