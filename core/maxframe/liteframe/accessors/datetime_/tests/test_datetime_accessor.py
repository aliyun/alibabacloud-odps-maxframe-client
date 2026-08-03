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
import pytest

from maxframe.liteframe import LiteFrame
from maxframe.liteframe.accessors.datetime_.core import LiteFrameDatetimeExpr
from maxframe.liteframe.expressions import NamedExpr
from maxframe.liteframe.operators.project import LiteFrameProjection
from maxframe.utils import wrap_arrow_dtype


def test_datetime_accessor_single_column_validation():
    """Test accessor only works on single-column LiteFrame."""
    df = LiteFrame({"a": pd.date_range("2020-01-01", periods=3), "b": [1, 2, 3]})
    with pytest.raises(ValueError, match="multi-column"):
        _ = df.dt.year


def test_datetime_accessor_dtype_validation():
    """Test accessor raises on non-datetime dtype."""
    df = LiteFrame({"col": ["hello", "world"]})
    with pytest.raises(AttributeError, match="datetimelike"):
        _ = df.dt.year


@pytest.mark.parametrize(
    "method,expected_dtype",
    [
        ("year", wrap_arrow_dtype(pa.int32())),
        ("month", wrap_arrow_dtype(pa.int32())),
        ("day", wrap_arrow_dtype(pa.int32())),
        ("hour", wrap_arrow_dtype(pa.int32())),
        ("minute", wrap_arrow_dtype(pa.int32())),
        ("second", wrap_arrow_dtype(pa.int32())),
        ("microsecond", wrap_arrow_dtype(pa.int32())),
        ("nanosecond", wrap_arrow_dtype(pa.int32())),
        ("dayofweek", wrap_arrow_dtype(pa.int32())),
        ("dayofyear", wrap_arrow_dtype(pa.int32())),
        ("weekofyear", wrap_arrow_dtype(pa.int32())),
        ("quarter", wrap_arrow_dtype(pa.int32())),
        ("days_in_month", wrap_arrow_dtype(pa.int32())),
        ("daysinmonth", wrap_arrow_dtype(pa.int32())),
        ("is_month_start", wrap_arrow_dtype(pa.bool_())),
        ("is_month_end", wrap_arrow_dtype(pa.bool_())),
        ("is_quarter_start", wrap_arrow_dtype(pa.bool_())),
        ("is_quarter_end", wrap_arrow_dtype(pa.bool_())),
        ("is_year_start", wrap_arrow_dtype(pa.bool_())),
        ("is_year_end", wrap_arrow_dtype(pa.bool_())),
        ("date", wrap_arrow_dtype(pa.date32())),
        ("time", wrap_arrow_dtype(pa.time64("us"))),
    ],
)
def test_datetime_property_dtype_inference(method, expected_dtype):
    """Test datetime properties: dtype inference without execution."""
    df = LiteFrame({"col": pd.date_range("2020-01-01", periods=3)})
    result = getattr(df.dt, method)

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
    assert isinstance(projection.expr, LiteFrameDatetimeExpr)
    assert projection.expr.method == method
    assert projection.expr.is_property is True

    # Verify dtype inference
    assert result.dtypes.iloc[0] == expected_dtype
    assert projection.dtype == expected_dtype


def test_datetime_accessor_with_tz_dtype():
    """Test accessor works with timezone-aware datetime dtype."""
    df = LiteFrame({"col": pd.date_range("2020-01-01", periods=3, tz="UTC")})
    result = df.dt.year
    assert isinstance(result, LiteFrame)
    assert result.dtypes.iloc[0] == wrap_arrow_dtype(pa.int32())


from maxframe.liteframe.accessors.datetime_.core import (
    DATETIME_MULTI_METHOD_OUTPUT_DTYPES,
    DatetimeMultiColumnHandler,
    LiteFrameDatetimeMultiExpr,
)


def test_datetime_multi_expr_construction():
    """LiteFrameDatetimeMultiExpr extends LiteFrameMultiColumnExpr with method field."""
    from maxframe.liteframe.expressions import LiteFrameColumn, LiteFrameMultiColumnExpr

    col_expr = LiteFrameColumn(name="ts", dtype=pd.DatetimeTZDtype(tz="UTC"))
    multi = LiteFrameDatetimeMultiExpr(
        operand=col_expr,
        method="isocalendar",
        is_property=False,
        output_names=["year", "week", "day"],
        dtypes_dict={"year": "Int32", "week": "Int32", "day": "Int32"},
    )
    assert isinstance(multi, LiteFrameMultiColumnExpr)
    assert multi.method == "isocalendar"
    assert multi.is_property is False
    assert multi.output_names == ["year", "week", "day"]


def test_datetime_multi_column_handler_infer_dtypes():
    """DatetimeMultiColumnHandler.infer_dtypes returns correct mapping for isocalendar."""
    result = DatetimeMultiColumnHandler.infer_dtypes(
        "isocalendar", pd.DatetimeTZDtype(tz="UTC")
    )
    assert result == DATETIME_MULTI_METHOD_OUTPUT_DTYPES["isocalendar"]
    assert list(result.keys()) == ["year", "week", "day"]
    assert all(v == wrap_arrow_dtype(pa.int32()) for v in result.values())


from maxframe.liteframe.expressions import LiteFrameMultiColumnSubExpr


def test_isocalendar_accessor_structure():
    """Test isocalendar() produces a multi-column LiteFrame with year/week/day."""
    df = LiteFrame({"col": pd.date_range("2020-01-01", periods=3)})
    result = df.dt.isocalendar()

    # Verify result is LiteFrame with 3 columns
    assert isinstance(result, LiteFrame)
    assert list(result.columns) == ["year", "week", "day"]

    # Verify operator structure
    op = result.data.op
    assert isinstance(op, LiteFrameProjection)
    assert len(op.projections) == 3

    # Each projection is a NamedExpr wrapping a LiteFrameMultiColumnSubExpr
    for i, name in enumerate(["year", "week", "day"]):
        proj = op.projections[i]
        assert isinstance(proj, NamedExpr)
        assert proj.name == name
        assert isinstance(proj.expr, LiteFrameMultiColumnSubExpr)
        assert proj.expr.output_name == name
        assert proj.expr.dtype == wrap_arrow_dtype(pa.int32())

    # Verify output dtypes
    for col_name in ["year", "week", "day"]:
        assert result.dtypes[col_name] == wrap_arrow_dtype(pa.int32())


def test_datetime_accessor_after_projection_fuses():
    """Datetime accessor on a projection-produced column should fuse."""
    from maxframe.liteframe.expressions import LiteFrameColumn
    from maxframe.liteframe.initializer import LiteFrame as LF

    lf = LF({"ts": pd.date_range("2020-01-01", periods=2)})
    lf2 = lf.select("ts")
    result = lf2.dt.year

    op = result.data.op
    assert isinstance(op, LiteFrameProjection)
    expr = op.projections[0].expr
    assert hasattr(expr, "operand")
    assert isinstance(expr.operand, LiteFrameColumn)
    assert result.data.inputs[0] is lf._data
