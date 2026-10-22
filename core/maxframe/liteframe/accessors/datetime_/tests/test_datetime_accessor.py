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
import pandas as pd
import pyarrow as pa
import pytest

from maxframe.liteframe import LiteFrame
from maxframe.liteframe.accessors.datetime_.core import (
    DATETIME_MULTI_METHOD_OUTPUT_DTYPES,
    DatetimeMultiColumnHandler,
    LiteFrameDatetimeExpr,
    LiteFrameDatetimeMultiExpr,
)
from maxframe.liteframe.expressions import (
    LiteFrameColumn,
    LiteFrameMultiColumnExpr,
    LiteFrameMultiColumnSubExpr,
    NamedExpr,
)
from maxframe.liteframe.initializer import LiteFrame as LF
from maxframe.liteframe.operators.project import LiteFrameProjection
from maxframe.utils import is_arrow_dtype_supported, wrap_arrow_dtype


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
        # One representative per dtype class; per-method dtype correctness
        # is covered by DATETIME_PROPERTY_OUTPUT_DTYPE + execution tests.
        ("year", wrap_arrow_dtype(pa.int32())),
        ("is_month_start", wrap_arrow_dtype(pa.bool_())),
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


def test_datetime_multi_expr_construction():
    """LiteFrameDatetimeMultiExpr extends LiteFrameMultiColumnExpr with method field."""
    col_expr = LiteFrameColumn(name="ts", dtype=pd.DatetimeTZDtype(tz="UTC"))
    multi = LiteFrameDatetimeMultiExpr(
        operands=[col_expr],
        method="isocalendar",
        is_property=False,
        output_names=["year", "week", "day"],
        dtypes={"year": "Int32", "week": "Int32", "day": "Int32"},
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
    lf = LF({"ts": pd.date_range("2020-01-01", periods=2)})
    lf2 = lf.select("ts")
    result = lf2.dt.year

    op = result.data.op
    assert isinstance(op, LiteFrameProjection)
    expr = op.projections[0].expr
    assert hasattr(expr, "operand")
    assert isinstance(expr.operand, LiteFrameColumn)
    assert result.data.inputs[0] is lf._data


# ---------------------------------------------------------------------------
# Timedelta / duration accessor
# ---------------------------------------------------------------------------


def _td_series():
    return pd.Series(
        pd.to_timedelta(
            np.array([93784005006000, -129600000000000, 7], dtype="timedelta64[ns]")
        )
    )


@pytest.mark.parametrize(
    "method,expected_dtype",
    [
        ("days", wrap_arrow_dtype(pa.int64())),
        ("seconds", wrap_arrow_dtype(pa.int32())),
        ("microseconds", wrap_arrow_dtype(pa.int32())),
        ("nanoseconds", wrap_arrow_dtype(pa.int32())),
    ],
)
def test_timedelta_property_dtype_inference(method, expected_dtype):
    """Timedelta properties infer the correct dtype without execution."""
    df = LiteFrame({"col": _td_series()})
    result = getattr(df.dt, method)

    assert isinstance(result, LiteFrame)
    op = result.data.op
    assert isinstance(op, LiteFrameProjection)
    projection = op.projections[0]
    assert isinstance(projection.expr, LiteFrameDatetimeExpr)
    assert projection.expr.method == method
    assert projection.expr.is_property is True
    assert result.dtypes.iloc[0] == expected_dtype
    assert projection.dtype == expected_dtype


def test_timedelta_total_seconds_is_method():
    """.dt.total_seconds() is a method returning float64."""
    df = LiteFrame({"col": _td_series()})
    result = df.dt.total_seconds()

    op = result.data.op
    projection = op.projections[0]
    assert isinstance(projection.expr, LiteFrameDatetimeExpr)
    assert projection.expr.method == "total_seconds"
    assert projection.expr.is_property is False
    assert result.dtypes.iloc[0] == wrap_arrow_dtype(pa.float64())


def test_timedelta_components_structure():
    """.dt.components produces a 7-column int64 LiteFrame (property, not method)."""
    df = LiteFrame({"col": _td_series()})
    result = df.dt.components

    expected_cols = [
        "days",
        "hours",
        "minutes",
        "seconds",
        "milliseconds",
        "microseconds",
        "nanoseconds",
    ]
    assert isinstance(result, LiteFrame)
    assert list(result.columns) == expected_cols
    for col_name in expected_cols:
        assert result.dtypes[col_name] == wrap_arrow_dtype(pa.int64())

    op = result.data.op
    assert isinstance(op, LiteFrameProjection)
    assert len(op.projections) == 7
    for proj in op.projections:
        assert isinstance(proj.expr, LiteFrameMultiColumnSubExpr)


def test_timedelta_accessor_accepts_arrow_duration():
    """ArrowDtype(duration[ns]) is recognized as datetimelike."""
    df = LiteFrame({"col": _td_series()})
    # Should not raise; accessing .dt already validated the dtype.
    _ = df.dt.days


def test_timedelta_multi_column_handler_infer_dtypes():
    """DatetimeMultiColumnHandler.infer_dtypes returns 7 int64 columns for components."""
    result = DatetimeMultiColumnHandler.infer_dtypes("components", None)
    assert set(result) == {
        "days",
        "hours",
        "minutes",
        "seconds",
        "milliseconds",
        "microseconds",
        "nanoseconds",
    }
    assert all(v == wrap_arrow_dtype(pa.int64()) for v in result.values())
    assert "components" in DATETIME_MULTI_METHOD_OUTPUT_DTYPES


# ---------------------------------------------------------------------------
# New method-style accessors: strftime, day_name, month_name, normalize,
# round, floor, ceil
# ---------------------------------------------------------------------------


def _ts_series():
    return pd.Series(pd.date_range("2020-01-01 10:30:45", periods=3, freq="h"))


@pytest.mark.parametrize(
    "method,expected_dtype",
    [
        ("strftime", wrap_arrow_dtype(pa.string())),
        ("day_name", wrap_arrow_dtype(pa.string())),
        ("month_name", wrap_arrow_dtype(pa.string())),
    ],
)
def test_datetime_string_method_dtype_inference(method, expected_dtype):
    """String-returning dt methods infer string dtype without execution."""
    df = LiteFrame({"col": _ts_series()})
    if method == "strftime":
        result = df.dt.strftime("%Y-%m-%d")
    else:
        result = getattr(df.dt, method)()

    assert isinstance(result, LiteFrame)
    op = result.data.op
    projection = op.projections[0]
    assert isinstance(projection.expr, LiteFrameDatetimeExpr)
    assert projection.expr.method == method
    assert projection.expr.is_property is False
    assert result.dtypes.iloc[0] == expected_dtype
    assert projection.dtype == expected_dtype


def test_datetime_strftime_passes_args():
    """strftime captures the format string in expr.args."""
    df = LiteFrame({"col": _ts_series()})
    result = df.dt.strftime("%Y/%m/%d")

    projection = result.data.op.projections[0]
    assert projection.expr.args == ("%Y/%m/%d",)
    assert projection.expr.kwargs == {}


@pytest.mark.parametrize("method", ["normalize", "round", "floor", "ceil"])
def test_datetime_preserving_method_dtype_inference(method):
    """normalize/round/floor/ceil preserve the input datetime dtype."""
    df = LiteFrame({"col": _ts_series()})
    if method == "normalize":
        result = df.dt.normalize()
    else:
        result = getattr(df.dt, method)("1h")

    input_dtype = df.dtypes.iloc[0]
    projection = result.data.op.projections[0]
    assert isinstance(projection.expr, LiteFrameDatetimeExpr)
    assert projection.expr.method == method
    assert projection.expr.is_property is False
    assert result.dtypes.iloc[0] == input_dtype
    assert projection.dtype == input_dtype


@pytest.mark.parametrize("method", ["round", "floor", "ceil"])
def test_datetime_round_method_passes_freq(method):
    """round/floor/ceil capture the freq string in expr.args."""
    df = LiteFrame({"col": _ts_series()})
    result = getattr(df.dt, method)("30min")

    projection = result.data.op.projections[0]
    assert projection.expr.args == ("30min",)
    assert projection.expr.kwargs == {}


# ---------------------------------------------------------------------------
# Dtype-method gating (review #7)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["days"])
def test_dt_duration_method_on_datetime_raises(method):
    """timedelta-only properties raise AttributeError on a datetime column
    at graph-construction time, matching pandas."""
    df = LiteFrame({"col": _ts_series()})
    with pytest.raises(AttributeError, match="timedelta"):
        _ = getattr(df.dt, method)


def test_dt_total_seconds_on_datetime_raises():
    """total_seconds (timedelta-only method) raises on a datetime column."""
    df = LiteFrame({"col": _ts_series()})
    with pytest.raises(AttributeError, match="timedelta"):
        _ = df.dt.total_seconds()


def test_dt_components_on_datetime_raises():
    """components (timedelta-only multi-column) raises on a datetime column."""
    df = LiteFrame({"col": _ts_series()})
    with pytest.raises(AttributeError, match="timedelta"):
        _ = df.dt.components


@pytest.mark.parametrize("method", ["year"])
def test_dt_datetime_method_on_timedelta_raises(method):
    """datetime-only properties raise AttributeError on a timedelta column,
    matching pandas (gated both directions)."""
    df = LiteFrame({"col": _td_series()})
    with pytest.raises(AttributeError, match="datetimelike"):
        _ = getattr(df.dt, method)


def test_dt_isocalendar_on_timedelta_raises():
    """isocalendar (datetime-only multi-column) raises on a timedelta column."""
    df = LiteFrame({"col": _td_series()})
    with pytest.raises(AttributeError, match="datetimelike"):
        _ = df.dt.isocalendar()


@pytest.mark.parametrize("method", ["round", "floor", "ceil"])
def test_dt_round_floor_ceil_on_timedelta_succeeds(method):
    """round/floor/ceil are shared between datetime and timedelta (pandas
    supports both), so they must not raise on a timedelta column (review #7
    follow-up: the coarse gate treated them as datetime-only)."""
    df = LiteFrame({"col": _td_series()})
    result = getattr(df.dt, method)("D")
    # Builds a projection without raising; dtype preserved as timedelta.
    projection = result.data.op.projections[0]
    assert isinstance(projection.expr, LiteFrameDatetimeExpr)
    assert projection.expr.method == method


def test_dt_timedelta_seconds_canonicalizes_to_ms_metadata():
    """timedelta64[s] canonicalizes to duration[ms] in the core dtype
    owner so metadata matches the physical Polars schema (review
    follow-up: canonicalization lived only in the executor, so metadata
    said duration[s] while the physical result was Duration(ms))."""
    if not is_arrow_dtype_supported():
        pytest.skip("pyarrow>=7.0 required for pd.ArrowDtype")
    s_dtype = pd.ArrowDtype(pa.duration("s"))
    df = LiteFrame({"col": pd.Series([129600], dtype=s_dtype)})
    result = df.dt.floor("D")
    # Metadata must report duration[ms], matching the codegen's physical
    # Duration(ms) (Polars rejects 's').
    expected = wrap_arrow_dtype(pa.duration("ms"))
    assert result.dtypes.iloc[0] == expected
    assert result.data.op.projections[0].dtype == expected
