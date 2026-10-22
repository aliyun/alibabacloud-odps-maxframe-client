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

import datetime
import decimal

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pytest

from maxframe.liteframe.core import LITEFRAME_TYPE
from maxframe.liteframe.datasource.from_local import from_local_df  # noqa: F401
from maxframe.liteframe.expressions import (
    LiteFrameAdd,
    LiteFrameBaseExpr,
    LiteFrameColumn,
    LiteFrameDiv,
    LiteFrameExpr,
    LiteFrameFloorDiv,
    LiteFrameGt,
    LiteFrameInvert,
    LiteFrameLiteral,
    LiteFrameLShift,
    LiteFrameMul,
    LiteFrameMultiColumnExpr,
    LiteFrameMultiColumnSubExpr,
    LiteFrameNeg,
    LiteFrameRShift,
    LiteFrameUDFExpr,
    NamedExpr,
    _infer_literal_dtype,
    _infer_result_dtype,
    col,
)
from maxframe.liteframe.initializer import LiteFrame as LF
from maxframe.liteframe.operators.project import LiteFrameProjection
from maxframe.udf import BuiltinFunction
from maxframe.utils import wrap_arrow_dtype

# ---------------------------------------------------------------------------
# col() normalization
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "input_name, expected_name",
    [
        ("A", "a"),
        ("MyCol", "mycol"),
    ],
)
def test_col_normalizes_name_to_lowercase(input_name, expected_name):
    c = col(input_name)
    assert isinstance(c, LiteFrameColumn)
    assert c.name == expected_name
    assert c.dtype is None


# ---------------------------------------------------------------------------
# Literal dtype inference
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value, expected_dtype",
    [
        # bool / int / float / str / bytes — one representative per family
        # plus the numpy-scalar and Arrow-scalar coercion of each.
        (True, pa.bool_()),
        (np.bool_(True), pa.bool_()),
        (42, pa.int64()),
        (np.int64(42), pa.int64()),
        (pa.scalar(42), pa.int64()),
        (3.14, pa.float64()),
        (np.float64(3.14), pa.float64()),
        (pa.scalar(3.14), pa.float64()),
        ("hello", pa.string()),
        (pa.scalar("x"), pa.string()),
        (b"x", pa.binary()),
        (pa.scalar(b"x"), pa.binary()),
        # Temporal / decimal / null — mirrors _literal_dispatcher.
        (
            datetime.datetime(2023, 6, 15, 10, 30, 0, tzinfo=datetime.timezone.utc),
            pa.timestamp("us", tz="UTC"),
        ),
        (datetime.datetime(2023, 6, 15, 10, 30, 0), pa.timestamp("us")),
        (pd.Timestamp("2023-06-15 10:30:45"), pa.timestamp("ns")),
        (
            pd.Timestamp("2023-06-15 10:30:45", tz="UTC"),
            pa.timestamp("ns", tz="UTC"),
        ),
        (datetime.date(2023, 6, 15), pa.date32()),
        (datetime.time(10, 30, 45), pa.time64("us")),
        (datetime.timedelta(days=1), pa.duration("us")),
        (decimal.Decimal("1.50"), pa.decimal128(3, 2)),
        (decimal.Decimal("100"), pa.decimal128(3, 0)),
        (pd.NA, pa.null()),
        (None, pa.null()),
        # np.datetime64/np.timedelta64: ns maps to ns, coarse units (D/h/m/s)
        # map to us (Polars-native; polars cannot cast to 's').  Y/M are
        # rejected as ambiguous (tested separately).
        (np.datetime64("2023-06-15T10:30:45.123456789"), pa.timestamp("ns")),
        (np.datetime64("2023-06-15T10:30:45"), pa.timestamp("us")),
        (np.timedelta64(1, "ns"), pa.duration("ns")),
        (np.timedelta64(3600, "s"), pa.duration("us")),
    ],
)
def test_infer_literal_dtype(value, expected_dtype):
    assert _infer_literal_dtype(value) == wrap_arrow_dtype(expected_dtype)


@pytest.mark.parametrize(
    "value",
    [
        pa.scalar([1, 2, 3]),
        pa.scalar({"a": 1}),
        pa.scalar(
            [("a", 1)],
            type=pa.list_(
                pa.struct([pa.field("key", pa.string()), pa.field("value", pa.int64())])
            ),
        ),
    ],
    ids=["list_scalar", "struct_scalar", "list_of_struct_scalar"],
)
def test_infer_literal_dtype_rejects_nested_arrow_scalar(value):
    """Nested Arrow scalars (list/struct/map) are accepted by inference but
    rejected by codegen (only primitive scalars serialise); reject at
    inference too so the failure is clear and the boundary is consistent."""
    with pytest.raises(TypeError, match="does not support nested Arrow scalar"):
        _infer_literal_dtype(value)


@pytest.mark.parametrize("unit", ["Y", "M"], ids=["year", "month"])
@pytest.mark.parametrize(
    "make_value",
    [np.timedelta64, np.datetime64],
    ids=["timedelta64", "datetime64"],
)
def test_infer_literal_dtype_rejects_ambiguous_time_units(make_value, unit):
    """Y/M numpy temporal units are ambiguous (month/year length varies) and
    rejected at inference — pd.Timedelta/pd.Timestamp cannot accept them."""
    with pytest.raises(TypeError, match="ambiguous calendar unit"):
        _infer_literal_dtype(make_value(1, unit))


# ---------------------------------------------------------------------------
# Binary op dtype propagation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "left_dtype, right_dtype, expected_result_dtype",
    [
        (None, None, None),
        (None, pa.float64(), None),
        (pa.int64(), pa.float64(), "not_none"),
    ],
)
def test_binary_op_dtype_propagation(left_dtype, right_dtype, expected_result_dtype):
    a = (
        col("A")
        if left_dtype is None
        else LiteFrameColumn(name="A", dtype=wrap_arrow_dtype(left_dtype))
    )
    b = (
        col("B")
        if right_dtype is None
        else LiteFrameColumn(name="B", dtype=wrap_arrow_dtype(right_dtype))
    )
    result = a + b
    assert isinstance(result, LiteFrameAdd)
    if expected_result_dtype is None:
        assert result.dtype is None
    else:
        assert result.dtype is not None


# ---------------------------------------------------------------------------
# True division always produces float (pandas/polars/numpy semantics)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "op_cls, left_dtype, right_dtype, expected",
    [
        # int / int must be float64, not int64 -- otherwise groupby mean
        # (sum/count) would be planned as int (e77ffcc60 regression).
        (LiteFrameDiv, pa.int64(), pa.int64(), pa.float64()),
        (LiteFrameDiv, pa.int64(), pa.float64(), pa.float64()),
        (LiteFrameDiv, pa.float64(), pa.int64(), pa.float64()),
        # Floor division preserves the promoted integer dtype.
        (LiteFrameFloorDiv, pa.int64(), pa.int64(), pa.int64()),
        (LiteFrameFloorDiv, pa.int64(), pa.float64(), pa.float64()),
    ],
)
def test_true_division_infers_float_dtype(op_cls, left_dtype, right_dtype, expected):
    """True division always produces float; floor division keeps int/int int."""
    result = _infer_result_dtype(
        wrap_arrow_dtype(left_dtype), wrap_arrow_dtype(right_dtype), op_cls
    )
    assert result == wrap_arrow_dtype(expected)


def test_true_division_decimal_defers_to_full_promotion():
    """Decimal division returns None (defers to the full promotion path).

    No engine returns float64 for decimal division (polars -> Decimal(38, x),
    pandas -> object Decimal); the late-bind fallback must not declare
    float64 or the declared dtype mismatches the runtime result.
    """
    dec = wrap_arrow_dtype(pa.decimal128(10, 2))
    assert _infer_result_dtype(dec, dec, LiteFrameDiv) is None
    assert _infer_result_dtype(dec, wrap_arrow_dtype(pa.int64()), LiteFrameDiv) is None
    assert _infer_result_dtype(wrap_arrow_dtype(pa.int64()), dec, LiteFrameDiv) is None


def test_true_division_bind_resolves_float_dtype():
    """Binding an untyped int/int division expression resolves to float64."""
    lf = LF({"a": [1, 2, 3]})
    expr = col("a") / col("a")
    assert expr.dtype is None
    bound = expr.bind(lf._data._physical_dtypes)
    assert bound.dtype == wrap_arrow_dtype(pa.float64())


# ---------------------------------------------------------------------------
# Binary op with scalar (forward and reverse)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "expr_func, expected_literal_side",
    [
        (lambda a: a + 1, "right"),
        (lambda a: 1 + a, "left"),
    ],
)
def test_scalar_binary_op_with_none_dtype(expr_func, expected_literal_side):
    a = col("A")
    result = expr_func(a)
    assert isinstance(result, LiteFrameAdd)
    assert result.dtype is None
    lit = getattr(result, expected_literal_side)
    assert isinstance(lit, LiteFrameLiteral)
    assert lit.value == 1
    assert lit.dtype is None


# ---------------------------------------------------------------------------
# Chained / unary ops
# ---------------------------------------------------------------------------


def test_chained_binary_ops_propagate_none():
    a = col("A")
    b = col("B")
    result = a + b * 2
    assert isinstance(result, LiteFrameAdd)
    assert result.dtype is None
    assert isinstance(result.right, LiteFrameMul)
    assert result.right.dtype is None


def test_unary_op_with_none_dtype():
    a = col("A")
    result = -a
    assert isinstance(result, LiteFrameNeg)
    assert result.dtype is None


def test_invert_with_none_dtype():
    a = col("A")
    result = ~a
    assert isinstance(result, LiteFrameInvert)
    assert result.dtype is None


# ---------------------------------------------------------------------------
# bind — column resolution
# ---------------------------------------------------------------------------


def test_bind_column_resolves_dtype():
    dtypes = pd.Series(
        {"a": wrap_arrow_dtype(pa.int64()), "b": wrap_arrow_dtype(pa.float64())}
    )
    c = col("A")  # will be normalized to "a" by col()
    bound = c.bind(dtypes)
    assert isinstance(bound, LiteFrameColumn)
    assert bound.name == "a"
    assert bound.dtype == wrap_arrow_dtype(pa.int64())


def test_bind_column_already_bound_unchanged():
    dtypes = pd.Series({"a": wrap_arrow_dtype(pa.int64())})
    c = LiteFrameColumn(name="A", dtype=wrap_arrow_dtype(pa.int32()))
    bound = c.bind(dtypes)
    assert bound.dtype == wrap_arrow_dtype(pa.int32())  # explicit dtype preserved
    assert bound.name == "a"  # name is still normalized


def test_bind_column_missing_raises_keyerror():
    dtypes = pd.Series({"a": wrap_arrow_dtype(pa.int64())})
    c = col("Z")
    with pytest.raises(KeyError, match="z"):
        c.bind(dtypes)


# ---------------------------------------------------------------------------
# bind — literal resolution
# ---------------------------------------------------------------------------


def test_bind_literal_resolves_dtype():
    dtypes = pd.Series({"a": wrap_arrow_dtype(pa.int64())})
    lit = LiteFrameLiteral(value=42)
    bound = lit.bind(dtypes)
    assert bound.dtype == wrap_arrow_dtype(pa.int64())


def test_bind_literal_already_has_dtype():
    dtypes = pd.Series({"a": wrap_arrow_dtype(pa.int64())})
    lit = LiteFrameLiteral(value=42, dtype=wrap_arrow_dtype(pa.float64()))
    bound = lit.bind(dtypes)
    assert bound.dtype == wrap_arrow_dtype(pa.float64())  # explicit dtype preserved


# ---------------------------------------------------------------------------
# bind — compound expressions
# ---------------------------------------------------------------------------


def test_bind_binary_expr_resolves_children_and_infers_dtype():
    dtypes = pd.Series(
        {"a": wrap_arrow_dtype(pa.int64()), "b": wrap_arrow_dtype(pa.float64())}
    )
    expr = col("A") + col("B")
    bound = expr.bind(dtypes)
    assert isinstance(bound, LiteFrameAdd)
    assert bound.dtype == wrap_arrow_dtype(pa.float64())  # int64 + float64 -> float64
    assert bound.left.dtype == wrap_arrow_dtype(pa.int64())
    assert bound.right.dtype == wrap_arrow_dtype(pa.float64())


def test_bind_binary_expr_with_scalar():
    dtypes = pd.Series({"a": wrap_arrow_dtype(pa.int64())})
    expr = col("A") + 1
    bound = expr.bind(dtypes)
    assert isinstance(bound, LiteFrameAdd)
    assert bound.left.dtype == wrap_arrow_dtype(pa.int64())
    assert bound.right.dtype == wrap_arrow_dtype(
        pa.int64()
    )  # inferred from literal value
    assert bound.dtype == wrap_arrow_dtype(pa.int64())


def test_bind_chained_expression():
    dtypes = pd.Series(
        {"a": wrap_arrow_dtype(pa.int64()), "b": wrap_arrow_dtype(pa.float64())}
    )
    expr = col("A") + col("B") * 2
    bound = expr.bind(dtypes)
    assert isinstance(bound, LiteFrameAdd)
    assert bound.dtype == wrap_arrow_dtype(pa.float64())
    assert isinstance(bound.right, LiteFrameMul)
    assert bound.right.dtype == wrap_arrow_dtype(pa.float64())


def test_bind_unary_expr():
    dtypes = pd.Series({"a": wrap_arrow_dtype(pa.int64())})
    expr = -col("A")
    bound = expr.bind(dtypes)
    assert isinstance(bound, LiteFrameNeg)
    assert bound.operand.dtype == wrap_arrow_dtype(pa.int64())
    assert bound.dtype == wrap_arrow_dtype(pa.int64())


def test_bind_named_expr():
    dtypes = pd.Series({"a": wrap_arrow_dtype(pa.int64())})
    expr = NamedExpr(name="X", expr=col("A") + 1)
    bound = expr.bind(dtypes)
    assert isinstance(bound, NamedExpr)
    assert bound.name == "x"  # name is normalized to lowercase
    assert isinstance(bound.expr, LiteFrameAdd)
    assert bound.expr.dtype == wrap_arrow_dtype(pa.int64())
    assert bound.dtype == wrap_arrow_dtype(pa.int64())


def test_bind_does_not_mutate_original():
    dtypes = pd.Series({"a": wrap_arrow_dtype(pa.int64())})
    original = col("A")
    assert original.dtype is None
    bound = original.bind(dtypes)
    assert bound.dtype == wrap_arrow_dtype(pa.int64())
    assert original.dtype is None  # original unchanged


def test_bind_mixed_tree_with_some_explicit_dtypes():
    dtypes = pd.Series(
        {"a": wrap_arrow_dtype(pa.int64()), "b": wrap_arrow_dtype(pa.float64())}
    )
    # One leaf has explicit dtype, one doesn't
    a = LiteFrameColumn(name="A", dtype=wrap_arrow_dtype(pa.int32()))
    b = col("B")
    expr = a + b
    bound = expr.bind(dtypes)
    assert bound.left.dtype == wrap_arrow_dtype(pa.int32())  # explicit preserved
    assert bound.right.dtype == wrap_arrow_dtype(pa.float64())  # resolved from schema
    assert bound.dtype == wrap_arrow_dtype(pa.float64())  # int32 + float64 -> float64


# ---------------------------------------------------------------------------
# Multi-column expression tests
# ---------------------------------------------------------------------------


def test_multi_column_expr_construction():
    """LiteFrameMultiColumnExpr stores output_names and dtypes."""
    col_expr = LiteFrameColumn(name="ts", dtype=np.dtype("datetime64[ns]"))
    multi = LiteFrameMultiColumnExpr(
        operands=[col_expr],
        output_names=["year", "week", "day"],
        dtypes={"year": "Int32", "week": "Int32", "day": "Int32"},
    )
    assert multi.output_names == ["year", "week", "day"]
    assert multi.dtypes == {"year": "Int32", "week": "Int32", "day": "Int32"}
    assert isinstance(multi.operands[0], LiteFrameColumn)
    assert isinstance(multi, LiteFrameBaseExpr)
    assert not isinstance(multi, LiteFrameExpr)


def test_multi_column_sub_expr_construction():
    """LiteFrameMultiColumnSubExpr bridges to single-column world."""
    col_expr = LiteFrameColumn(name="ts", dtype=np.dtype("datetime64[ns]"))
    multi = LiteFrameMultiColumnExpr(
        operands=[col_expr],
        output_names=["year", "week", "day"],
        dtypes={"year": "Int32", "week": "Int32", "day": "Int32"},
    )
    sub = LiteFrameMultiColumnSubExpr(
        multi_expr=multi,
        output_name="year",
        dtype=wrap_arrow_dtype(pa.int32()),
    )
    assert sub.output_name == "year"
    assert sub.dtype == wrap_arrow_dtype(pa.int32())
    assert isinstance(sub, LiteFrameExpr)  # has dtype, works with NamedExpr
    assert isinstance(sub.multi_expr, LiteFrameMultiColumnExpr)


def test_multi_column_sub_expr_rename():
    """LiteFrameMultiColumnSubExpr.rename() produces a NamedExpr."""
    col_expr = LiteFrameColumn(name="ts", dtype=np.dtype("datetime64[ns]"))
    multi = LiteFrameMultiColumnExpr(
        operands=[col_expr],
        output_names=["year", "week", "day"],
        dtypes={"year": "Int32", "week": "Int32", "day": "Int32"},
    )
    sub = LiteFrameMultiColumnSubExpr(
        multi_expr=multi,
        output_name="year",
        dtype=wrap_arrow_dtype(pa.int32()),
    )
    named = sub.rename("year")
    assert isinstance(named, NamedExpr)
    assert named.name == "year"
    assert named.dtype == wrap_arrow_dtype(pa.int32())


def test_bind_multi_column_expr():
    """bind() resolves operand dtype in LiteFrameMultiColumnExpr."""
    dtypes = pd.Series({"ts": np.dtype("datetime64[ns]")})
    col_expr = col("ts")
    multi = LiteFrameMultiColumnExpr(
        operands=[col_expr],
        output_names=["year", "week", "day"],
        dtypes={"year": "Int32", "week": "Int32", "day": "Int32"},
    )
    bound = multi.bind(dtypes)
    assert isinstance(bound, LiteFrameMultiColumnExpr)
    assert isinstance(bound.operands[0], LiteFrameColumn)
    assert bound.operands[0].dtype == np.dtype("datetime64[ns]")
    assert bound.output_names == ["year", "week", "day"]
    assert bound.dtypes == {"year": "Int32", "week": "Int32", "day": "Int32"}


def test_bind_multi_column_sub_expr():
    """bind() resolves dtype from multi_expr's dtypes."""
    dtypes = pd.Series({"ts": np.dtype("datetime64[ns]")})
    col_expr = col("ts")
    multi = LiteFrameMultiColumnExpr(
        operands=[col_expr],
        output_names=["year", "week", "day"],
        dtypes={"year": "Int32", "week": "Int32", "day": "Int32"},
    )
    sub = LiteFrameMultiColumnSubExpr(
        multi_expr=multi,
        output_name="year",
        dtype=None,
    )
    bound = sub.bind(dtypes)
    assert isinstance(bound, LiteFrameMultiColumnSubExpr)
    assert bound.dtype == pd.Int32Dtype()
    assert bound.output_name == "year"


def test_bind_multi_column_sub_expr_polars_dtype():
    """Regression: ``bind`` resolves a polars dtype
    from ``dtypes`` via ``to_pandas_dtype`` instead of crashing."""
    dtypes = pd.Series({"a": wrap_arrow_dtype(pa.int64())})
    udf = LiteFrameUDFExpr(
        func=lambda df: df,
        mode="transform_chunk",
        operands=[LiteFrameColumn(name="a", dtype="int64")],
        output_names=["a2"],
        dtypes={"a2": pl.Int64},
        input_frame_type="polars",
        check_output_dtypes="raises",
    )
    sub = LiteFrameMultiColumnSubExpr(multi_expr=udf, output_name="a2", dtype=pl.Int64)
    bound = sub.bind(dtypes)
    assert isinstance(bound, LiteFrameMultiColumnSubExpr)
    assert bound.dtype == np.dtype("int64")


def test_udf_expr_dtype_property_polars_dtype():
    """Regression: the map-mode ``.dtype`` property
    resolves a polars dtype via ``to_pandas_dtype`` instead of crashing."""
    udf = LiteFrameUDFExpr(
        func=lambda x: x,
        mode="map",
        operands=[LiteFrameColumn(name="a", dtype="int64")],
        output_names=["a"],
        dtypes={"a": pl.Int64},
        input_frame_type="pandas",
        check_output_dtypes="raises",
    )
    assert udf.dtype == np.dtype("int64")


def test_bind_named_expr_with_sub_expr():
    """bind() works with NamedExpr wrapping LiteFrameMultiColumnSubExpr."""
    dtypes = pd.Series({"ts": np.dtype("datetime64[ns]")})
    col_expr = col("ts")
    multi = LiteFrameMultiColumnExpr(
        operands=[col_expr],
        output_names=["year", "week", "day"],
        dtypes={"year": "Int32", "week": "Int32", "day": "Int32"},
    )
    sub = LiteFrameMultiColumnSubExpr(
        multi_expr=multi,
        output_name="year",
        dtype=None,
    )
    named = NamedExpr(name="year", expr=sub)
    bound = named.bind(dtypes)
    assert isinstance(bound, NamedExpr)
    assert bound.name == "year"
    assert isinstance(bound.expr, LiteFrameMultiColumnSubExpr)
    assert bound.dtype == pd.Int32Dtype()


# ---------------------------------------------------------------------------
# Select / col integration tests
# ---------------------------------------------------------------------------


def test_select_with_col_expr():
    lf = LF({"a": [1, 2, 3], "b": [4, 5, 6]})
    result = lf.select(col("a"))
    assert isinstance(result, LITEFRAME_TYPE)
    assert list(result.columns) == ["a"]

    op = result.data.op
    assert isinstance(op, LiteFrameProjection)
    assert len(op.projections) == 1
    assert isinstance(op.projections[0], LiteFrameColumn)
    assert op.projections[0].name == "a"
    assert op.projections[0].dtype is not None


def test_select_with_composed_expr():
    lf = LF({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
    result = lf.select(col("a") + col("b"))
    assert isinstance(result, LITEFRAME_TYPE)
    assert len(result.columns) == 1

    op = result.data.op
    assert isinstance(op, LiteFrameProjection)
    assert len(op.projections) == 1
    assert isinstance(op.projections[0], NamedExpr)
    assert isinstance(op.projections[0].expr, LiteFrameAdd)
    assert op.projections[0].expr.dtype is not None


def test_select_multiple_exprs():
    lf = LF({"a": [1, 2, 3], "b": [4, 5, 6]})
    result = lf.select(col("a"), col("b"))
    assert isinstance(result, LITEFRAME_TYPE)
    assert list(result.columns) == ["a", "b"]

    op = result.data.op
    assert isinstance(op, LiteFrameProjection)
    assert len(op.projections) == 2


def test_select_with_string_columns_still_works():
    lf = LF({"a": [1, 2, 3], "b": [4, 5, 6]})
    result = lf.select(["a", "b"])
    assert isinstance(result, LITEFRAME_TYPE)
    assert list(result.columns) == ["a", "b"]


def test_select_expr_missing_column_raises_keyerror():
    lf = LF({"a": [1, 2, 3]})
    with pytest.raises(KeyError):
        lf.select(col("z"))


def test_col_case_insensitive():
    """col() normalizes names to lowercase for case-insensitive access."""
    lf = LF({"a": [1, 2, 3], "b": [4, 5, 6]})
    # Access using uppercase names should work
    result = lf.select(col("A"), col("B"))
    assert list(result.columns) == ["a", "b"]

    # Mixed case
    result2 = lf.select(col("a"))
    assert list(result2.columns) == ["a"]


def test_rename_normalizes_name():
    """rename() normalizes both old and new names."""
    lf = LF({"a": [1, 2, 3], "b": [4, 5, 6]})
    # Rename using uppercase key/value
    result = lf.rename(columns={"A": "X", "B": "Y"})
    assert list(result.columns) == ["x", "y"]


def test_bind_expr_case_insensitive_lookup():
    """bind() looks up columns in a lowercased dtypes index."""
    dtypes = pd.Series({"a": wrap_arrow_dtype(pa.int64())})
    # col("A") normalizes to "a", which is found in the dtypes index
    c = col("A")
    bound = c.bind(dtypes)
    assert bound.name == "a"
    assert bound.dtype == wrap_arrow_dtype(pa.int64())


# ---------------------------------------------------------------------------
# Scalar literal dtype inference in binary ops (int + float promotion)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "expr_func,expected_literal_dtype,expected_result_dtype",
    [
        # int64 column + float scalar -> float64
        (lambda a: a + 1.5, pa.float64(), pa.float64()),
        # float scalar + int64 column -> float64
        (lambda a: 1.5 + a, pa.float64(), pa.float64()),
        # int64 column + int scalar -> int64
        (lambda a: a + 1, pa.int64(), pa.int64()),
        # int scalar + int64 column -> int64
        (lambda a: 1 + a, pa.int64(), pa.int64()),
    ],
)
def test_scalar_literal_dtype_inferred_from_value(
    expr_func, expected_literal_dtype, expected_result_dtype
):
    """Scalar literals in expression-level binary ops get dtype from value, not column."""
    a = col("A")
    result = expr_func(a)
    assert result.dtype is None  # unbound, dtype not yet resolved


def test_int_col_plus_float_literal_promotes_to_float64():
    """int64 + 1.5 must infer float64 result, not int64."""
    dtypes = pd.Series({"a": wrap_arrow_dtype(pa.int64())})
    expr = col("A") + 1.5
    bound = expr.bind(dtypes)
    assert isinstance(bound, LiteFrameAdd)
    assert bound.left.dtype == wrap_arrow_dtype(pa.int64())
    assert bound.right.dtype == wrap_arrow_dtype(pa.float64())
    assert bound.dtype == wrap_arrow_dtype(pa.float64())


def test_float_literal_plus_int_col_promotes_to_float64():
    """1.5 + int64 must infer float64 result, not int64."""
    dtypes = pd.Series({"a": wrap_arrow_dtype(pa.int64())})
    expr = 1.5 + col("A")
    bound = expr.bind(dtypes)
    assert isinstance(bound, LiteFrameAdd)
    assert bound.left.dtype == wrap_arrow_dtype(pa.float64())
    assert bound.right.dtype == wrap_arrow_dtype(pa.int64())
    assert bound.dtype == wrap_arrow_dtype(pa.float64())


# ---------------------------------------------------------------------------
# Composed column-level expressions
# ---------------------------------------------------------------------------


def test_composed_col_expr_tree_structure():
    """((col("a") + col("b")) / col("a")) > col("b") produces correct tree."""
    a, b = col("a"), col("b")
    expr = ((a + b) / a) > b

    # Outermost: LiteFrameGt
    assert isinstance(expr, LiteFrameGt)
    assert isinstance(expr.right, LiteFrameColumn)
    assert expr.right.name == "b"

    # Left of Gt: LiteFrameDiv
    div = expr.left
    assert isinstance(div, LiteFrameDiv)
    assert isinstance(div.right, LiteFrameColumn)
    assert div.right.name == "a"

    # Left of Div: LiteFrameAdd
    add = div.left
    assert isinstance(add, LiteFrameAdd)
    assert isinstance(add.left, LiteFrameColumn)
    assert add.left.name == "a"
    assert isinstance(add.right, LiteFrameColumn)
    assert add.right.name == "b"


# ---------------------------------------------------------------------------
# Bitwise shift expressions
# ---------------------------------------------------------------------------

_SHIFT_UNBOUND = [
    # (name, op_fn, expr_cls, literal_side, literal_value)
    ("lshift", lambda a: a << 2, LiteFrameLShift, "right", 2),
    ("rshift", lambda a: a >> 1, LiteFrameRShift, "right", 1),
    ("rlshift", lambda a: 2 << a, LiteFrameLShift, "left", 2),
    ("rrshift", lambda a: 16 >> a, LiteFrameRShift, "left", 16),
]


@pytest.mark.parametrize(
    "name,op_fn,expr_cls,literal_side,value",
    _SHIFT_UNBOUND,
    ids=[t[0] for t in _SHIFT_UNBOUND],
)
def test_shift_unbound_returns_shift_expr(name, op_fn, expr_cls, literal_side, value):
    a = col("A")
    result = op_fn(a)
    assert isinstance(result, expr_cls)
    assert result.dtype is None
    lit = getattr(result, literal_side)
    assert isinstance(lit, LiteFrameLiteral)
    assert lit.value == value


_SHIFT_BOUND = [
    ("lshift", lambda a: a << 2, LiteFrameLShift),
    ("rshift", lambda a: a >> 1, LiteFrameRShift),
]


@pytest.mark.parametrize(
    "name,op_fn,expr_cls", _SHIFT_BOUND, ids=[t[0] for t in _SHIFT_BOUND]
)
def test_shift_bound_infers_int_dtype(name, op_fn, expr_cls):
    dtypes = pd.Series({"a": wrap_arrow_dtype(pa.int64())})
    bound = op_fn(col("A")).bind(dtypes)
    assert isinstance(bound, expr_cls)
    assert bound.left.dtype == wrap_arrow_dtype(pa.int64())
    assert bound.right.dtype == wrap_arrow_dtype(pa.int64())
    # Shift preserves the left operand's integer dtype.
    assert bound.dtype == wrap_arrow_dtype(pa.int64())


# ---------------------------------------------------------------------------
# Bitwise And/Or/Xor/Invert are dtype-aware (int preserves, bool -> bool)
# ---------------------------------------------------------------------------

_BITWISE_DTYPE = [
    # (name, op_fn, input_pa_dtype, expected_pa_dtype)
    ("int_and", lambda a: a & 3, pa.int64(), pa.int64()),
    ("bool_and", lambda a: a & True, pa.bool_(), pa.bool_()),
    ("int_invert", lambda a: ~a, pa.int64(), pa.int64()),
    ("bool_invert", lambda a: ~a, pa.bool_(), pa.bool_()),
]


@pytest.mark.parametrize(
    "name,op_fn,input_pa,expected_pa",
    _BITWISE_DTYPE,
    ids=[t[0] for t in _BITWISE_DTYPE],
)
def test_bitwise_dtype_inference(name, op_fn, input_pa, expected_pa):
    """Bitwise &, |, ^, ~ are dtype-aware: bool -> bool, int -> int."""
    dtypes = pd.Series({"a": wrap_arrow_dtype(input_pa)})
    bound = op_fn(col("A")).bind(dtypes)
    assert bound.dtype == wrap_arrow_dtype(expected_pa)


# LiteFrameUDFExpr tests
# ---------------------------------------------------------------------------


def test_udf_expr_map_mode():
    """LiteFrameUDFExpr stores func, mode, and fields correctly."""
    func = lambda x: x * 2
    expr = LiteFrameUDFExpr(
        func=func,
        mode="map",
        operands=[LiteFrameColumn(name="a", dtype=None)],
        output_names=["a"],
        dtypes={"a": wrap_arrow_dtype(pa.int64())},
        input_frame_type=None,
    )
    assert expr.mode == "map"
    assert expr.input_frame_type is None
    assert expr.check_output_dtypes == "raises"


def test_udf_expr_bind_preserves_fields():
    """.bind() on LiteFrameUDFExpr preserves all UDF-specific fields."""
    func = lambda x: x + 1
    dtypes = pd.Series({"a": wrap_arrow_dtype(pa.int64())})
    expr = LiteFrameUDFExpr(
        func=func,
        mode="transform_chunk",
        operands=[LiteFrameColumn(name="a", dtype=None)],
        output_names=["a"],
        dtypes={"a": wrap_arrow_dtype(pa.int64())},
        input_frame_type="pandas",
        args=(1,),
        kwargs={"scale": 2},
    )
    bound = expr.bind(dtypes)
    assert isinstance(bound, LiteFrameUDFExpr)
    assert bound.mode == "transform_chunk"
    assert bound.input_frame_type == "pandas"
    assert bound.args == (1,)
    assert bound.kwargs == {"scale": 2}
    assert bound.operands[0].dtype == wrap_arrow_dtype(pa.int64())  # dtype resolved


def test_base_expr_has_custom_code_default():
    """LiteFrameBaseExpr and subclasses default to has_custom_code=False."""
    col_expr = LiteFrameColumn(name="a", dtype=None)
    assert col_expr.has_custom_code is False


def test_udf_expr_has_custom_code():
    """LiteFrameUDFExpr.has_custom_code returns True for regular functions."""
    expr = LiteFrameUDFExpr(
        func=lambda x: x * 2,
        mode="map",
        operands=[LiteFrameColumn(name="a", dtype=None)],
        output_names=["a"],
        dtypes={"a": wrap_arrow_dtype(pa.int64())},
    )
    assert expr.has_custom_code is True


def _identity(x):
    return x


def test_udf_expr_has_custom_code_builtin():
    """LiteFrameUDFExpr.has_custom_code returns False for BuiltinFunction."""
    bf = BuiltinFunction(_identity)
    expr = LiteFrameUDFExpr(
        func=bf,
        mode="map",
        operands=[LiteFrameColumn(name="a", dtype=None)],
        output_names=["a"],
        dtypes={"a": wrap_arrow_dtype(pa.int64())},
    )
    assert expr.has_custom_code is False


def test_named_expr_delegates_has_custom_code():
    """NamedExpr delegates has_custom_code to its inner expr."""
    udf_expr = LiteFrameUDFExpr(
        func=lambda x: x,
        mode="map",
        operands=[LiteFrameColumn(name="a", dtype=None)],
        output_names=["a"],
        dtypes={"a": wrap_arrow_dtype(pa.int64())},
    )
    named = NamedExpr(
        name="a",
        expr=udf_expr,
        dtype=wrap_arrow_dtype(pa.int64()),
    )
    assert named.has_custom_code is True

    # Plain column expression
    named_plain = NamedExpr(
        name="a",
        expr=LiteFrameColumn(name="a", dtype=None),
        dtype=wrap_arrow_dtype(pa.int64()),
    )
    assert named_plain.has_custom_code is False


def test_multi_column_sub_expr_delegates_has_custom_code():
    """LiteFrameMultiColumnSubExpr delegates to multi_expr."""
    udf_expr = LiteFrameUDFExpr(
        func=lambda x: x,
        mode="transform_chunk",
        operands=[LiteFrameColumn(name="a", dtype=None)],
        output_names=["a"],
        dtypes={"a": wrap_arrow_dtype(pa.int64())},
    )
    sub_expr = LiteFrameMultiColumnSubExpr(
        multi_expr=udf_expr,
        output_name="a",
        dtype=wrap_arrow_dtype(pa.int64()),
    )
    assert sub_expr.has_custom_code is True


def test_projection_has_custom_code_with_udf():
    """LiteFrameProjection.has_custom_code() returns True when a UDF projection is present."""
    udf_expr = LiteFrameUDFExpr(
        func=lambda x: x,
        mode="map",
        operands=[LiteFrameColumn(name="a", dtype=None)],
        output_names=["a"],
        dtypes={"a": wrap_arrow_dtype(pa.int64())},
    )
    named = NamedExpr(
        name="a",
        expr=udf_expr,
        dtype=wrap_arrow_dtype(pa.int64()),
    )
    proj_op = LiteFrameProjection(projections=[named])
    assert proj_op.has_custom_code() is True


def test_projection_has_custom_code_without_udf():
    """LiteFrameProjection.has_custom_code() returns False for plain column projections."""
    col_expr = LiteFrameColumn(name="a", dtype=wrap_arrow_dtype(pa.int64()))
    proj_op = LiteFrameProjection(projections=[col_expr])
    assert proj_op.has_custom_code() is False


# ---------------------------------------------------------------------------
# bind() preserves accessor-inferred dtypes (regression: .dt/.str accessor
# dtypes were overwritten by the operand dtype via _infer_unary_result_dtype)
# ---------------------------------------------------------------------------


def _dt_lf():
    return LF(pd.DataFrame({"dt": pd.to_datetime(["2023-01-31", "2023-04-01"])}))


def _str_lf():
    return LF({"s": ["hello", "world"]}, dtype=wrap_arrow_dtype(pa.string()))


@pytest.mark.parametrize(
    "lf_factory,accessor_fn,expected_dtype",
    [
        (_dt_lf, lambda lf: lf["dt"].dt.year, wrap_arrow_dtype(pa.int32())),
        (_dt_lf, lambda lf: lf["dt"].dt.month, wrap_arrow_dtype(pa.int32())),
        (_dt_lf, lambda lf: lf["dt"].dt.day, wrap_arrow_dtype(pa.int32())),
        (_dt_lf, lambda lf: lf["dt"].dt.dayofweek, wrap_arrow_dtype(pa.int32())),
        (_dt_lf, lambda lf: lf["dt"].dt.quarter, wrap_arrow_dtype(pa.int32())),
        (_dt_lf, lambda lf: lf["dt"].dt.is_month_start, wrap_arrow_dtype(pa.bool_())),
        (
            _dt_lf,
            lambda lf: lf["dt"].dt.strftime("%Y-%m-%d"),
            wrap_arrow_dtype(pa.string()),
        ),
        (_str_lf, lambda lf: lf["s"].str.len(), wrap_arrow_dtype(pa.int64())),
        (_str_lf, lambda lf: lf["s"].str.contains("o"), wrap_arrow_dtype(pa.bool_())),
        (_str_lf, lambda lf: lf["s"].str.upper(), wrap_arrow_dtype(pa.string())),
    ],
    ids=[
        "dt.year",
        "dt.month",
        "dt.day",
        "dt.dayofweek",
        "dt.quarter",
        "dt.is_month_start",
        "dt.strftime",
        "str.len",
        "str.contains",
        "str.upper",
    ],
)
def test_bind_preserves_accessor_inferred_dtype(
    lf_factory, accessor_fn, expected_dtype
):
    """bind() must not overwrite a dtype already inferred by an accessor."""
    lf = lf_factory()
    proj = accessor_fn(lf).op.projections[0]
    bound = proj.bind(lf.dtypes)
    inner = bound.expr if isinstance(bound, NamedExpr) else bound
    assert inner.dtype == expected_dtype
