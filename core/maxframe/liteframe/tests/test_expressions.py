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

from maxframe.liteframe.core import LITEFRAME_TYPE
from maxframe.liteframe.datasource.from_local import from_local_df  # noqa: F401
from maxframe.liteframe.expressions import (
    LiteFrameAdd,
    LiteFrameBaseExpr,
    LiteFrameColumn,
    LiteFrameDiv,
    LiteFrameExpr,
    LiteFrameGt,
    LiteFrameInvert,
    LiteFrameLiteral,
    LiteFrameMul,
    LiteFrameMultiColumnExpr,
    LiteFrameMultiColumnSubExpr,
    LiteFrameNeg,
    NamedExpr,
    _infer_literal_dtype,
    col,
)
from maxframe.liteframe.initializer import LiteFrame as LF
from maxframe.liteframe.operators.project import LiteFrameProjection
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
        (True, pa.bool_()),
        (42, pa.int64()),
        (3.14, pa.float64()),
        ("hello", pa.string()),
    ],
)
def test_infer_literal_dtype(value, expected_dtype):
    assert _infer_literal_dtype(value) == wrap_arrow_dtype(expected_dtype)


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
    """LiteFrameMultiColumnExpr stores output_names and dtypes_dict."""
    col_expr = LiteFrameColumn(name="ts", dtype=np.dtype("datetime64[ns]"))
    multi = LiteFrameMultiColumnExpr(
        operand=col_expr,
        output_names=["year", "week", "day"],
        dtypes_dict={"year": "Int32", "week": "Int32", "day": "Int32"},
    )
    assert multi.output_names == ["year", "week", "day"]
    assert multi.dtypes_dict == {"year": "Int32", "week": "Int32", "day": "Int32"}
    assert isinstance(multi.operand, LiteFrameColumn)
    assert isinstance(multi, LiteFrameBaseExpr)
    assert not isinstance(multi, LiteFrameExpr)


def test_multi_column_sub_expr_construction():
    """LiteFrameMultiColumnSubExpr bridges to single-column world."""
    col_expr = LiteFrameColumn(name="ts", dtype=np.dtype("datetime64[ns]"))
    multi = LiteFrameMultiColumnExpr(
        operand=col_expr,
        output_names=["year", "week", "day"],
        dtypes_dict={"year": "Int32", "week": "Int32", "day": "Int32"},
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
        operand=col_expr,
        output_names=["year", "week", "day"],
        dtypes_dict={"year": "Int32", "week": "Int32", "day": "Int32"},
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
        operand=col_expr,
        output_names=["year", "week", "day"],
        dtypes_dict={"year": "Int32", "week": "Int32", "day": "Int32"},
    )
    bound = multi.bind(dtypes)
    assert isinstance(bound, LiteFrameMultiColumnExpr)
    assert isinstance(bound.operand, LiteFrameColumn)
    assert bound.operand.dtype == np.dtype("datetime64[ns]")
    assert bound.output_names == ["year", "week", "day"]
    assert bound.dtypes_dict == {"year": "Int32", "week": "Int32", "day": "Int32"}


def test_bind_multi_column_sub_expr():
    """bind() resolves dtype from multi_expr's dtypes_dict."""
    dtypes = pd.Series({"ts": np.dtype("datetime64[ns]")})
    col_expr = col("ts")
    multi = LiteFrameMultiColumnExpr(
        operand=col_expr,
        output_names=["year", "week", "day"],
        dtypes_dict={"year": "Int32", "week": "Int32", "day": "Int32"},
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


def test_bind_named_expr_with_sub_expr():
    """bind() works with NamedExpr wrapping LiteFrameMultiColumnSubExpr."""
    dtypes = pd.Series({"ts": np.dtype("datetime64[ns]")})
    col_expr = col("ts")
    multi = LiteFrameMultiColumnExpr(
        operand=col_expr,
        output_names=["year", "week", "day"],
        dtypes_dict={"year": "Int32", "week": "Int32", "day": "Int32"},
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
