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

import datetime as dt

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from maxframe.codegen.spe.liteframe.expressions import LiteFrameExprEvaluator
from maxframe.liteframe.core import LiteFrame
from maxframe.liteframe.expressions import (
    LiteFrameAdd,
    LiteFrameAnd,
    LiteFrameAstype,
    LiteFrameColumn,
    LiteFrameEq,
    LiteFrameExpr,
    LiteFrameGt,
    LiteFrameInvert,
    LiteFrameLiteral,
    LiteFrameLt,
    LiteFrameMul,
    LiteFrameNe,
    LiteFrameOr,
    NamedExpr,
    SourceCompatibilityError,
    col,
)
from maxframe.liteframe.initializer import LiteFrame as LF
from maxframe.liteframe.operators.filter import LiteFrameFilter
from maxframe.liteframe.operators.indexing import (
    LiteFrameIlocGetItem,
    LiteFrameIlocSetItem,
)
from maxframe.liteframe.operators.project import LiteFrameProjection
from maxframe.utils import is_arrow_dtype_supported, wrap_arrow_dtype


def test_getitem_single_column():
    lf = LF({"a": [1, 2, 3], "b": [4, 5, 6]})
    result = lf["a"]
    assert isinstance(result, LiteFrame)
    assert list(result.columns) == ["a"]

    # Verify operator
    op = result.data.op
    assert isinstance(op, LiteFrameProjection)
    assert len(op.projections) == 1
    assert isinstance(op.projections[0], LiteFrameColumn)
    assert op.projections[0].name == "a"


def test_getitem_multiple_columns():
    lf = LF({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
    result = lf[["a", "c"]]
    assert isinstance(result, LiteFrame)
    assert list(result.columns) == ["a", "c"]

    # Verify operator
    op = result.data.op
    assert isinstance(op, LiteFrameProjection)
    assert len(op.projections) == 2
    assert all(isinstance(p, LiteFrameColumn) for p in op.projections)
    assert [p.name for p in op.projections] == ["a", "c"]


def test_select():
    lf = LF({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
    result = lf.select(["a", "c"])
    assert isinstance(result, LiteFrame)
    assert list(result.columns) == ["a", "c"]

    # Verify operator
    op = result.data.op
    assert isinstance(op, LiteFrameProjection)
    assert len(op.projections) == 2
    assert all(isinstance(p, LiteFrameColumn) for p in op.projections)
    assert [p.name for p in op.projections] == ["a", "c"]


def test_drop():
    lf = LF({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
    result = lf.drop(["b"])
    assert isinstance(result, LiteFrame)
    assert list(result.columns) == ["a", "c"]

    # Verify operator (drop uses select internally)
    op = result.data.op
    assert isinstance(op, LiteFrameProjection)
    assert len(op.projections) == 2
    assert [p.name for p in op.projections] == ["a", "c"]


def test_rename():
    lf = LF({"a": [1, 2, 3], "b": [4, 5, 6]})
    result = lf.rename(columns={"a": "x"})
    assert isinstance(result, LiteFrame)
    assert list(result.columns) == ["x", "b"]

    # Verify operator
    op = result.data.op
    assert isinstance(op, LiteFrameProjection)
    assert len(op.projections) == 2

    # First projection should be NamedExpr for renamed column
    assert isinstance(op.projections[0], NamedExpr)
    assert op.projections[0].name == "x"
    assert isinstance(op.projections[0].expr, LiteFrameColumn)
    assert op.projections[0].expr.name == "a"

    # Second projection should be LiteFrameColumn for unchanged column
    assert isinstance(op.projections[1], LiteFrameColumn)
    assert op.projections[1].name == "b"


def test_assign():
    lf = LF({"a": [1, 2, 3]})
    result = lf.assign(b=lambda x: x["a"])
    assert isinstance(result, LiteFrame)
    assert "b" in list(result.columns)

    # Verify operator
    op = result.data.op
    assert isinstance(op, LiteFrameProjection)
    assert len(op.projections) == 2

    # First projection should be existing column
    assert isinstance(op.projections[0], LiteFrameColumn)
    assert op.projections[0].name == "a"

    # Second projection should be NamedExpr for new column
    assert isinstance(op.projections[1], NamedExpr)
    assert op.projections[1].name == "b"


def test_setitem():
    lf = LF({"a": [1, 2, 3]})
    lf["b"] = lf["a"]
    assert isinstance(lf, LiteFrame)
    assert "b" in list(lf.columns)

    # Verify operator
    op = lf.data.op
    assert isinstance(op, LiteFrameProjection)
    assert len(op.projections) == 2

    # First projection should be existing column
    assert isinstance(op.projections[0], LiteFrameColumn)
    assert op.projections[0].name == "a"

    # Second projection should be NamedExpr for new column
    assert isinstance(op.projections[1], NamedExpr)
    assert op.projections[1].name == "b"


def test_setitem_preserves_expression():
    """lf["c"] = lf["a"] + 1 preserves the arithmetic expression graph."""
    lf = LF({"a": [1, 2, 3]})
    lf["c"] = lf["a"] + 1

    op = lf.data.op
    assert isinstance(op, LiteFrameProjection)
    assert len(op.projections) == 2

    # The new column should be a NamedExpr wrapping an Add expression,
    # not a plain column reference.
    named = op.projections[1]
    assert isinstance(named, NamedExpr)
    assert named.name == "c"
    assert isinstance(named.expr, LiteFrameAdd)
    assert isinstance(named.expr.left, LiteFrameColumn)
    assert named.expr.left.name == "a"
    assert isinstance(named.expr.right, LiteFrameLiteral)


def _count_node_types(expr, type_):
    """Count nodes of ``type_`` in an expression tree (cycle-safe)."""
    seen = set()
    stack = [expr]
    n = 0
    while stack:
        e = stack.pop()
        if id(e) in seen:
            continue
        seen.add(id(e))
        if isinstance(e, type_):
            n += 1
        for attr in (
            "operand",
            "left",
            "right",
            "condition",
            "true_value",
            "false_value",
            "expr",
            "multi_expr",
        ):
            v = getattr(e, attr, None)
            if v is not None and hasattr(v, "_op_type_"):
                stack.append(v)
        stack.extend(
            v for v in (getattr(e, "operands", None) or []) if hasattr(v, "_op_type_")
        )
    return n


def test_setitem_self_derived_column_not_double_inlined():
    """A self-derived column must not be re-inlined into a later expression.

    Regression (e2e test_yunqi_demo): ``lf["price"] = lf["price"].str...
    .astype(float)`` then ``lf["log_price"] = np.log(lf["price"])`` fused the
    price transform into ``log_price`` twice
    (``astype(astype(col("price")))``), crashing with
    ``InvalidOperationError: expected String type, got: f64``.
    """
    lf = LF({"price": ["$1", "$2", "$3"]})
    lf["price"] = lf["price"].str.replace(r"[\$]", "", regex=True).astype(float)
    lf["log_price"] = np.log(lf["price"])

    op = lf.data.op
    log_named = next(p for p in op.projections if p.name == "log_price")
    # Single Astype in log_price => the price transform inlined once, not twice
    assert _count_node_types(log_named.expr, LiteFrameAstype) == 1


# Case-insensitivity tests


def test_getitem_case_insensitive():
    """getitem normalizes column names for case-insensitive access."""
    lf = LF({"a": [1, 2, 3], "b": [4, 5, 6]})
    result = lf["A"]
    assert isinstance(result, LiteFrame)
    assert list(result.columns) == ["a"]

    # Multiple columns with mixed case
    result2 = lf[["A", "B"]]
    assert list(result2.columns) == ["a", "b"]


def test_select_case_insensitive():
    """select normalizes column names for case-insensitive access."""
    lf = LF({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
    result = lf.select(["A", "C"])
    assert isinstance(result, LiteFrame)
    assert list(result.columns) == ["a", "c"]


def test_drop_case_insensitive():
    """drop normalizes column names for case-insensitive access."""
    lf = LF({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
    result = lf.drop(["B"])
    assert list(result.columns) == ["a", "c"]


def test_rename_case_insensitive():
    """rename normalizes both old and new column names."""
    lf = LF({"a": [1, 2, 3], "b": [4, 5, 6]})
    result = lf.rename(columns={"A": "X"})
    assert list(result.columns) == ["x", "b"]


def test_assign_overwrite_existing_column():
    """assign with an existing column name replaces it, no duplicate projections."""
    lf = LF({"a": [1, 2, 3], "b": [4, 5, 6]})
    result = lf.assign(b=lambda x: x["a"])

    assert isinstance(result, LiteFrame)
    assert list(result.columns) == ["a", "b"]

    # Should have exactly 2 projections, not 3 (no duplicate 'b')
    op = result.data.op
    assert isinstance(op, LiteFrameProjection)
    assert len(op.projections) == 2

    # First projection: existing column 'a'
    assert isinstance(op.projections[0], LiteFrameColumn)
    assert op.projections[0].name == "a"

    # Second projection: NamedExpr for overwritten column 'b'
    assert isinstance(op.projections[1], NamedExpr)
    assert op.projections[1].name == "b"

    # Output dtypes should have no duplicate index
    assert len(result.dtypes.index) == len(set(result.dtypes.index))


def test_assign_overwrite_with_literal():
    """assign with an existing column name and literal value replaces it."""
    lf = LF({"a": [1, 2, 3], "b": [4, 5, 6]})
    result = lf.assign(a=10)

    op = result.data.op
    assert len(op.projections) == 2
    assert isinstance(op.projections[0], NamedExpr)
    assert op.projections[0].name == "a"
    assert isinstance(op.projections[1], LiteFrameColumn)
    assert op.projections[1].name == "b"


def test_assign_normalizes_name():
    """assign normalizes kwarg names to lowercase."""
    lf = LF({"a": [1, 2, 3]})
    result = lf.assign(B=lambda x: x["A"])
    assert "b" in list(result.columns)


def test_setitem_normalizes_key():
    """setitem normalizes the key to lowercase."""
    lf = LF({"a": [1, 2, 3]})
    lf["B"] = lf["A"]
    assert "b" in list(lf.columns)


@pytest.mark.parametrize(
    "n_rows,idx,expected_nrows",
    [
        (4, slice(1, 3), 2),
        (3, 0, 1),
        (5, slice(None, None, 2), 3),
        (3, -1, 1),
        (5, slice(None, None, -1), 5),
    ],
    ids=["slice", "int", "step", "negative_int", "negative_step"],
)
def test_iloc_getitem(n_rows, idx, expected_nrows):
    lf = LF({"a": list(range(n_rows)), "b": list(range(n_rows, n_rows * 2))})
    result = lf.iloc[idx]
    assert isinstance(result, LiteFrame)
    assert result.shape[0] == expected_nrows
    assert list(result.columns) == list(lf.columns)
    assert isinstance(result.data.op, LiteFrameIlocGetItem)


def test_iloc_col_and_row():
    lf = LF({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
    result = lf.iloc[1:3, [0, 2]]
    assert isinstance(result, LiteFrame)
    assert result.shape[0] == 2
    assert list(result.columns) == ["a", "c"]


def test_iloc_no_op():
    lf = LF({"a": [1, 2, 3], "b": [4, 5, 6]})
    result = lf.iloc[:, :]
    assert result is lf


def test_head():
    lf = LF({"a": [1, 2, 3, 4, 5], "b": [6, 7, 8, 9, 10]})
    result = lf.head(3)
    assert isinstance(result, LiteFrame)
    assert result.shape[0] == 3
    assert isinstance(result.data.op, LiteFrameIlocGetItem)


def test_head_default():
    lf = LF({"a": list(range(10)), "b": list(range(10, 20))})
    result = lf.head()
    assert isinstance(result, LiteFrame)
    assert result.shape[0] == 5


# Boolean indexing tests


@pytest.mark.parametrize(
    "mask_func,expected_pred_type",
    [
        (lambda lf: lf["a"] == 2, LiteFrameEq),
        (lambda lf: lf["a"] != 2, LiteFrameNe),
        (lambda lf: lf["a"] > 1, LiteFrameGt),
        (lambda lf: lf["a"] >= 1, LiteFrameExpr),  # Ge, checked by expr class below
        (lambda lf: lf["a"] < 3, LiteFrameLt),
        (lambda lf: lf["a"] <= 3, LiteFrameExpr),  # Le, checked by expr class below
    ],
    ids=["eq", "ne", "gt", "ge", "lt", "le"],
)
def test_getitem_boolean_mask_comparison(mask_func, expected_pred_type):
    """lf[lf["col"] <op> value] uses expression-based filter."""
    lf = LF({"a": [1, 2, 3], "b": [4, 5, 6]})
    mask = mask_func(lf)
    assert isinstance(mask, LiteFrame)

    result = lf[mask]
    assert isinstance(result, LiteFrame)
    assert list(result.columns) == ["a", "b"]

    # Verify it uses expression-based filter
    op = result.data.op
    assert isinstance(op, LiteFrameFilter)
    assert isinstance(op.predicate, LiteFrameExpr)


def test_getitem_boolean_mask_negation():
    """lf[~(lf["col"] == value)] uses expression-based filter with invert."""
    lf = LF({"a": [1, 2, 3], "b": [4, 5, 6]})
    mask = lf["a"] == 2
    neg_mask = ~mask
    assert isinstance(neg_mask, LiteFrame)

    result = lf[neg_mask]
    assert isinstance(result, LiteFrame)

    op = result.data.op
    assert isinstance(op, LiteFrameFilter)
    assert isinstance(op.predicate, LiteFrameInvert)


@pytest.mark.parametrize(
    ("data", "expect_invert_predicate"),
    [
        ([0, 1, 2], False),  # ~int_col -> integer bitwise NOT, not fused
        ([True, False, True], True),  # ~bool_col -> bool, fused as LiteFrameInvert
    ],
    ids=["int_not_fused", "bool_fused"],
)
def test_getitem_invert_column_predicate_fusion(data, expect_invert_predicate):
    """lf[~col] fuses as a LiteFrameInvert predicate only for boolean columns.

    ``~`` on an integer column is a dtype-aware bitwise NOT (integer result),
    so it must fall back to the boolean-mask path instead of feeding an
    integer expression into ``LiteFrameFilter``. On a boolean column it
    yields a boolean result and fuses as a ``LiteFrameInvert`` predicate.
    """
    lf = LF({"a": data})
    result = lf[~lf["a"]]
    assert isinstance(result, LiteFrame)

    op = result.data.op
    assert isinstance(op, LiteFrameFilter)
    assert isinstance(op.predicate, LiteFrameInvert) is expect_invert_predicate


@pytest.mark.parametrize(
    "combine_op,expected_pred_type",
    [
        (lambda m1, m2: m1 & m2, LiteFrameAnd),
        (lambda m1, m2: m1 | m2, LiteFrameOr),
    ],
    ids=["and", "or"],
)
def test_getitem_boolean_compound(combine_op, expected_pred_type):
    """lf[(lf["c1"] > 1) & (lf["c2"] < 6)] uses expression-based filter."""
    lf = LF({"a": [1, 2, 3], "b": [4, 5, 6]})
    mask1 = lf["a"] > 1
    mask2 = lf["b"] < 6
    combined = combine_op(mask1, mask2)
    assert isinstance(combined, LiteFrame)

    result = lf[combined]
    assert isinstance(result, LiteFrame)

    op = result.data.op
    assert isinstance(op, LiteFrameFilter)
    assert isinstance(op.predicate, expected_pred_type)


def test_getitem_cross_source_mask_raises():
    """lf[other_lf["a"] > 5] raises SourceCompatibilityError."""
    lf1 = LF({"a": [1, 2, 3]})
    lf2 = LF({"a": [4, 5, 6]})

    with pytest.raises(SourceCompatibilityError) as exc_info:
        lf1[lf2["a"] > 1]

    assert "different sources" in str(exc_info.value)


# getitem slice delegation tests


@pytest.mark.parametrize(
    "slc,expected_nrows",
    [(slice(1, 3), 2), (slice(1, None), 3), (slice(None, 2), 2)],
    ids=["1:3", "1:", ":2"],
)
def test_getitem_slice_delegates_to_iloc(slc, expected_nrows):
    """lf[slc] delegates to iloc, producing a LiteFrameIlocGetItem operator."""
    lf = LF({"a": [1, 2, 3, 4], "b": [5, 6, 7, 8]})
    result = lf[slc]
    assert isinstance(result, LiteFrame)
    assert result.shape[0] == expected_nrows
    assert isinstance(result.data.op, LiteFrameIlocGetItem)


@pytest.mark.parametrize(
    "slc,expected_nrows",
    [(slice(None), 4), (slice(None, None, 2), 2), (slice(None, None, -1), 4)],
    ids=[":", "::2", "::-1"],
)
def test_getitem_slice_no_integer_bounds_delegates_to_iloc(slc, expected_nrows):
    """lf[:], lf[::2], lf[::-1] delegate to iloc (positional, like pandas)."""
    lf = LF({"a": [1, 2, 3, 4], "b": [5, 6, 7, 8]})
    result = lf[slc]
    assert isinstance(result, LiteFrame)
    assert result.shape[0] == expected_nrows


def test_getitem_slice_all_returns_same_object():
    """lf[:] is a no-op that returns the same LiteFrame."""
    lf = LF({"a": [1, 2, 3, 4], "b": [5, 6, 7, 8]})
    result = lf[:]
    assert result is lf


def test_getitem_slice_string_bounds_raises():
    """lf['a':'e'] raises KeyError (label-based slicing not supported on range index)."""
    lf = LF({"a": [1, 2, 3, 4], "b": [5, 6, 7, 8]})
    with pytest.raises(KeyError, match="Unsupported slice key"):
        lf["a":"e"]


# iloc setitem tests


def test_iloc_setitem_scalar():
    """lf.iloc[0:2, 1] = 99 modifies the correct cells."""
    lf = LF({"a": [1, 2, 3], "b": [4, 5, 6]})
    lf.iloc[0:2, 1] = 99
    assert isinstance(lf, LiteFrame)
    assert isinstance(lf.data.op, LiteFrameIlocSetItem)
    assert lf.data.op.indexes[0] == slice(0, 2)
    assert lf.data.op.indexes[1] == 1
    assert lf.data.op.value == 99


def test_iloc_setitem_preserves_metadata():
    """lf.iloc setitem preserves range_columns metadata."""
    lf = LF({"a": [1, 2, 3], "b": [4, 5, 6]})
    original_metadata = lf.frame_metadata
    lf.iloc[0, 0] = 99
    assert lf.frame_metadata == original_metadata


def test_iloc_setitem_non_scalar_raises():
    """Non-scalar value raises NotImplementedError."""
    lf = LF({"a": [1, 2, 3]})
    with pytest.raises(NotImplementedError, match="Only scalar"):
        lf.iloc[0:2, 0] = [10, 20]


def test_setitem_slice_delegates_to_iloc():
    """lf[1:3] = value delegates to iloc setitem."""
    lf = LF({"a": [1, 2, 3, 4], "b": [5, 6, 7, 8]})
    lf[1:3] = 99
    assert isinstance(lf.data.op, LiteFrameIlocSetItem)
    assert lf.data.op.indexes[0] == slice(1, 3)
    assert lf.data.op.indexes[1] == slice(None)
    assert lf.data.op.value == 99


def test_iloc_getitem_invalid_column_indexer_raises():
    """lf.iloc[:, 'a'] raises ValueError instead of returning all columns."""
    lf = LF({"a": [1, 2, 3], "b": [4, 5, 6]})
    with pytest.raises(ValueError, match="Location based indexing"):
        lf.iloc[:, "a"]


def test_iloc_getitem_out_of_bounds_row_raises():
    """Out-of-bounds integer row raises IndexError."""
    lf = LF({"a": [1, 2, 3], "b": [4, 5, 6]})
    with pytest.raises(IndexError, match="out-of-bounds"):
        lf.iloc[5]


def test_iloc_getitem_out_of_bounds_col_raises():
    """Out-of-bounds integer column raises IndexError."""
    lf = LF({"a": [1, 2, 3], "b": [4, 5, 6]})
    with pytest.raises(IndexError, match="out-of-bounds"):
        lf.iloc[:, 5]


# Fusion tests


def test_select_after_arithmetic_fuses():
    """(lf + 5).select("a") should fuse to a single projection on root source."""
    lf = LF({"a": [1, 2, 3], "b": [4, 5, 6]})
    result = (lf + 5).select("a")

    op = result.data.op
    assert isinstance(op, LiteFrameProjection)
    expr = op.projections[0].expr
    assert isinstance(expr, LiteFrameAdd)
    assert isinstance(expr.left, LiteFrameColumn)
    assert expr.left.name == "a"
    assert isinstance(expr.right, LiteFrameLiteral)
    assert expr.right.value == 5

    # Input should be the root source
    assert result.data.inputs[0] is lf._data


def test_drop_after_arithmetic_fuses():
    """(lf + 5).drop("b") should fuse pass-through columns."""
    lf = LF({"a": [1, 2, 3], "b": [4, 5, 6]})
    result = (lf + 5).drop("b")

    op = result.data.op
    assert isinstance(op, LiteFrameProjection)
    # Column "a" should carry the Add expr from the input projection
    expr_a = op.projections[0].expr
    assert isinstance(expr_a, LiteFrameAdd)
    assert result.data.inputs[0] is lf._data


def test_rename_after_arithmetic_fuses():
    """(lf + 5).rename({"a": "x"}) should fuse pass-through columns."""
    lf = LF({"a": [1, 2, 3], "b": [4, 5, 6]})
    result = (lf + 5).rename({"a": "x"})

    op = result.data.op
    assert isinstance(op, LiteFrameProjection)
    # Column "x" should wrap the Add expr from the input projection
    proj_x = op.projections[0]
    assert isinstance(proj_x, NamedExpr)
    assert proj_x.name == "x"
    assert isinstance(proj_x.expr, LiteFrameAdd)
    # Column "b" should also carry the Add expr
    proj_b = op.projections[1]
    assert isinstance(proj_b.expr, LiteFrameAdd)
    assert result.data.inputs[0] is lf._data


def test_assign_after_arithmetic_fuses():
    """(lf + 5).assign(c=10) should fuse existing columns."""
    lf = LF({"a": [1, 2, 3], "b": [4, 5, 6]})
    result = (lf + 5).assign(c=10)

    op = result.data.op
    assert isinstance(op, LiteFrameProjection)
    # Columns "a" and "b" should carry the Add expr
    assert isinstance(op.projections[0].expr, LiteFrameAdd)
    assert isinstance(op.projections[1].expr, LiteFrameAdd)
    # Column "c" should be the new literal
    proj_c = op.projections[2]
    assert isinstance(proj_c, NamedExpr)
    assert isinstance(proj_c.expr, LiteFrameLiteral)
    assert result.data.inputs[0] is lf._data


def test_setitem_after_arithmetic_fuses():
    """(lf + 5)["c"] = 10 should fuse existing columns."""
    lf = LF({"a": [1, 2, 3], "b": [4, 5, 6]})
    result = lf + 5
    result["c"] = 10

    op = result.data.op
    assert isinstance(op, LiteFrameProjection)
    # Columns "a" and "b" should carry the Add expr
    assert isinstance(op.projections[0].expr, LiteFrameAdd)
    assert isinstance(op.projections[1].expr, LiteFrameAdd)
    # Column "c" should be the new literal
    proj_c = op.projections[2]
    assert isinstance(proj_c, NamedExpr)
    assert isinstance(proj_c.expr, LiteFrameLiteral)
    # Input should be the root source
    assert result.data.inputs[0] is lf._data


def test_filter_on_projected_source_uses_root_source():
    """source[source["a"] > 10] should filter root source, not the projection.

    Regression: when source = lf + 5, the fused mask predicate references
    root source columns.  The filter must be applied to the root source
    and the projection re-applied on top, otherwise the projection is
    double-applied.
    """
    lf = LF({"a": [4, 6]})
    source = lf + 5  # projection: Add(Col("a"), Lit(5))
    mask = source["a"] > 10  # fused: Gt(Add(Col("a"), Lit(5)), Lit(10))
    result = source[mask]

    # The filter should read from the root source (lf), not from source
    filter_op = result.data.op
    assert isinstance(filter_op, LiteFrameProjection)
    # The input to the projection should be a filter on the root source
    filter_tileable = result.data.inputs[0]
    assert isinstance(filter_tileable.op, LiteFrameFilter)
    assert filter_tileable.inputs[0] is lf._data


def test_select_expr_on_projection_rewrites_column_refs():
    """(lf + 5).select(col("a") * 2) should rewrite col("a") through the projection.

    Regression: without rewriting, col("a") references root source column "a"
    but the projection adds 5 to it.  The fused expression should be
    Mul(Add(Col("a"), Lit(5)), Lit(2)), not Mul(Col("a"), Lit(2)).
    """

    lf = LF({"a": [1, 2]})
    result = (lf + 5).select(col("a") * 2)

    op = result.data.op
    assert isinstance(op, LiteFrameProjection)
    expr = op.projections[0].expr
    # Should be Mul(Add(Col("a"), Lit(5)), Lit(2))
    assert isinstance(expr, LiteFrameMul)
    assert isinstance(expr.left, LiteFrameAdd)
    assert isinstance(expr.left.left, LiteFrameColumn)
    assert expr.left.left.name == "a"
    assert isinstance(expr.left.right, LiteFrameLiteral)
    assert expr.left.right.value == 5
    # Input should be root source
    assert result.data.inputs[0] is lf._data


def test_select_expr_rejects_duplicate_positional_name_collision():
    """select(col("1"), col("a") + 1) must reject the duplicate name "1".

    Regression: the positional fallback name ``str(i)`` for an anonymous
    expression can collide with an explicitly selected integer-string
    column.  ``col("1")`` and ``col("a") + 1`` (position 1 -> name "1")
    both resolve to ``"1"``, which would silently overwrite one entry in
    name-keyed metadata/SPE or emit duplicate Polars aliases in DPE.
    """
    lf = LF({"1": [10], "a": [1]})
    with pytest.raises(ValueError, match="Duplicate output column name '1'"):
        lf.select(col("1"), col("a") + 1)


def test_assign_expr_on_projection_rewrites_column_refs():
    """(lf + 5).assign(c=col("a") + 1) should rewrite col("a") through the projection.

    Regression: without rewriting, col("a") references root source column "a"
    but the projection adds 5.  The new column c should be Add(Add(Col("a"),
    Lit(5)), Lit(1)), not Add(Col("a"), Lit(1)).
    """

    lf = LF({"a": [1, 2]})
    result = (lf + 5).assign(c=col("a") + 1)

    op = result.data.op
    assert isinstance(op, LiteFrameProjection)
    # Column "a" should carry the Add expr
    assert isinstance(op.projections[0].expr, LiteFrameAdd)
    # Column "c" should be Add(Add(Col("a"), Lit(5)), Lit(1))
    proj_c = op.projections[1]
    assert isinstance(proj_c, NamedExpr)
    assert isinstance(proj_c.expr, LiteFrameAdd)
    assert isinstance(proj_c.expr.left, LiteFrameAdd)
    assert isinstance(proj_c.expr.left.left, LiteFrameColumn)
    assert proj_c.expr.left.left.name == "a"
    assert isinstance(proj_c.expr.left.right, LiteFrameLiteral)
    assert proj_c.expr.left.right.value == 5
    assert isinstance(proj_c.expr.right, LiteFrameLiteral)
    assert proj_c.expr.right.value == 1
    # Input should be root source
    assert result.data.inputs[0] is lf._data


def test_setitem_expr_on_projection_rewrites_column_refs():
    """(lf + 5)["c"] = col("a") + 1 should rewrite col("a") through the projection.

    Regression: without rewriting, col("a") references root source column "a"
    but the projection adds 5.
    """

    lf = LF({"a": [1, 2]})
    result = lf + 5
    result["c"] = col("a") + 1

    op = result.data.op
    assert isinstance(op, LiteFrameProjection)
    # Column "a" should carry the Add expr
    assert isinstance(op.projections[0].expr, LiteFrameAdd)
    # Column "c" should be Add(Add(Col("a"), Lit(5)), Lit(1))
    proj_c = op.projections[1]
    assert isinstance(proj_c, NamedExpr)
    assert isinstance(proj_c.expr, LiteFrameAdd)
    assert isinstance(proj_c.expr.left, LiteFrameAdd)
    assert isinstance(proj_c.expr.left.left, LiteFrameColumn)
    assert proj_c.expr.left.left.name == "a"
    # Input should be root source
    assert result.data.inputs[0] is lf._data


def test_select_after_map_preserves_udf_dtype():
    """select() after map() should fuse through UDF projections while
    preserving the output dtype on the wrapped NamedExpr."""
    lf = LF({"a": [1, 2, 3], "b": [4, 5, 6]})

    # map() produces NamedExpr(name=col, expr=LiteFrameUDFExpr, dtype=out_dtype)
    mapped = lf.map(lambda x: x * 2, dtype="int64")

    # select() should fuse through the UDF projection without losing dtype
    result = mapped.select("a")

    op = result.data.op
    assert isinstance(op, LiteFrameProjection)
    proj = op.projections[0]
    # The fused projection should carry the UDF inner expression
    assert proj.expr is not None
    # The dtype must be preserved (not None) so that LiteFrameProjection
    # can read it without failing
    assert proj.dtype is not None
    # The input should be the root source (fusion happened)
    assert result.data.inputs[0] is lf._data


def _make_mapped_lf():
    lf = LF({"a": [1, 2], "b": [3.0, 4.0]})
    return lf.map(lambda x: x + 1, dtype="int64", skip_infer=True)


@pytest.mark.parametrize(
    "value_expr, test_id",
    [
        pytest.param(
            lambda mapped: mapped.select("a"), "select_named_expr", id="from_select"
        ),
        pytest.param(lambda mapped: col("a"), "col_udf_expr", id="from_col"),
    ],
)
def test_assign_renames_bound_expr_to_target(value_expr, test_id):
    """assign(c=<expr>) must name the new column "c".

    Two related regression paths:
    - from_select: the bound NamedExpr from .select("a") carried its own name
      ("a"), so "c" was ignored and "a" was overwritten.
    - from_col: col("a") rewrites to a bare map-mode LiteFrameUDFExpr (no
      .rename()), which previously raised AttributeError; assign must still
      produce a column named "c".
    """
    mapped = _make_mapped_lf()
    result = mapped.assign(c=value_expr(mapped))
    assert isinstance(result, LiteFrame)
    assert list(result.columns) == ["a", "b", "c"]
    proj_c = result.data.op.projections[-1]
    assert isinstance(proj_c, NamedExpr)
    assert proj_c.name == "c"


@pytest.mark.parametrize(
    "method,affix,expected_cols",
    [
        ("add_prefix", "col_", ["col_a", "col_b"]),
        ("add_suffix", "_val", ["a_val", "b_val"]),
    ],
    ids=["prefix", "suffix"],
)
def test_add_prefix_suffix(method, affix, expected_cols):
    lf = LF({"a": [1, 2, 3], "b": [4, 5, 6]})
    result = getattr(lf, method)(affix)
    assert isinstance(result, LiteFrame)
    assert list(result.columns) == expected_cols


@pytest.mark.parametrize(
    "method,affix,expected",
    [
        ("add_prefix", "Col_", ["col_a"]),
        ("add_suffix", "_Val", ["a_val"]),
    ],
    ids=["prefix", "suffix"],
)
def test_add_prefix_suffix_case_insensitive(method, affix, expected):
    """Prefix/suffix is normalized to lowercase, matching rename."""
    lf = LF({"a": [1, 2, 3]})
    result = getattr(lf, method)(affix)
    assert list(result.columns) == expected


@pytest.mark.parametrize(
    "method,affix,cols",
    [
        ("add_prefix", "x_", ["x_a", "x_b"]),
        ("add_suffix", "_x", ["a_x", "b_x"]),
    ],
    ids=["prefix", "suffix"],
)
def test_add_prefix_suffix_preserves_dtypes(method, affix, cols):
    lf = LF({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
    result = getattr(lf, method)(affix)
    assert result.dtypes[cols[0]] == lf.dtypes["a"]
    assert result.dtypes[cols[1]] == lf.dtypes["b"]


def test_add_prefix_after_arithmetic_fuses():
    """add_prefix on an arithmetic result should fuse into one projection."""
    lf = LF({"a": [1, 2, 3], "b": [4, 5, 6]})
    result = (lf + 5).add_prefix("col_")
    op = result.data.op
    assert isinstance(op, LiteFrameProjection)
    assert result.data.inputs[0] is lf._data
    assert isinstance(op.projections[0], NamedExpr)
    assert isinstance(op.projections[0].expr, LiteFrameAdd)


@pytest.mark.parametrize(
    "op,expected",
    [
        pytest.param(lambda r: r["value"], ["value"], id="getitem_scalar"),
        pytest.param(lambda r: r[["a", "value"]], ["a", "value"], id="getitem_list"),
        pytest.param(
            lambda r: r.select(["a", "value"]), ["a", "value"], id="select_names"
        ),
        pytest.param(lambda r: r.select(col("value")), ["value"], id="select_expr"),
        pytest.param(lambda r: r.fillna({"a": 0}), ["a", "value"], id="fillna_dict"),
        pytest.param(
            lambda r: r.astype({"a": "float64"}), ["a", "value"], id="astype_dict"
        ),
        pytest.param(
            lambda r: r.assign(z=r["value"]), ["a", "value", "z"], id="assign"
        ),
    ],
)
def test_rename_alias_preserved(op, expected):
    """rename aliases survive subsequent projection operations (#86799306)."""
    r = LF({"a": [1, 2], "b": [3, 4]}).rename(columns={"b": "value"})
    assert list(op(r).columns) == expected


@pytest.mark.parametrize(
    "setup,expected",
    [
        # fused projection carries the new name, not the old
        pytest.param(
            lambda: [
                p.name
                for p in LF({"a": [1], "b": [2]})
                .rename(columns={"b": "value"})[["a", "value"]]
                .data.op.projections
            ],
            lambda names: "value" in names and "b" not in names,
            id="fused_projection_names",
        ),
        # untouched-alias: second rename renames "a"→"label", but "value"
        # (mismatched inner LiteFrameColumn "b") must be wrapped by the
        # _resolve_named_expr guard, not passed through bare.
        pytest.param(
            lambda: list(
                LF({"a": [1, 2], "b": [3, 4]})
                .rename(columns={"b": "value"})
                .rename(columns={"a": "label"})
                .columns
            ),
            lambda cols: cols == ["label", "value"],
            id="untouched_alias_in_chained_rename",
        ),
        # setitem after rename keeps renamed column intact
        pytest.param(
            lambda: (lambda r: (r.__setitem__("c", r["value"]), list(r.columns))[1])(
                LF({"a": [1], "b": [2]}).rename(columns={"b": "value"})
            ),
            lambda cols: cols == ["a", "value", "c"],
            id="setitem_keeps_alias",
        ),
        # setitem with rank() exercises the aligned-setitem path after rename
        pytest.param(
            lambda: (lambda r: (r.__setitem__("a", r["a"].rank()), list(r.columns))[1])(
                LF({"a": [1, 2], "b": [2, 1]}).rename(columns={"b": "value"})
            ),
            lambda cols: cols == ["value", "a"],
            id="setitem_rank",
        ),
    ],
)
def test_rename_alias_regression_oracles(setup, expected):
    """Branch-specific regression oracles for rename alias preservation (#86799306)."""
    assert expected(setup())


# ---------------------------------------------------------------------------
# tail / filter / select_dtypes / isin tests
# ---------------------------------------------------------------------------


def _eval_result(result, pdf):
    """Evaluate a LiteFrame projection result against *pdf* via the SPE evaluator."""
    evaluator = LiteFrameExprEvaluator(pdf)
    out = {}
    for proj in result.data.op.projections:
        expr = proj.expr if hasattr(proj, "expr") else proj
        out[proj.name] = evaluator.eval(expr)
    return pd.DataFrame(out)


@pytest.mark.parametrize(
    "n,expected_rows",
    [(3, 3), (5, 5), (0, 0), (10, None), (-1, 9), (-2, 8), (-10, 0), (-11, 0)],
    ids=[
        "tail_3",
        "tail_default",
        "tail_zero",
        "more_than_rows",
        "tail_negative_one",
        "tail_negative_two",
        "tail_negative_rows",
        "beyond_negative_rows",
    ],
)
def test_tail(n, expected_rows):
    """``tail(n)`` with negative n keeps all rows except the first |n| (pandas)."""
    result = LF({"a": list(range(10)), "b": list(range(10, 20))}).tail(n)
    assert isinstance(result, LiteFrame)
    if expected_rows is None:
        # Row count may be unknown (np.nan) when n exceeds the frame height.
        assert result.shape[0] is np.nan or result.shape[0] == 10
    else:
        assert result.shape[0] == expected_rows
    op = result.data.op
    assert isinstance(op, LiteFrameIlocGetItem)
    assert op.indexes[0] == (slice(0, 0) if n == 0 else slice(-n, None))
    assert op.indexes[1] == slice(None)


@pytest.mark.parametrize(
    "data,kwargs,expected",
    [
        pytest.param(
            {"aa": [1], "ab": [2], "cc": [3]},
            {"items": ["aa", "cc"]},
            ["aa", "cc"],
            id="items",
        ),
        pytest.param(
            {"aa": [1], "ab": [2], "cc": [3]}, {"like": "a"}, ["aa", "ab"], id="like"
        ),
        pytest.param(
            {"aa": [1], "ab": [2], "cc": [3]}, {"regex": "^a"}, ["aa", "ab"], id="regex"
        ),
        pytest.param({"a": [1], "b": [2]}, {"items": ["zzz"]}, [], id="no_matches"),
        pytest.param(
            {"a": [1], "b": [2]}, {"items": ["a", "zzz"]}, ["a"], id="unknown_ignored"
        ),
        pytest.param(
            {"a": [1], "b": [2], "c": [3]},
            {"items": ["c", "a"]},
            ["c", "a"],
            id="items_order",
        ),
    ],
)
def test_filter(data, kwargs, expected):
    pdf = pd.DataFrame(data)
    result = LF(pdf).filter(**kwargs)
    assert list(result.columns) == expected
    if expected:
        assert list(_eval_result(result, pdf).columns) == expected


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({}, id="no_selector"),
        pytest.param({"items": ["a"], "like": "a"}, id="two_selectors"),
    ],
)
def test_filter_raises(kwargs):
    with pytest.raises(TypeError):
        LF({"a": [1]}).filter(**kwargs)


@pytest.mark.parametrize(
    "kwargs,expected",
    [
        pytest.param({"include": "number"}, ["a", "b"], id="include_number"),
        pytest.param({"include": "int"}, ["a"], id="include_int"),
        pytest.param({"include": "float"}, ["b"], id="include_float"),
        pytest.param({"include": "integer"}, ["a"], id="integer"),
        pytest.param({"include": "floating"}, ["b"], id="floating"),
        pytest.param({"include": "datetime64"}, ["d"], id="datetime64"),
        pytest.param({"include": int}, ["a"], id="builtin_int"),
        pytest.param({"include": float}, ["b"], id="builtin_float"),
        pytest.param({"exclude": "number"}, ["c", "d"], id="exclude_number"),
        pytest.param({"include": ["int", "float"]}, ["a", "b"], id="include_list"),
        pytest.param(
            {"include": "number", "exclude": "int"}, ["b"], id="include_exclude"
        ),
        pytest.param({"include": "timedelta64"}, [], id="no_match"),
    ],
)
def test_select_dtypes(kwargs, expected):
    pdf = pd.DataFrame(
        {
            "a": [1, 2],
            "b": [1.0, 2.0],
            "c": ["x", "y"],
            "d": pd.to_datetime(["2020-01-01", "2020-01-02"]),
        }
    )
    result = LF(pdf).select_dtypes(**kwargs)
    assert list(result.columns) == expected
    if expected:
        assert list(_eval_result(result, pdf).columns) == expected


@pytest.mark.parametrize(
    "kwargs,exc",
    [
        pytest.param({}, ValueError, id="no_include_exclude"),
        pytest.param({"include": "bogus"}, TypeError, id="unknown_spec"),
        pytest.param(
            {"include": np.int64, "exclude": int}, ValueError, id="numpy_overlap"
        ),
        pytest.param(
            {"include": "number", "exclude": np.number}, ValueError, id="alias_overlap"
        ),
    ],
)
def test_select_dtypes_raises(kwargs, exc):
    with pytest.raises(exc):
        LF({"a": [1]}).select_dtypes(**kwargs)


_NUMPY_WIDTH_DF = pd.DataFrame(
    {"i32": np.array([1], dtype=np.int32), "i64": np.array([1], dtype=np.int64)}
)


@pytest.mark.parametrize(
    "pdf,include,expected",
    [
        pytest.param(
            pd.DataFrame({"a": [1, 2], "t": pd.to_timedelta([1, 2], unit="D")}),
            "int",
            ["a"],
            id="timedelta_not_integer",
        ),
        pytest.param(_NUMPY_WIDTH_DF, np.int32, ["i32"], id="exact_int32"),
        pytest.param(_NUMPY_WIDTH_DF, np.int64, ["i64"], id="exact_int64"),
    ],
)
def test_select_dtypes_dtype_specific(pdf, include, expected):
    """Alias/numpy specs match exactly what pandas matches."""
    assert list(LF(pdf).select_dtypes(include=include).columns) == expected


_TIME32_DF = pd.DataFrame(
    {
        "t": pd.Series([dt.time(0, 0, 1)], dtype=wrap_arrow_dtype(pa.time32("s"))),
        "n": [1],
    }
)
_TZ_DF = pd.DataFrame(
    {
        "d": pd.Series(
            [pd.Timestamp("2020-01-01", tz="UTC")],
            dtype=wrap_arrow_dtype(pa.timestamp("us", tz="UTC")),
        ),
        "n": [1],
    }
)


@pytest.mark.parametrize(
    "pdf,kwargs,expected",
    [
        pytest.param(_TIME32_DF, {"include": "object"}, ["t"], id="time_object"),
        pytest.param(
            _TIME32_DF, {"exclude": "object"}, ["n"], id="time_exclude_object"
        ),
        pytest.param(
            _TIME32_DF, {"include": "datetime64"}, [], id="time_not_datetime64"
        ),
    ],
)
def test_select_dtypes_time_family(pdf, kwargs, expected):
    """An Arrow time column probes as object (pandas has no numpy time dtype)."""
    assert list(LF(pdf).select_dtypes(**kwargs).columns) == expected


@pytest.mark.parametrize(
    "kwargs,expected_native,expected_legacy",
    [
        pytest.param({"include": "datetime64"}, ["d"], [], id="include_datetime64"),
        pytest.param(
            {"exclude": "datetime64"}, ["n"], ["d", "n"], id="exclude_datetime64"
        ),
        pytest.param({"include": "datetimetz"}, [], ["d"], id="include_datetimetz"),
        pytest.param({"include": "object"}, [], [], id="include_object"),
    ],
)
def test_select_dtypes_tz_aware_family(kwargs, expected_native, expected_legacy):
    """A tz-aware column keeps pandas' family: a native Arrow dtype reads as
    datetime64, a FakeArrowDtype (probed as ``pd.DatetimeTZDtype``) as datetimetz."""
    expected = expected_native if is_arrow_dtype_supported() else expected_legacy
    assert list(LF(_TZ_DF).select_dtypes(**kwargs).columns) == expected


def test_filter_items_listlike_and_dedup():
    """filter accepts pd.Index/np.ndarray and deduplicates (pandas)."""
    pdf = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
    assert list(LF(pdf).filter(items=pd.Index(["c", "a"])).columns) == ["c", "a"]
    assert list(LF(pdf).filter(items=np.array(["b", "a"])).columns) == ["b", "a"]
    assert list(LF(pdf).filter(items=["a", "a", "b"]).columns) == ["a", "b"]


@pytest.mark.parametrize(
    "data,values,expected_cols",
    [
        pytest.param({"a": [1, 2, 3], "b": [4, 5, 6]}, [1, 4], ["a", "b"], id="list"),
        pytest.param(
            {"a": [1, 2, 3], "b": [4, 5, 6]}, {"a": [1, 2]}, ["a", "b"], id="dict"
        ),
        pytest.param({"a": ["x", "y", "z"]}, "x", ["a"], id="single_string"),
    ],
)
def test_isin(data, values, expected_cols):
    pdf = pd.DataFrame(data)
    result = LF(pdf).isin(values)
    assert isinstance(result, LiteFrame)
    assert list(result.columns) == expected_cols
    assert all(result.dtypes[c].kind == "b" for c in result.dtypes.index)
    assert isinstance(result.data.op, LiteFrameProjection)
    evaluated = _eval_result(result, pdf)
    check_values = (
        values["a"]
        if isinstance(values, dict)
        else ([values] if isinstance(values, str) else values)
    )
    expected_a = pdf["a"].isin(check_values)
    assert evaluated["a"].tolist() == expected_a.tolist()


@pytest.mark.parametrize(
    "values",
    [
        pytest.param([[1, 2], [3]], id="nested_list"),
        pytest.param("tileable", id="tileable"),
        # pandas containers compare by index/column alignment; a plain
        # ``list(...)`` would silently test the Series' values or the
        # DataFrame's column labels instead.
        pytest.param(pd.Series([1, 2], index=["x", "y"]), id="pandas_series"),
        pytest.param(pd.DataFrame({"a": [1]}), id="pandas_frame"),
        pytest.param({"a": pd.Series([1], index=["absent"])}, id="dict_of_series"),
    ],
)
def test_isin_rejects_invalid_values(values):
    """isin rejects nested/tileable values and pandas Series/DataFrame inputs."""
    lf = LF({"a": [1, 2, 3]})
    with pytest.raises(NotImplementedError):
        lf.isin(lf if isinstance(values, str) else values)
