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

import pytest

from maxframe.liteframe.core import LiteFrame
from maxframe.liteframe.expressions import (
    LiteFrameAdd,
    LiteFrameAnd,
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
)
from maxframe.liteframe.indexing.iloc import LiteFrameIlocGetItem, LiteFrameIlocSetItem
from maxframe.liteframe.initializer import LiteFrame as LF
from maxframe.liteframe.operators.filter import LiteFrameFilter
from maxframe.liteframe.operators.project import LiteFrameProjection


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
    from maxframe.liteframe.expressions import col

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


def test_assign_expr_on_projection_rewrites_column_refs():
    """(lf + 5).assign(c=col("a") + 1) should rewrite col("a") through the projection.

    Regression: without rewriting, col("a") references root source column "a"
    but the projection adds 5.  The new column c should be Add(Add(Col("a"),
    Lit(5)), Lit(1)), not Add(Col("a"), Lit(1)).
    """
    from maxframe.liteframe.expressions import col

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
    from maxframe.liteframe.expressions import col

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
