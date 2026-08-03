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

from maxframe.liteframe.core import RANGE_COL_NAME, LiteFrame
from maxframe.liteframe.expressions import (
    LiteFrameAdd,
    LiteFrameAnd,
    LiteFrameColumn,
    LiteFrameEq,
    LiteFrameFillNA,
    LiteFrameInvert,
    LiteFrameIsNA,
    LiteFrameLiteral,
    LiteFrameOr,
    LiteFrameWhere,
    NamedExpr,
)
from maxframe.liteframe.initializer import LiteFrame as LF
from maxframe.liteframe.operators.filter import LiteFrameFilter
from maxframe.liteframe.operators.project import LiteFrameProjection
from maxframe.liteframe.operators.source import LiteFrameReadODPSTable
from maxframe.protocol import DefaultIndexType
from maxframe.utils import wrap_arrow_dtype

bool_dtype = wrap_arrow_dtype(pa.bool_())


def test_isna():
    lf = LF({"a": [1, np.nan, 3]})
    result = lf.isna()
    assert isinstance(result, LiteFrame)

    # Verify operator structure: projection with LiteFrameIsNA per column
    op = result.data.op
    assert isinstance(op, LiteFrameProjection)
    assert len(op.projections) == 1

    proj = op.projections[0]
    assert isinstance(proj, NamedExpr)
    assert proj.name == "a"
    assert isinstance(proj.expr, LiteFrameIsNA)
    assert isinstance(proj.expr.operand, LiteFrameColumn)
    assert proj.expr.operand.name == "a"
    # Result dtype should be bool
    assert result.dtypes["a"] == bool_dtype


def test_notna():
    lf = LF({"a": [1, np.nan, 3]})
    result = lf.notna()
    assert isinstance(result, LiteFrame)

    # Verify operator structure: notna is invert(isna), so the projection
    # contains Invert(IsNA) per column
    op = result.data.op
    assert isinstance(op, LiteFrameProjection)
    assert len(op.projections) == 1

    proj = op.projections[0]
    assert isinstance(proj, NamedExpr)
    assert proj.name == "a"
    assert isinstance(proj.expr, LiteFrameInvert)
    # The inner operand of the Invert should be IsNA
    assert isinstance(proj.expr.operand, LiteFrameIsNA)
    assert isinstance(proj.expr.operand.operand, LiteFrameColumn)
    assert proj.expr.operand.operand.name == "a"
    # Result dtype should be bool
    assert result.dtypes["a"] == bool_dtype


def test_fillna_scalar():
    lf = LF({"a": [1, np.nan, 3]})
    result = lf.fillna(0)
    assert isinstance(result, LiteFrame)

    # Verify operator structure: projection with LiteFrameFillNA per column
    op = result.data.op
    assert isinstance(op, LiteFrameProjection)
    assert len(op.projections) == 1

    proj = op.projections[0]
    assert isinstance(proj, NamedExpr)
    assert proj.name == "a"
    assert isinstance(proj.expr, LiteFrameFillNA)
    # The left operand should be the column reference
    assert isinstance(proj.expr.left, LiteFrameColumn)
    assert proj.expr.left.name == "a"
    # The right operand should be a literal with the fill value
    assert isinstance(proj.expr.right, LiteFrameLiteral)
    assert proj.expr.right.value == 0
    # Original dtype is preserved (fillna does not change type)
    assert result.dtypes["a"] == lf.dtypes["a"]


def test_fillna_multi_column():
    lf = LF({"a": [1, np.nan, 3], "b": [np.nan, 5, 6]})
    result = lf.fillna(-1)
    assert isinstance(result, LiteFrame)

    op = result.data.op
    assert isinstance(op, LiteFrameProjection)
    assert len(op.projections) == 2

    for proj in op.projections:
        assert isinstance(proj, NamedExpr)
        assert isinstance(proj.expr, LiteFrameFillNA)
        assert isinstance(proj.expr.left, LiteFrameColumn)
        assert isinstance(proj.expr.right, LiteFrameLiteral)
        assert proj.expr.right.value == -1


def test_fillna_dict():
    lf = LF({"a": [1, np.nan, 3], "b": [np.nan, 5, 6]})
    result = lf.fillna({"a": 0})
    assert isinstance(result, LiteFrame)

    op = result.data.op
    assert isinstance(op, LiteFrameProjection)
    assert len(op.projections) == 2

    # Column "a" should have FillNA with value 0
    proj_a = op.projections[0]
    assert proj_a.name == "a"
    assert isinstance(proj_a.expr, LiteFrameFillNA)
    assert isinstance(proj_a.expr.right, LiteFrameLiteral)
    assert proj_a.expr.right.value == 0

    # Column "b" should be a pass-through (a Column reference, possibly bare)
    proj_b = op.projections[1]
    assert proj_b.name == "b"
    assert isinstance(proj_b, LiteFrameColumn)
    assert proj_b.name == "b"


def test_dropna_single_column():
    lf = LF({"a": [1, np.nan, 3]})
    result = lf.dropna()
    assert isinstance(result, LiteFrame)
    assert np.isnan(result.shape[0])

    # Verify operator is Filter; single-column dropna uses Invert(IsNA)
    # as the predicate
    op = result.data.op
    assert isinstance(op, LiteFrameFilter)

    predicate = op.predicate
    assert isinstance(predicate, LiteFrameInvert)
    inner = predicate.operand
    assert isinstance(inner, LiteFrameIsNA)
    assert isinstance(inner.operand, LiteFrameColumn)
    assert inner.operand.name == "a"


def test_dropna_multi_column_how_any():
    lf = LF({"a": [1, np.nan, 3], "b": [4, 5, 6]})
    result = lf.dropna(how="any")
    assert isinstance(result, LiteFrame)
    assert np.isnan(result.shape[0])

    # Verify operator: Filter with Invert(Or(IsNA(a), IsNA(b)))
    op = result.data.op
    assert isinstance(op, LiteFrameFilter)

    predicate = op.predicate
    assert isinstance(predicate, LiteFrameInvert)
    inner = predicate.operand
    # how="any" combines IsNA with Or
    assert isinstance(inner, LiteFrameOr)
    assert isinstance(inner.left, LiteFrameIsNA)
    assert isinstance(inner.right, LiteFrameIsNA)
    assert inner.left.operand.name == "a"
    assert inner.right.operand.name == "b"


def test_dropna_multi_column_how_all():
    lf = LF({"a": [1, np.nan, 3], "b": [4, 5, 6]})
    result = lf.dropna(how="all")
    assert isinstance(result, LiteFrame)
    assert np.isnan(result.shape[0])

    # Verify operator: Filter with Invert(And(IsNA(a), IsNA(b)))
    op = result.data.op
    assert isinstance(op, LiteFrameFilter)

    predicate = op.predicate
    assert isinstance(predicate, LiteFrameInvert)
    inner = predicate.operand
    # how="all" combines IsNA with And
    assert isinstance(inner, LiteFrameAnd)
    assert isinstance(inner.left, LiteFrameIsNA)
    assert isinstance(inner.right, LiteFrameIsNA)
    assert inner.left.operand.name == "a"
    assert inner.right.operand.name == "b"


def test_dropna_subset():
    lf = LF({"a": [1, np.nan, 3], "b": [np.nan, 5, 6]})
    result = lf.dropna(subset=["a"])
    assert isinstance(result, LiteFrame)
    assert np.isnan(result.shape[0])

    # Verify operator: Filter with Invert(IsNA(a)) as predicate,
    # column b not checked
    op = result.data.op
    assert isinstance(op, LiteFrameFilter)

    predicate = op.predicate
    assert isinstance(predicate, LiteFrameInvert)
    inner = predicate.operand
    assert isinstance(inner, LiteFrameIsNA)
    assert isinstance(inner.operand, LiteFrameColumn)
    assert inner.operand.name == "a"


@pytest.mark.parametrize("how", ["any", "all"])
def test_dropna_empty_subset(how):
    """dropna(subset=[]) matches pandas: how='any' keeps all rows,
    how='all' drops all rows."""
    lf = LF({"a": [1, np.nan, 3], "b": [4, 5, 6]})
    result = lf.dropna(subset=[], how=how)
    assert isinstance(result, LiteFrame)

    # The predicate should be Invert(Literal): how='any' → Literal(False)
    # → Invert → True (keep all); how='all' → Literal(True) → Invert → False
    # (drop all)
    op = result.data.op
    assert isinstance(op, LiteFrameFilter)
    predicate = op.predicate
    assert isinstance(predicate, LiteFrameInvert)
    inner = predicate.operand
    assert isinstance(inner, LiteFrameLiteral)
    assert inner.value is (how == "all")


def test_replace_scalar():
    from maxframe.liteframe.expressions import LiteFrameEq, LiteFrameWhere

    lf = LF({"a": [1, 2, 3], "b": [4, 5, 6]})
    result = lf.replace(2, 99)
    assert isinstance(result, LiteFrame)

    # Verify operator structure: projection with LiteFrameWhere per column
    op = result.data.op
    assert isinstance(op, LiteFrameProjection)
    assert len(op.projections) == 2

    for proj in op.projections:
        assert isinstance(proj, NamedExpr)
        assert isinstance(proj.expr, LiteFrameWhere)
        # condition is Eq(Column, Literal(2))
        assert isinstance(proj.expr.condition, LiteFrameEq)
        assert isinstance(proj.expr.condition.left, LiteFrameColumn)
        assert isinstance(proj.expr.condition.right, LiteFrameLiteral)
        assert proj.expr.condition.right.value == 2
        # true_value is Literal(99)
        assert isinstance(proj.expr.true_value, LiteFrameLiteral)
        assert proj.expr.true_value.value == 99
        # false_value is Column (original value)
        assert isinstance(proj.expr.false_value, LiteFrameColumn)


def test_replace_none_falls_back_to_fillna():
    """replace(to_replace=None, value=0) should behave like fillna(0)."""
    lf = LF({"a": [1, np.nan, 3]})
    result = lf.replace(to_replace=None, value=0)
    assert isinstance(result, LiteFrame)

    op = result.data.op
    assert isinstance(op, LiteFrameProjection)
    proj = op.projections[0]
    assert isinstance(proj.expr, LiteFrameFillNA)


def test_isna_on_range_column_materializes():
    """isna() on a frame with a range index column should set
    materialize_range_cols=True so the executor materializes the
    virtual __index__ column before generating IsNA expressions."""
    dtypes = pd.Series(
        {
            "a": wrap_arrow_dtype(pa.int64()),
            RANGE_COL_NAME: wrap_arrow_dtype(pa.int64()),
        }
    )
    op_src = LiteFrameReadODPSTable(
        table_name="t",
        dtypes=dtypes,
        default_index_type=DefaultIndexType.range,
    )
    lf = op_src._new_liteframe_from_source(shape=(10, 2))

    result = lf.isna()
    op = result.data.op
    assert isinstance(op, LiteFrameProjection)
    assert op.materialize_range_cols is True


def test_replace_on_range_column_materializes():
    """replace() on a frame with a range index column should set
    materialize_range_cols=True so the executor materializes the
    virtual __index__ column before generating replace expressions."""
    dtypes = pd.Series(
        {
            "a": wrap_arrow_dtype(pa.int64()),
            RANGE_COL_NAME: wrap_arrow_dtype(pa.int64()),
        }
    )
    op_src = LiteFrameReadODPSTable(
        table_name="t",
        dtypes=dtypes,
        default_index_type=DefaultIndexType.range,
    )
    lf = op_src._new_liteframe_from_source(shape=(10, 2))

    result = lf.replace(1, 99)
    op = result.data.op
    assert isinstance(op, LiteFrameProjection)
    assert op.materialize_range_cols is True


# ---------------------------------------------------------------------------
# Fusion tests: operators on projection outputs should fuse to root source
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "method, kwargs, expected_expr_cls",
    [
        ("isna", {}, LiteFrameIsNA),
        ("fillna", {"value": 0}, LiteFrameFillNA),
    ],
    ids=["isna", "fillna_scalar"],
)
def test_unary_missing_op_after_arithmetic_fuses(method, kwargs, expected_expr_cls):
    """(lf + 5).isna() / fillna(0) should fuse to a single projection on root source."""
    lf = LF({"a": [1, 2, 3]})
    result = getattr(lf + 5, method)(**kwargs)

    op = result.data.op
    assert isinstance(op, LiteFrameProjection)
    expr = op.projections[0].expr
    assert isinstance(expr, expected_expr_cls)
    inner = expr.operand if isinstance(expr, LiteFrameIsNA) else expr.left
    assert isinstance(inner, LiteFrameAdd)
    assert isinstance(inner.left, LiteFrameColumn)
    assert inner.left.name == "a"
    assert isinstance(inner.right, LiteFrameLiteral)
    assert inner.right.value == 5
    assert result.data.inputs[0] is lf._data


def test_fillna_dict_after_arithmetic_fuses():
    """(lf + 5).fillna({"a": 0}) should fuse pass-through columns."""
    lf = LF({"a": [1, 2], "b": [3, 4]})
    result = (lf + 5).fillna({"a": 0})

    op = result.data.op
    assert isinstance(op, LiteFrameProjection)
    # Column "a" should be FillNA(Add(Col, Lit), Lit)
    proj_a = op.projections[0]
    assert isinstance(proj_a, NamedExpr)
    assert isinstance(proj_a.expr, LiteFrameFillNA)
    assert isinstance(proj_a.expr.left, LiteFrameAdd)
    # Column "b" should carry the fused Add expr
    proj_b = op.projections[1]
    assert isinstance(proj_b, NamedExpr)
    assert isinstance(proj_b.expr, LiteFrameAdd)
    assert result.data.inputs[0] is lf._data


def test_replace_after_arithmetic_fuses():
    """(lf + 5).replace(7, 99) should fuse to a single projection on root source."""
    lf = LF({"a": [1, 2]})
    result = (lf + 5).replace(7, 99)

    op = result.data.op
    assert isinstance(op, LiteFrameProjection)
    expr = op.projections[0].expr
    # Should be Where(Eq(Add(Col, Lit), Lit), Lit, Add(Col, Lit))
    assert isinstance(expr, LiteFrameWhere)
    assert isinstance(expr.condition, LiteFrameEq)
    assert isinstance(expr.condition.left, LiteFrameAdd)
    assert isinstance(expr.condition.left.left, LiteFrameColumn)
    assert expr.condition.left.left.name == "a"
    assert result.data.inputs[0] is lf._data


def test_dropna_after_arithmetic_filters_root_source():
    """(lf + 5).dropna() should filter the root source and re-apply projection."""
    lf = LF({"a": [1, np.nan, 3]})
    source = lf + 5
    result = source.dropna()

    # The result should be a projection re-applied on a filtered root source
    op = result.data.op
    assert isinstance(op, LiteFrameProjection)
    filter_tileable = result.data.inputs[0]
    assert isinstance(filter_tileable.op, LiteFrameFilter)
    assert filter_tileable.inputs[0] is lf._data
