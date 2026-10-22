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

from maxframe.io.odpsio import pandas_to_odps_schema
from maxframe.liteframe.core import LiteFrame
from maxframe.liteframe.expressions import (
    LiteFrameAbs,
    LiteFrameAdd,
    LiteFrameAnd,
    LiteFrameBinUfuncExpr,
    LiteFrameColumn,
    LiteFrameDiv,
    LiteFrameEq,
    LiteFrameFloorDiv,
    LiteFrameGe,
    LiteFrameGt,
    LiteFrameInvert,
    LiteFrameLe,
    LiteFrameLiteral,
    LiteFrameLogicalNot,
    LiteFrameLShift,
    LiteFrameLt,
    LiteFrameMod,
    LiteFrameMul,
    LiteFrameNe,
    LiteFrameNeg,
    LiteFrameOr,
    LiteFramePow,
    LiteFrameRound,
    LiteFrameRShift,
    LiteFrameSub,
    LiteFrameUnaryUfuncExpr,
    LiteFrameXor,
    SourceCompatibilityError,
)
from maxframe.liteframe.initializer import LiteFrame as LF
from maxframe.liteframe.operators.project import LiteFrameProjection
from maxframe.utils import wrap_arrow_dtype

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BOOL_DTYPE = wrap_arrow_dtype(pa.bool_())


def _expr_class(result, col_index=0):
    """Return the inner expression class for a column in a projection result."""
    return result.data.op.projections[col_index].expr.__class__


# ---------------------------------------------------------------------------
# Binary arithmetic — public contract
# ---------------------------------------------------------------------------

_BINARY_OPS = [
    ("add", lambda lf: lf.add(10), LiteFrameAdd),
    ("sub", lambda lf: lf.sub(5), LiteFrameSub),
    ("mul", lambda lf: lf.mul(3), LiteFrameMul),
    ("div", lambda lf: lf.div(2), LiteFrameDiv),
    ("floordiv", lambda lf: lf.floordiv(3), LiteFrameFloorDiv),
    ("mod", lambda lf: lf.mod(3), LiteFrameMod),
    ("pow", lambda lf: lf.pow(2), LiteFramePow),
]


@pytest.mark.parametrize(
    "name,op_fn,expr_cls", _BINARY_OPS, ids=[t[0] for t in _BINARY_OPS]
)
def test_binary_arithmetic_returns_liteframe(name, op_fn, expr_cls):
    lf = LF({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
    result = op_fn(lf)
    assert isinstance(result, LiteFrame)
    assert list(result.columns) == ["a", "b"]
    assert _expr_class(result) is expr_cls


# ---------------------------------------------------------------------------
# Dunder operators (forward + reverse)
# ---------------------------------------------------------------------------

_DUNDER_OPS = [
    ("add", lambda lf: lf + 10, LiteFrameAdd),
    ("radd", lambda lf: 10 + lf, LiteFrameAdd),
    ("sub", lambda lf: lf - 5, LiteFrameSub),
    ("rsub", lambda lf: 5 - lf, LiteFrameSub),
    ("mul", lambda lf: lf * 3, LiteFrameMul),
    ("rmul", lambda lf: 3 * lf, LiteFrameMul),
    ("truediv", lambda lf: lf / 2, LiteFrameDiv),
    ("rtruediv", lambda lf: 2 / lf, LiteFrameDiv),
    ("floordiv", lambda lf: lf // 3, LiteFrameFloorDiv),
    ("rfloordiv", lambda lf: 3 // lf, LiteFrameFloorDiv),
    ("mod", lambda lf: lf % 3, LiteFrameMod),
    ("rmod", lambda lf: 3 % lf, LiteFrameMod),
    ("pow", lambda lf: lf**2, LiteFramePow),
    ("rpow", lambda lf: 2**lf, LiteFramePow),
]


@pytest.mark.parametrize(
    "name,op_fn,expr_cls", _DUNDER_OPS, ids=[t[0] for t in _DUNDER_OPS]
)
def test_dunder_arithmetic_returns_liteframe(name, op_fn, expr_cls):
    lf = LF({"a": [1, 2, 3]})
    result = op_fn(lf)
    assert isinstance(result, LiteFrame)
    assert _expr_class(result) is expr_cls


# ---------------------------------------------------------------------------
# Reverse non-commutative ops: literal must be on the left
# ---------------------------------------------------------------------------

_NON_COMMUTATIVE_R_OPS = [
    ("rsub", lambda lf: 10 - lf, LiteFrameSub),
    ("rdiv", lambda lf: 10 / lf, LiteFrameDiv),
    ("rfloordiv", lambda lf: 10 // lf, LiteFrameFloorDiv),
    ("rmod", lambda lf: 10 % lf, LiteFrameMod),
    ("rpow", lambda lf: 2**lf, LiteFramePow),
]


@pytest.mark.parametrize(
    "name,op_fn,expr_cls",
    _NON_COMMUTATIVE_R_OPS,
    ids=[t[0] for t in _NON_COMMUTATIVE_R_OPS],
)
def test_reverse_op_literal_on_left(name, op_fn, expr_cls):
    """Reverse non-commutative ops must place the literal on the left side."""
    lf = LF({"a": [1, 2, 3]})
    result = op_fn(lf)
    expr = result.data.op.projections[0].expr
    assert isinstance(expr, expr_cls)
    assert isinstance(expr.left, LiteFrameLiteral)


def test_rsub_operand_values():
    """10 - lf should produce LiteFrameSub(literal(10), column('a'))."""
    lf = LF({"a": [1, 2, 3]})
    result = 10 - lf
    expr = result.data.op.projections[0].expr
    assert expr.left.value == 10
    assert expr.right.name == "a"


def test_rdiv_operand_values():
    """10 / lf should produce LiteFrameDiv(literal(10), column('a'))."""
    lf = LF({"a": [2, 5, 10]})
    result = 10 / lf
    expr = result.data.op.projections[0].expr
    assert expr.left.value == 10
    assert expr.right.name == "a"


# ---------------------------------------------------------------------------
# Comparison operators
# ---------------------------------------------------------------------------

_COMP_OPS = [
    ("eq", lambda lf: lf.eq(2), LiteFrameEq),
    ("ne", lambda lf: lf.ne(2), LiteFrameNe),
    ("lt", lambda lf: lf.lt(3), LiteFrameLt),
    ("le", lambda lf: lf.le(3), LiteFrameLe),
    ("gt", lambda lf: lf.gt(1), LiteFrameGt),
    ("ge", lambda lf: lf.ge(1), LiteFrameGe),
]


@pytest.mark.parametrize(
    "name,op_fn,expr_cls", _COMP_OPS, ids=[t[0] for t in _COMP_OPS]
)
def test_comparison_method_returns_liteframe(name, op_fn, expr_cls):
    lf = LF({"a": [1, 2, 3]})
    result = op_fn(lf)
    assert isinstance(result, LiteFrame)
    assert _expr_class(result) is expr_cls


_COMP_DUNDER_OPS = [
    ("eq", lambda lf: lf == 2, LiteFrameEq),
    ("ne", lambda lf: lf != 2, LiteFrameNe),
    ("lt", lambda lf: lf < 3, LiteFrameLt),
    ("le", lambda lf: lf <= 3, LiteFrameLe),
    ("gt", lambda lf: lf > 1, LiteFrameGt),
    ("ge", lambda lf: lf >= 1, LiteFrameGe),
]


@pytest.mark.parametrize(
    "name,op_fn,expr_cls", _COMP_DUNDER_OPS, ids=[t[0] for t in _COMP_DUNDER_OPS]
)
def test_comparison_dunder_returns_liteframe(name, op_fn, expr_cls):
    lf = LF({"a": [1, 2, 3]})
    result = op_fn(lf)
    assert isinstance(result, LiteFrame)
    assert _expr_class(result) is expr_cls


# ---------------------------------------------------------------------------
# Unary operators
# ---------------------------------------------------------------------------


def test_neg():
    lf = LF({"a": [1, 2, 3]})
    result = -lf
    assert isinstance(result, LiteFrame)
    assert _expr_class(result) is LiteFrameNeg


def test_invert():
    lf = LF({"a": [True, False, True]})
    result = ~lf
    assert isinstance(result, LiteFrame)
    assert _expr_class(result) is LiteFrameInvert


def test_abs():
    lf = LF({"a": [1, -2, 3], "b": [-4.0, 5.0, -6.0]})
    result = abs(lf)
    assert isinstance(result, LiteFrame)
    assert list(result.columns) == ["a", "b"]
    assert _expr_class(result) is LiteFrameAbs


def test_round():
    lf = LF({"a": [1.234, 2.567, 3.891]})
    result = lf.round(decimals=2)
    assert isinstance(result, LiteFrame)
    assert _expr_class(result) is LiteFrameRound
    assert result.data.op.projections[0].expr.decimals == 2


def test_round_default_decimals():
    lf = LF({"a": [1.234]})
    result = lf.round()
    assert result.data.op.projections[0].expr.decimals == 0


# ---------------------------------------------------------------------------
# Logical / bitwise operators
# ---------------------------------------------------------------------------

_LOGICAL_OPS = [
    ("and", lambda lf1, lf2: lf1 & lf2, LiteFrameAnd),
    ("or", lambda lf1, lf2: lf1 | lf2, LiteFrameOr),
    ("xor", lambda lf1, lf2: lf1 ^ lf2, LiteFrameXor),
]


@pytest.mark.parametrize(
    "name,op_fn,expr_cls", _LOGICAL_OPS, ids=[t[0] for t in _LOGICAL_OPS]
)
def test_logical_operator_returns_liteframe(name, op_fn, expr_cls):
    lf = LF({"a": [True, False, True]})
    # Same-source operation: lf & lf, lf | lf, lf ^ lf
    result = op_fn(lf, lf)
    assert isinstance(result, LiteFrame)
    assert _expr_class(result) is expr_cls


# ---------------------------------------------------------------------------
# Dtype inference contract
# ---------------------------------------------------------------------------


def test_comparison_produces_bool_dtype():
    """Comparison operators should produce bool dtype."""
    lf = LF({"a": [1, 2, 3]})
    result = lf.eq(2)
    assert result.dtypes["a"] == _BOOL_DTYPE


def test_invert_produces_bool_dtype():
    """Invert (~) should produce bool dtype."""
    lf = LF({"a": [True, False, True]})
    result = ~lf
    assert result.dtypes["a"] == _BOOL_DTYPE


def test_add_preserves_numeric_dtype():
    """Addition of int+scalar should preserve int64 dtype."""
    lf = LF({"a": [1, 2, 3]})
    result = lf + 10
    assert result.dtypes["a"] == lf.dtypes["a"]


def test_add_preserves_float_dtype():
    """float + scalar should preserve float64 dtype."""
    lf = LF({"b": [4.0, 5.0, 6.0]})
    result = lf + 10
    assert result.dtypes["b"] == wrap_arrow_dtype(pa.float64())


@pytest.mark.parametrize(
    "scalar, expected_pa_dtype, expected_odps_type",
    [
        (np.int64(5), pa.int64(), "bigint"),
        (np.int32(5), pa.int64(), "bigint"),
        (np.float64(5.0), pa.float64(), "double"),
    ],
    ids=["np_int64", "np_int32", "np_float64"],
)
def test_numpy_scalar_odps_schema_dtype(scalar, expected_pa_dtype, expected_odps_type):
    """Arithmetic with NumPy scalars (e.g. a value fetched from a DPE result
    via ``.iloc[0, 0]``) must preserve the numeric dtype.  Regression for the
    yunqi demo round-trip: ``count() - total_length`` with a np.int64 scalar
    previously inferred ``string``, so the result materialized to an ODPS temp
    table with STRING columns and every count value came back as str."""
    raw = pd.DataFrame(
        {
            "num": [1, 2, 3, None, 5],
            "str_col": ["a", "b", None, "d", "e"],
            "float_col": [1.5, None, 3.5, 4.5, 5.5],
        }
    )
    result = LF(raw).count() - scalar
    assert (result.dtypes == wrap_arrow_dtype(expected_pa_dtype)).all()

    # DPE LiteFrameToODPSTableTiler._create_target_table infers the temp
    # table schema from the declared dtypes with unknown_as_string=True.
    empty_df = pd.DataFrame([], columns=result.dtypes.index).astype(result.dtypes)
    odps_schema, _ = pandas_to_odps_schema(
        empty_df, unknown_as_string=True, ignore_index=True
    )
    assert [c.type.name for c in odps_schema.columns] == [expected_odps_type] * len(
        raw.columns
    )


# ---------------------------------------------------------------------------
# Bitwise shift operators
# ---------------------------------------------------------------------------

_SHIFT_OPS = [
    ("lshift", lambda lf: lf << 2, LiteFrameLShift),
    ("rlshift", lambda lf: 2 << lf, LiteFrameLShift),
    ("rshift", lambda lf: lf >> 1, LiteFrameRShift),
    ("rrshift", lambda lf: 16 >> lf, LiteFrameRShift),
]


@pytest.mark.parametrize(
    "name,op_fn,expr_cls", _SHIFT_OPS, ids=[t[0] for t in _SHIFT_OPS]
)
def test_shift_operator_returns_liteframe(name, op_fn, expr_cls):
    lf = LF({"a": [1, 2, 3, 4]})
    result = op_fn(lf)
    assert isinstance(result, LiteFrame)
    assert _expr_class(result) is expr_cls


# Bitwise int ops that should preserve int64 dtype on an int64 column.
_BITWISE_INT_DTYPE_OPS = [
    ("lshift", lambda lf: lf["a"] << 2),
    ("rshift", lambda lf: lf["a"] >> 1),
    ("and", lambda lf: lf["a"] & 3),
    ("or", lambda lf: lf["a"] | 3),
    ("xor", lambda lf: lf["a"] ^ 1),
    ("invert", lambda lf: ~lf["a"]),
]


@pytest.mark.parametrize(
    "name,op_fn", _BITWISE_INT_DTYPE_OPS, ids=[t[0] for t in _BITWISE_INT_DTYPE_OPS]
)
def test_bitwise_int_preserves_int_dtype(name, op_fn):
    """Bitwise &, |, ^, ~, <<, >> on an int64 column preserve int64 dtype."""
    lf = LF({"a": [1, 2, 3, 4]})
    result = op_fn(lf)
    assert result.dtypes["a"] == wrap_arrow_dtype(pa.int64())


def test_shift_fuses_to_root_source():
    """(lf["a"] << 2) >> 1 should fuse into a single projection on the root."""
    lf = LF({"a": [1, 2, 3, 4]})
    result = (lf["a"] << 2) >> 1

    op = result.data.op
    assert isinstance(op, LiteFrameProjection)
    assert isinstance(op.projections[0].expr, LiteFrameRShift)
    assert isinstance(op.projections[0].expr.left, LiteFrameLShift)
    assert result.data.inputs[0] is lf._data


# ---------------------------------------------------------------------------
# LiteFrame hashability (after __eq__ override)
# ---------------------------------------------------------------------------


def test_liteframe_hashable():
    lf = LF({"a": [1, 2, 3]})
    h = hash(lf)
    assert isinstance(h, int)
    s = {lf}
    assert len(s) == 1
    d = {lf: "value"}
    assert d[lf] == "value"


# ---------------------------------------------------------------------------
# Source compatibility validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "op_fn",
    [
        lambda lf1, lf2: lf1 + lf2,
        lambda lf1, lf2: (lf1 + 10) / lf2,
    ],
    ids=["direct", "derived"],
)
def test_cross_source_binary_operation_raises(op_fn):
    """Cross-source ops raise SourceCompatibilityError."""
    lf1 = LF({"a": [1, 2]})
    lf2 = LF({"a": [3, 4]})

    with pytest.raises(SourceCompatibilityError) as exc_info:
        op_fn(lf1, lf2)

    assert "different sources" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Projection expression structure
# ---------------------------------------------------------------------------


_SAME_SOURCE_OPS = [
    ("add", lambda lf: lf + lf, LiteFrameAdd, False),
    ("gt", lambda lf: lf.gt(lf), LiteFrameGt, True),
]


@pytest.mark.parametrize(
    "name,op_fn,expr_cls,bool_result",
    _SAME_SOURCE_OPS,
    ids=[t[0] for t in _SAME_SOURCE_OPS],
)
def test_same_source_projection_structure(name, op_fn, expr_cls, bool_result):
    """Same-source ops produce projection with ExprClass(Column, Column)."""
    lf = LF({"a": [1, 2, 3], "b": [4, 5, 6]})
    result = op_fn(lf)

    for proj in result.data.op.projections:
        expr = proj.expr
        assert isinstance(expr, expr_cls)
        assert isinstance(expr.left, LiteFrameColumn)
        assert isinstance(expr.right, LiteFrameColumn)
        assert expr.left.name == proj.name
        assert expr.right.name == proj.name
        if bool_result:
            assert expr.dtype == _BOOL_DTYPE


_SAME_SOURCE_CHAINED_OPS = [
    ("div_chained", lambda lf: (lf + lf) / lf, LiteFrameDiv, False),
    ("gt_chained", lambda lf: ((lf + lf) / lf).gt(lf), LiteFrameGt, True),
]


@pytest.mark.parametrize(
    "name,op_fn,expr_cls,bool_result",
    _SAME_SOURCE_CHAINED_OPS,
    ids=[t[0] for t in _SAME_SOURCE_CHAINED_OPS],
)
def test_same_source_chained_projection_fuses(name, op_fn, expr_cls, bool_result):
    """Chained same-source ops fuse left operand expression from prior projection."""
    lf = LF({"a": [1, 2, 3], "b": [4, 5, 6]})
    result = op_fn(lf)

    for proj in result.data.op.projections:
        expr = proj.expr
        assert isinstance(expr, expr_cls)
        # Left operand is fused from the prior projection, not a bare column
        assert not isinstance(expr.left, LiteFrameColumn)
        # Right operand is resolved from root source (bare column)
        assert isinstance(expr.right, LiteFrameColumn)
        assert expr.right.name == proj.name
        if bool_result:
            assert expr.dtype == _BOOL_DTYPE


def test_scalar_binary_projection_structure():
    """lf + 10 produces projection with Add(Column, Literal) per column."""
    lf = LF({"a": [1, 2], "b": [4, 5]})
    result = lf + 10

    for proj in result.data.op.projections:
        expr = proj.expr
        assert isinstance(expr, LiteFrameAdd)
        assert isinstance(expr.left, LiteFrameColumn)
        assert isinstance(expr.right, LiteFrameLiteral)
        assert expr.left.name == proj.name
        assert expr.right.value == 10


def test_chained_projection_with_unary_merges_exprs():
    """~(lf > 3) merges the comparison into the invert projection."""
    lf = LF({"a": [1, 2, 3], "b": [4, 5, 6]})
    result = ~(lf > 3)

    for proj in result.data.op.projections:
        expr = proj.expr
        assert isinstance(expr, LiteFrameInvert)
        assert isinstance(expr.operand, LiteFrameGt)


# ---------------------------------------------------------------------------
# Chained binary scalar fusion
# ---------------------------------------------------------------------------


def test_chained_binary_scalar_fuses_to_single_projection():
    """(lf + 5) + 3 should produce a single projection, not a chain."""
    lf = LF({"a": [1, 2, 3]})
    chained = (lf + 5) + 3

    op = chained.data.op
    assert isinstance(op, LiteFrameProjection)

    # The expression tree should be Add(Add(Col("a"), Lit(5)), Lit(3))
    expr = op.projections[0].expr
    assert isinstance(expr, LiteFrameAdd)
    assert isinstance(expr.right, LiteFrameLiteral)
    assert expr.right.value == 3
    assert isinstance(expr.left, LiteFrameAdd)
    assert isinstance(expr.left.left, LiteFrameColumn)
    assert expr.left.left.name == "a"
    assert isinstance(expr.left.right, LiteFrameLiteral)
    assert expr.left.right.value == 5

    # Input should be the root source, not the intermediate projection
    assert chained.data.inputs[0] is lf._data


def test_chained_unary_fuses_to_root_source():
    """-(lf + 5) should produce a single projection reading from root source."""
    lf = LF({"a": [1, 2, 3]})
    chained = -(lf + 5)

    op = chained.data.op
    assert isinstance(op, LiteFrameProjection)

    # Expr should be Neg(Add(Col("a"), Lit(5)))
    expr = op.projections[0].expr
    assert isinstance(expr, LiteFrameNeg)
    assert isinstance(expr.operand, LiteFrameAdd)
    assert isinstance(expr.operand.left, LiteFrameColumn)
    assert expr.operand.left.name == "a"

    # Input should be the root source, not the intermediate projection
    assert chained.data.inputs[0] is lf._data


def test_cross_column_logical_fuses_to_root_source():
    """(lf["a"] > 5) & (lf["b"] < 10) should read from root source."""
    lf = LF({"a": [1, 2, 3], "b": [4, 5, 6]})
    mask = (lf["a"] > 5) & (lf["b"] < 10)

    op = mask.data.op
    assert isinstance(op, LiteFrameProjection)

    expr = op.projections[0].expr
    assert isinstance(expr, LiteFrameAnd)
    assert isinstance(expr.left, LiteFrameGt)
    assert isinstance(expr.right, LiteFrameLt)

    # Input should be the root source
    assert mask.data.inputs[0] is lf._data


def test_chained_binary_liteframe_fuses_both_sides():
    """(lf + 5) + (lf * 2) should fuse both operand expression trees."""
    lf = LF({"a": [1, 2, 3]})
    left_op = lf + 5
    right_op = lf * 2
    result = left_op + right_op

    op = result.data.op
    assert isinstance(op, LiteFrameProjection)

    # Expr should be Add(Add(Col("a"), Lit(5)), Mul(Col("a"), Lit(2)))
    expr = op.projections[0].expr
    assert isinstance(expr, LiteFrameAdd)
    assert isinstance(expr.left, LiteFrameAdd)
    assert isinstance(expr.left.left, LiteFrameColumn)
    assert expr.left.left.name == "a"
    assert isinstance(expr.left.right, LiteFrameLiteral)
    assert expr.left.right.value == 5
    assert isinstance(expr.right, LiteFrameMul)
    assert isinstance(expr.right.left, LiteFrameColumn)
    assert expr.right.left.name == "a"
    assert isinstance(expr.right.right, LiteFrameLiteral)
    assert expr.right.right.value == 2

    # Input should be the root source
    assert result.data.inputs[0] is lf._data


# ---------------------------------------------------------------------------
# NumPy ufunc dispatch via __array_ufunc__
# ---------------------------------------------------------------------------

_DIRECT_UNARY_UFUNCS = [
    "exp",
    "log",
    "log10",
    "log1p",
    "sqrt",
    "cbrt",
    "sin",
    "cos",
    "tan",
    "arcsin",
    "arccos",
    "arctan",
    "sinh",
    "cosh",
    "tanh",
    "arcsinh",
    "arccosh",
    "arctanh",
    "sign",
    "degrees",
    "radians",
]

_MAP_BATCH_UNARY_UFUNCS = [
    "exp2",
    "log2",
    "expm1",
    "square",
    # floor/ceil use the map_batches path so int inputs promote to float64,
    # matching numpy (polars .floor()/.ceil() preserve the integer dtype).
    "floor",
    "ceil",
    "reciprocal",
    "rint",
    "trunc",
    "spacing",
]

_BOOL_UFUNCS = ["isfinite", "isinf", "isnan"]


@pytest.mark.parametrize(
    "ufunc_name",
    _DIRECT_UNARY_UFUNCS + _MAP_BATCH_UNARY_UFUNCS,
    ids=_DIRECT_UNARY_UFUNCS + _MAP_BATCH_UNARY_UFUNCS,
)
def test_unary_ufunc_returns_liteframe(ufunc_name):
    lf = LF({"a": [1.0, 2.0, 3.0]})
    result = getattr(np, ufunc_name)(lf)
    assert isinstance(result, LiteFrame)
    expr = result.data.op.projections[0].expr
    assert isinstance(expr, LiteFrameUnaryUfuncExpr)
    assert expr.ufunc_name == ufunc_name


@pytest.mark.parametrize("ufunc_name", _BOOL_UFUNCS, ids=_BOOL_UFUNCS)
def test_bool_ufunc_produces_bool_dtype(ufunc_name):
    lf = LF({"a": [1.0, 2.0, 3.0]})
    result = getattr(np, ufunc_name)(lf)
    assert isinstance(result, LiteFrame)
    expr = result.data.op.projections[0].expr
    assert isinstance(expr, LiteFrameUnaryUfuncExpr)
    assert expr.ufunc_name == ufunc_name
    assert result.dtypes["a"] == _BOOL_DTYPE


# Integer-input ufuncs whose advertised dtype is version-independent.
# floor/ceil/trunc/rint deterministically promote int64 -> float64 (classic
# numpy 1.x contract; numpy >= 2.0 preserves int, but MaxFrame overrides to
# keep the advertised and executor result dtypes stable). square/sign
# preserve int64; exp/sqrt promote to float64 on every numpy version.
_INT_INPUT_UFUNC_DTYPE_CASES = [
    ("floor", pa.float64()),
    ("ceil", pa.float64()),
    ("trunc", pa.float64()),
    ("rint", pa.float64()),
    ("square", pa.int64()),
    ("sign", pa.int64()),
    ("exp", pa.float64()),
    ("sqrt", pa.float64()),
]


@pytest.mark.parametrize(
    "ufunc_name,expected_pa_dtype",
    _INT_INPUT_UFUNC_DTYPE_CASES,
    ids=[c[0] for c in _INT_INPUT_UFUNC_DTYPE_CASES],
)
def test_unary_ufunc_int_dtype_matches_numpy(ufunc_name, expected_pa_dtype):
    """np.<ufunc>(int64 LiteFrame) advertises a deterministic dtype.

    floor/ceil/trunc/rint on int64 promote to float64 regardless of the
    installed numpy version (numpy >= 2.0 preserves int, but MaxFrame
    follows the classic numpy 1.x contract). square/sign preserve int64;
    exp/sqrt promote to float64.
    """
    lf = LF({"a": [1, 2, 3]})
    result = getattr(np, ufunc_name)(lf)
    assert result.dtypes["a"] == wrap_arrow_dtype(expected_pa_dtype)


_BINARY_UFUNCS = ["arctan2", "hypot", "maximum", "minimum", "logaddexp"]


@pytest.mark.parametrize("ufunc_name", _BINARY_UFUNCS, ids=_BINARY_UFUNCS)
def test_binary_ufunc_returns_liteframe(ufunc_name):
    lf = LF({"a": [1.0, 2.0, 3.0]})
    result = getattr(np, ufunc_name)(lf, 1.0)
    assert isinstance(result, LiteFrame)
    expr = result.data.op.projections[0].expr
    assert isinstance(expr, LiteFrameBinUfuncExpr)
    assert expr.ufunc_name == ufunc_name


def test_existing_ufuncs_still_dispatch_to_dedicated_classes():
    """np.absolute, np.negative, np.add etc. keep their existing expression classes."""
    lf = LF({"a": [1, -2, 3]})
    assert _expr_class(np.absolute(lf)) is LiteFrameAbs
    assert _expr_class(np.fabs(lf)) is LiteFrameAbs
    assert _expr_class(np.negative(lf)) is LiteFrameNeg
    assert _expr_class(np.add(lf, 1)) is LiteFrameAdd
    assert _expr_class(np.subtract(lf, 1)) is LiteFrameSub
    assert _expr_class(np.multiply(lf, 2)) is LiteFrameMul


# np.left_shift / np.right_shift reuse the dedicated shift expression classes.
# Regression for the AI-review note: these ufuncs previously fell through to
# ``NotImplemented`` because they were absent from the ufunc dispatch table,
# even though ``LiteFrameLShift`` / ``LiteFrameRShift`` and their codegen existed.
_SHIFT_UFUNC_OPS = [
    ("left_shift", lambda lf: np.left_shift(lf, 2), LiteFrameLShift),
    ("right_shift", lambda lf: np.right_shift(lf, 1), LiteFrameRShift),
    ("rleft_shift", lambda lf: np.left_shift(2, lf), LiteFrameLShift),
    ("rright_shift", lambda lf: np.right_shift(16, lf), LiteFrameRShift),
]


@pytest.mark.parametrize(
    "name,op_fn,expr_cls",
    _SHIFT_UFUNC_OPS,
    ids=[t[0] for t in _SHIFT_UFUNC_OPS],
)
def test_shift_ufunc_returns_liteframe(name, op_fn, expr_cls):
    """np.left_shift / np.right_shift dispatch to the shift expr classes.

    Covers both operand orders; the reverse cases guard against the previous
    ``NotImplemented`` fallthrough (comment #2).
    """
    lf = LF({"a": [1, 2, 3]})
    result = op_fn(lf)
    assert isinstance(result, LiteFrame)
    assert _expr_class(result) is expr_cls
    # Shift preserves the left operand's integer dtype.
    assert result.dtypes["a"] == lf.dtypes["a"]


# np.logical_not must use LiteFrameLogicalNot (bool result), not LiteFrameInvert.
# Regression for the AI-review correctness note (comment #1): ``np.logical_not``
# and the ``~`` operator previously shared ``LiteFrameInvert``. Since
# ``LiteFrameInvert`` is now dtype-aware bitwise NOT (int -> int), reusing it
# made ``np.logical_not`` return bitwise integers instead of booleans. The
# ``~`` operator's int-preserving behavior is covered by
# ``test_bitwise_int_preserves_int_dtype``.
_LOGICAL_NOT_INPUTS = [
    ("int", {"a": [0, 1, 2]}),
    ("bool", {"a": [True, False, True]}),
]


@pytest.mark.parametrize(
    "name,data", _LOGICAL_NOT_INPUTS, ids=[t[0] for t in _LOGICAL_NOT_INPUTS]
)
def test_logical_not_uses_dedicated_bool_expr(name, data):
    """np.logical_not always yields LiteFrameLogicalNot with a bool dtype."""
    lf = LF(data)
    result = np.logical_not(lf)
    assert isinstance(result, LiteFrame)
    assert _expr_class(result) is LiteFrameLogicalNot
    assert result.dtypes["a"] == _BOOL_DTYPE


def test_excluded_ufuncs_return_notimplemented():
    """np.modf and np.frexp raise TypeError (multi-output, unsupported)."""
    lf = LF({"a": [1.0, 2.0, 3.0]})
    with pytest.raises(TypeError):
        np.modf(lf)
    with pytest.raises(TypeError):
        np.frexp(lf)
