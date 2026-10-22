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

from maxframe.codegen.spe.liteframe.expressions import LiteFrameExprEvaluator
from maxframe.liteframe.accessors.datetime_.core import (
    LiteFrameDatetimeExpr,
    LiteFrameDatetimeMultiExpr,
)
from maxframe.liteframe.accessors.string_.core import (
    LiteFrameStrExpr,
    LiteFrameStrMultiExpr,
)
from maxframe.liteframe.accessors.struct_.core import LiteFrameStructExpr
from maxframe.liteframe.arithmetic.clip import _clip_expr
from maxframe.liteframe.expressions import (
    LiteFrameAbs,
    LiteFrameAdd,
    LiteFrameAnd,
    LiteFrameAstype,
    LiteFrameBinUfuncExpr,
    LiteFrameColumn,
    LiteFrameEq,
    LiteFrameFillNA,
    LiteFrameGe,
    LiteFrameGt,
    LiteFrameInvert,
    LiteFrameIsNA,
    LiteFrameLe,
    LiteFrameLiteral,
    LiteFrameLogicalNot,
    LiteFrameLShift,
    LiteFrameLt,
    LiteFrameMultiColumnSubExpr,
    LiteFrameNeg,
    LiteFrameRound,
    LiteFrameRShift,
    LiteFrameSqrt,
    LiteFrameSub,
    LiteFrameTreeAnd,
    LiteFrameTreeOr,
    LiteFrameUnaryUfuncExpr,
    LiteFrameWhere,
    NamedExpr,
)
from maxframe.utils import wrap_arrow_dtype

I64 = wrap_arrow_dtype(pa.int64())
F64 = wrap_arrow_dtype(pa.float64())
BOOL = wrap_arrow_dtype(pa.bool_())
STR = wrap_arrow_dtype(pa.string())


def _col(name, pa_dtype):
    return LiteFrameColumn(name=name, dtype=wrap_arrow_dtype(pa_dtype))


def test_eval_column(test_df):
    evaluator = LiteFrameExprEvaluator(test_df)
    result = evaluator.eval(_col("a", pa.int64()))
    pd.testing.assert_series_equal(result, test_df["a"])


def test_eval_literal(test_df):
    evaluator = LiteFrameExprEvaluator(test_df)
    result = evaluator.eval(LiteFrameLiteral(value=5, dtype=I64))
    assert result == 5


@pytest.mark.parametrize(
    ("expr_factory", "op_callable"),
    [
        (lambda a, b: LiteFrameAdd(left=a, right=b, dtype=I64), lambda sa, sb: sa + sb),
        (lambda a, b: LiteFrameSub(left=b, right=a, dtype=I64), lambda sa, sb: sb - sa),
    ],
    ids=["add", "sub"],
)
def test_eval_binary_col_col(test_df, expr_factory, op_callable):
    evaluator = LiteFrameExprEvaluator(test_df)
    col_a = _col("a", pa.int64())
    col_b = _col("b", pa.int64())
    result = evaluator.eval(expr_factory(col_a, col_b))
    expected = op_callable(test_df["a"], test_df["b"])
    pd.testing.assert_series_equal(result, expected)


def test_eval_eq(test_df):
    evaluator = LiteFrameExprEvaluator(test_df)
    col_a = _col("a", pa.int64())
    lit = LiteFrameLiteral(value=2, dtype=I64)
    result = evaluator.eval(LiteFrameEq(left=col_a, right=lit, dtype=BOOL))
    expected = test_df["a"] == 2
    pd.testing.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    ("df_data", "pa_dtype", "expr_factory", "expected_factory"),
    [
        (
            {"a": [1, 2, 3, 4]},
            pa.int64(),
            lambda col: LiteFrameNeg(operand=col, dtype=I64),
            lambda s: -s,
        ),
        (
            {"a": [-1, 2, -3]},
            pa.int64(),
            lambda col: LiteFrameAbs(operand=col, dtype=I64),
            lambda s: s.abs(),
        ),
        (
            {"a": [True, False, True]},
            pa.bool_(),
            lambda col: LiteFrameInvert(operand=col, dtype=BOOL),
            lambda s: ~s,
        ),
        (
            {"a": [1.234, 2.567, 3.891]},
            pa.float64(),
            lambda col: LiteFrameRound(operand=col, decimals=2, dtype=F64),
            lambda s: s.round(2),
        ),
        (
            {"a": [4.0, 9.0, 16.0]},
            pa.float64(),
            lambda col: LiteFrameSqrt(operand=col, dtype=F64),
            lambda s: np.sqrt(s),
        ),
        (
            {"a": [1.0, None, 3.0]},
            pa.float64(),
            lambda col: LiteFrameIsNA(operand=col, dtype=BOOL),
            lambda s: s.isna(),
        ),
    ],
    ids=["neg", "abs", "invert", "round", "sqrt", "isna"],
)
def test_eval_unary(df_data, pa_dtype, expr_factory, expected_factory):
    df = pd.DataFrame(df_data)
    evaluator = LiteFrameExprEvaluator(df)
    col = _col("a", pa_dtype)
    result = evaluator.eval(expr_factory(col))
    expected = expected_factory(df["a"])
    pd.testing.assert_series_equal(result, expected)


def test_eval_astype():
    df = pd.DataFrame({"a": [1, 2, 3]}, dtype="int64")
    evaluator = LiteFrameExprEvaluator(df)
    col = _col("a", pa.int64())
    result = evaluator.eval(
        LiteFrameAstype(
            operand=col,
            target_dtype=F64,
            errors="raise",
            dtype=F64,
        )
    )
    expected = df["a"].astype("float64")
    pd.testing.assert_series_equal(result, expected)


def test_eval_where():
    df = pd.DataFrame({"a": [1, 2, 3, 4]})
    evaluator = LiteFrameExprEvaluator(df)
    col = _col("a", pa.int64())
    cond = LiteFrameEq(
        left=col,
        right=LiteFrameLiteral(value=2, dtype=I64),
        dtype=BOOL,
    )
    true_val = LiteFrameLiteral(value=100, dtype=I64)
    false_val = LiteFrameLiteral(value=0, dtype=I64)
    result = evaluator.eval(
        LiteFrameWhere(
            condition=cond,
            true_value=true_val,
            false_value=false_val,
            dtype=I64,
        )
    )
    expected = pd.Series([0, 100, 0, 0], index=df.index)
    pd.testing.assert_series_equal(result, expected)


def test_eval_fillna():
    df = pd.DataFrame({"a": [1.0, None, 3.0]})
    evaluator = LiteFrameExprEvaluator(df)
    col = _col("a", pa.float64())
    fill = LiteFrameLiteral(value=0, dtype=F64)
    result = evaluator.eval(LiteFrameFillNA(left=col, right=fill, dtype=F64))
    expected = df["a"].fillna(0)
    pd.testing.assert_series_equal(result, expected)


def test_eval_named_expr(test_df):
    evaluator = LiteFrameExprEvaluator(test_df)
    col_a = _col("a", pa.int64())
    result = evaluator.eval(NamedExpr(name="x", expr=col_a, dtype=I64))
    pd.testing.assert_series_equal(result, test_df["a"])


@pytest.mark.parametrize(
    "method,args,kwargs,out_dtype",
    [
        ("upper", (), {}, STR),
        ("lower", (), {}, STR),
        ("title", (), {}, STR),
        ("strip", (), {}, STR),
        ("capitalize", (), {}, STR),
        ("replace", ("l", "L"), {}, STR),
        ("zfill", (8,), {}, STR),
        ("count", ("l",), {}, I64),
        ("find", ("o",), {}, I64),
        ("startswith", ("he",), {}, BOOL),
        ("isalpha", (), {}, BOOL),
        ("isdigit", (), {}, BOOL),
    ],
)
def test_eval_string_accessor(method, args, kwargs, out_dtype):
    df = pd.DataFrame({"s": ["hello", "world", "foo bar"]})
    evaluator = LiteFrameExprEvaluator(df)
    col = _col("s", pa.string())
    expr = LiteFrameStrExpr(
        operand=col, method=method, args=args, kwargs=kwargs, dtype=out_dtype
    )
    result = evaluator.eval(expr)
    expected = getattr(df["s"].str, method)(*args, **kwargs)
    pd.testing.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "method,output_names,args,kwargs",
    [
        ("partition", ["0", "1", "2"], ("_",), {}),
        ("rpartition", ["0", "1", "2"], ("_",), {}),
        ("extract", ["0"], (r"([a-z])",), {}),
        ("split", ["0", "1"], ("_",), {"expand": True, "n": 1}),
    ],
)
def test_eval_str_multi_accessor(method, output_names, args, kwargs):
    df = pd.DataFrame({"s": ["a_b_c", "d_e", "f"]})
    evaluator = LiteFrameExprEvaluator(df)
    col = _col("s", pa.string())
    dtypes = {n: str(STR) for n in output_names}
    multi = LiteFrameStrMultiExpr(
        operands=[col],
        method=method,
        args=args,
        kwargs=kwargs,
        output_names=output_names,
        dtypes=dtypes,
    )
    result = evaluator.eval(multi)
    expected = getattr(df["s"].str, method)(*args, **kwargs)
    expected.columns = output_names
    pd.testing.assert_frame_equal(result, expected)

    # Sub-expr extraction matches per-column access
    for name in output_names:
        sub = LiteFrameMultiColumnSubExpr(multi_expr=multi, output_name=name, dtype=STR)
        col_result = evaluator.eval(sub)
        pd.testing.assert_series_equal(col_result, expected[name])


@pytest.mark.parametrize(
    "method,is_property,args,kwargs,out_dtype",
    [
        ("year", True, (), {}, wrap_arrow_dtype(pa.int32())),
        ("is_leap_year", True, (), {}, BOOL),
        ("strftime", False, ("%Y-%m-%d",), {}, STR),
        ("normalize", False, (), {}, wrap_arrow_dtype(pa.timestamp("ns"))),
        ("round", False, ("1h",), {}, wrap_arrow_dtype(pa.timestamp("ns"))),
        ("floor", False, ("1h",), {}, wrap_arrow_dtype(pa.timestamp("ns"))),
        ("ceil", False, ("1h",), {}, wrap_arrow_dtype(pa.timestamp("ns"))),
    ],
)
def test_eval_datetime_accessor(method, is_property, args, kwargs, out_dtype):
    df = pd.DataFrame({"dt": pd.date_range("2024-01-01", periods=3, freq="D")})
    evaluator = LiteFrameExprEvaluator(df)
    col = _col("dt", pa.timestamp("ns"))
    expr = LiteFrameDatetimeExpr(
        operand=col,
        method=method,
        is_property=is_property,
        args=args,
        kwargs=kwargs,
        dtype=out_dtype,
    )
    result = evaluator.eval(expr)
    if is_property:
        expected = getattr(df["dt"].dt, method)
    else:
        expected = getattr(df["dt"].dt, method)(*args, **kwargs)
    pd.testing.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "method,is_property,out_dtype",
    [
        ("days", True, I64),
        ("seconds", True, wrap_arrow_dtype(pa.int32())),
        ("microseconds", True, wrap_arrow_dtype(pa.int32())),
        ("nanoseconds", True, wrap_arrow_dtype(pa.int32())),
        ("total_seconds", False, F64),
    ],
)
def test_eval_timedelta_accessor(method, is_property, out_dtype):
    df = pd.DataFrame(
        {"td": pd.to_timedelta([93784005006000, -129600000000000, 7], unit="ns")}
    )
    evaluator = LiteFrameExprEvaluator(df)
    col = _col("td", pa.duration("ns"))
    expr = LiteFrameDatetimeExpr(
        operand=col,
        method=method,
        is_property=is_property,
        args=(),
        kwargs={},
        dtype=out_dtype,
    )
    result = evaluator.eval(expr)
    if is_property:
        expected = getattr(df["td"].dt, method)
    else:
        expected = getattr(df["td"].dt, method)()
    pd.testing.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "method,output_names",
    [
        ("isocalendar", ["year", "week", "day"]),
        (
            "components",
            [
                "days",
                "hours",
                "minutes",
                "seconds",
                "milliseconds",
                "microseconds",
                "nanoseconds",
            ],
        ),
    ],
)
def test_eval_datetime_multi_accessor(method, output_names):
    if method == "isocalendar":
        df = pd.DataFrame({"dt": pd.date_range("2024-01-01", periods=3, freq="D")})
        pa_type = pa.timestamp("ns")
        dtypes = {
            "year": wrap_arrow_dtype(pa.int32()),
            "week": wrap_arrow_dtype(pa.int32()),
            "day": wrap_arrow_dtype(pa.int32()),
        }
    else:
        df = pd.DataFrame(
            {"td": pd.to_timedelta([93784005006000, -129600000000000, 7], unit="ns")}
        )
        pa_type = pa.duration("ns")
        dtypes = {n: I64 for n in output_names}

    evaluator = LiteFrameExprEvaluator(df)
    col = _col(df.columns[0], pa_type)
    multi = LiteFrameDatetimeMultiExpr(
        operands=[col],
        method=method,
        is_property=(method == "components"),
        output_names=output_names,
        dtypes={k: str(v) for k, v in dtypes.items()},
    )
    result = evaluator.eval(multi)
    raw = getattr(df[df.columns[0]].dt, method)
    expected = raw() if callable(raw) else raw
    expected.columns = output_names
    pd.testing.assert_frame_equal(result, expected)


# ---------------------------------------------------------------------------
# Recently-added expression handlers (logical_not, tree_and/or, shifts, ufuncs)
# ---------------------------------------------------------------------------


def test_eval_logical_not():
    """logical_not casts to bool then inverts, matching numpy semantics."""
    df = pd.DataFrame({"a": [0, 1, 2, 0]})
    evaluator = LiteFrameExprEvaluator(df)
    col = _col("a", pa.int64())
    result = evaluator.eval(LiteFrameLogicalNot(operand=col, dtype=BOOL))
    assert result.tolist() == [True, False, False, True]


@pytest.mark.parametrize(
    ("tree_class", "thresholds", "expected"),
    [
        (LiteFrameTreeAnd, (1, 1), [False, True, True, True]),
        (LiteFrameTreeOr, (3, 35), [False, False, False, True]),
    ],
    ids=["and", "or"],
)
def test_eval_tree_and_or(test_df, tree_class, thresholds, expected):
    """tree_and/tree_or reduce multiple boolean conjuncts/disjuncts."""
    evaluator = LiteFrameExprEvaluator(test_df)
    col_a = _col("a", pa.int64())
    col_b = _col("b", pa.int64())
    thr_a, thr_b = thresholds
    expr = tree_class(
        operands=[
            LiteFrameGt(
                left=col_a, right=LiteFrameLiteral(value=thr_a, dtype=I64), dtype=BOOL
            ),
            LiteFrameGt(
                left=col_b, right=LiteFrameLiteral(value=thr_b, dtype=I64), dtype=BOOL
            ),
        ],
        dtype=BOOL,
    )
    result = evaluator.eval(expr)
    assert result.tolist() == expected


@pytest.mark.parametrize(
    ("shift_class", "expected"),
    [
        (LiteFrameLShift, [2, 4, 6, 8]),
        (LiteFrameRShift, [0, 1, 1, 2]),
    ],
    ids=["lshift", "rshift"],
)
def test_eval_shift(test_df, shift_class, expected):
    """<</>> use np.left_shift / np.right_shift."""
    evaluator = LiteFrameExprEvaluator(test_df)
    col_a = _col("a", pa.int64())
    expr = shift_class(
        left=col_a,
        right=LiteFrameLiteral(value=1, dtype=I64),
        dtype=I64,
    )
    result = evaluator.eval(expr)
    assert result.tolist() == expected


def test_eval_unary_ufunc(test_df):
    """Unary ufunc dispatches to np.<name>."""
    evaluator = LiteFrameExprEvaluator(test_df)
    col_a = _col("a", pa.int64())
    expr = LiteFrameUnaryUfuncExpr(operand=col_a, ufunc_name="sin", dtype=F64)
    result = evaluator.eval(expr)
    expected = np.sin(test_df["a"])
    np.testing.assert_array_almost_equal(result.to_numpy(), expected.to_numpy())


def test_eval_bin_ufunc(test_df):
    """Binary ufunc dispatches to np.<name>."""
    evaluator = LiteFrameExprEvaluator(test_df)
    col_a = _col("a", pa.int64())
    col_b = _col("b", pa.int64())
    expr = LiteFrameBinUfuncExpr(
        left=col_a,
        right=col_b,
        ufunc_name="maximum",
        dtype=I64,
    )
    result = evaluator.eval(expr)
    assert result.tolist() == [10, 20, 30, 40]


def test_eval_struct_field_nested_path():
    """C12: struct.field(['inner','x']) resolves a nested list path one segment
    at a time instead of raising TypeError: unhashable type: 'list'."""
    nested_dtype = wrap_arrow_dtype(
        pa.struct([("inner", pa.struct([("x", pa.int64()), ("y", pa.string())]))])
    )
    df = pd.DataFrame(
        {"s": [{"inner": {"x": 1, "y": "a"}}, {"inner": {"x": 2, "y": "b"}}]}
    )
    evaluator = LiteFrameExprEvaluator(df)
    col = LiteFrameColumn(name="s", dtype=nested_dtype)
    expr = LiteFrameStructExpr(
        operand=col, method="field", args=(["inner", "x"],), dtype=nested_dtype
    )
    result = evaluator.eval(expr)
    assert result.tolist() == [1, 2]


# ---------------------------------------------------------------------------
# Clip expression (composed from Where + Lt/Gt)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "lower,upper",
    [
        (2, 3),
        (None, 3),
        (2, None),
    ],
    ids=["both", "upper_only", "lower_only"],
)
def test_eval_clip(test_df, lower, upper):
    """clip composed via Where(Lt/Gt, bound, col) matches pd.clip."""
    evaluator = LiteFrameExprEvaluator(test_df)
    col_a = _col("a", pa.int64())
    expr = _clip_expr(col_a, I64, lower, upper)
    result = evaluator.eval(expr)
    expected = test_df["a"].clip(lower=lower, upper=upper)
    assert result.tolist() == expected.tolist()


# ---------------------------------------------------------------------------
# Between expression (composed from And(Ge/Gt, Le/Lt))
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "inclusive,left_cls,right_cls",
    [
        ("both", LiteFrameGe, LiteFrameLe),
        ("left", LiteFrameGe, LiteFrameLt),
        ("right", LiteFrameGt, LiteFrameLe),
        ("neither", LiteFrameGt, LiteFrameLt),
    ],
)
def test_eval_between(test_df, inclusive, left_cls, right_cls):
    """between composed via And(left_cmp, right_cmp) matches pd.Series.between."""
    evaluator = LiteFrameExprEvaluator(test_df)
    col_a = _col("a", pa.int64())
    left = left_cls(left=col_a, right=LiteFrameLiteral(value=2, dtype=I64), dtype=BOOL)
    right = right_cls(
        left=col_a, right=LiteFrameLiteral(value=3, dtype=I64), dtype=BOOL
    )
    expr = LiteFrameAnd(left=left, right=right, dtype=BOOL)
    result = evaluator.eval(expr)
    expected = test_df["a"].between(2, 3, inclusive=inclusive).tolist()
    assert result.tolist() == expected
