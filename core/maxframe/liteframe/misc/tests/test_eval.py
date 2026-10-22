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

from maxframe.liteframe import LiteFrame, col
from maxframe.liteframe.expressions import (
    LiteFrameAbs,
    LiteFrameAdd,
    LiteFrameAnd,
    LiteFrameColumn,
    LiteFrameFloorDiv,
    LiteFrameGt,
    LiteFrameInvert,
    LiteFrameIsInExpr,
    LiteFrameMod,
    LiteFrameNeg,
    LiteFramePow,
    SourceCompatibilityError,
)
from maxframe.liteframe.operators.filter import LiteFrameFilter
from maxframe.liteframe.operators.project import LiteFrameProjection
from maxframe.utils import wrap_arrow_dtype


def _lf():
    return LiteFrame({"A": [1, 2, 3, 4, 5], "B": [10, 20, 30, 40, 50]})


_BOOL = wrap_arrow_dtype(pa.bool_())


def _inner_expr(result):
    """Return the inner LiteFrameExpr of a single-projection eval result."""
    proj = result.op.projections[0]
    return proj.expr if hasattr(proj, "expr") else proj


@pytest.mark.parametrize(
    "expr, expected_type",
    [
        ("A + B", LiteFrameAdd),
        ("(A > 2) & (B < 50)", LiteFrameAnd),
        ("abs(A - 3)", LiteFrameAbs),
        ("A not in [1, 3, 5]", LiteFrameInvert),
        ("+A", LiteFrameColumn),  # unary plus is identity
        ("A % 2", LiteFrameMod),
        ("A ** 2", LiteFramePow),
        ("A // 2", LiteFrameFloorDiv),
        ("-A", LiteFrameNeg),
        ("1 < A < 3", LiteFrameAnd),  # chained compare folds into And
    ],
)
def test_eval_builds_expression(expr, expected_type):
    result = _lf().eval(expr)
    assert isinstance(result.op, LiteFrameProjection)
    assert isinstance(_inner_expr(result), expected_type)


@pytest.mark.parametrize(
    "expr, expected_dtype",
    [("A in [1, 3]", _BOOL), ("(A > 2) & (B < 50)", _BOOL)],
)
def test_eval_boolean_dtype(expr, expected_dtype):
    assert _lf().eval(expr).dtypes.iloc[0] == expected_dtype


def test_eval_isin_values():
    assert _inner_expr(_lf().eval("A in [1, 3, 5]")).values == [1, 3, 5]


def test_eval_not_in_wraps_isin():
    inner = _inner_expr(_lf().eval("A not in [1, 3, 5]"))
    assert isinstance(inner.operand, LiteFrameIsInExpr)


def test_eval_pure_expr_returns_one_column():
    assert len(_lf().eval("A - B").columns) == 1


def test_eval_does_not_mutate_source():
    lf = _lf()
    lf.eval("A + B")
    assert list(lf.columns) == ["a", "b"]


# ---------------------------------------------------------------------------
# eval: assignment
# ---------------------------------------------------------------------------


def test_eval_assignment_adds_column():
    lf = _lf()
    result = lf.eval("C = A + B")
    assert list(result.columns) == ["a", "b", "c"]
    assert "c" not in lf.columns


def test_eval_assignment_inplace_mutates_source():
    lf = _lf()
    assert lf.eval("C = A + B", inplace=True) is None
    assert "c" in lf.columns


def test_eval_multiline_assignment_forward_ref():
    result = _lf().eval(
        """
        C = A + B
        D = C - A
        """
    )
    assert list(result.columns) == ["a", "b", "c", "d"]


def test_eval_overwrite_existing_column():
    result = _lf().eval("A = A * 2")
    assert set(result.columns) == {"a", "b"}
    assert isinstance(result.op, LiteFrameProjection)


# ---------------------------------------------------------------------------
# eval: locals, backticks, case-insensitivity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "frame, expr, expected_type",
    [
        (None, "A > @threshold", LiteFrameGt),  # local var via @
        (LiteFrame({"C C": [1, 2], "D": [3, 4]}), "`C C` + D", LiteFrameAdd),
        (None, "a + b", LiteFrameAdd),  # case-insensitive
    ],
)
def test_eval_resolver_features(frame, expr, expected_type):
    lf = frame if frame is not None else _lf()
    threshold = 3  # noqa: F841  referenced via @threshold in the expr string
    assert isinstance(_inner_expr(lf.eval(expr)), expected_type)


# ---------------------------------------------------------------------------
# query
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "expr",
    [
        "A > 2",
        "(A > 2) & (B < 50)",
        "A in [1, 3, 5]",
        "A not in [1, 3, 5]",
        "~(A > 2)",
        "not (A > 2)",  # `not` keyword maps to ~
    ],
)
def test_query_builds_filter(expr):
    result = _lf().query(expr)
    assert isinstance(result.op, LiteFrameFilter)
    assert list(result.columns) == ["a", "b"]


def test_query_local_var_builds_filter():
    lf = _lf()
    threshold = 2  # noqa: F841  referenced via @threshold in the expr string
    assert isinstance(lf.query("A > @threshold").op, LiteFrameFilter)


def test_query_inplace_mutates_source():
    lf = _lf()
    assert lf.query("A > 2", inplace=True) is None
    assert isinstance(lf._data.op, LiteFrameFilter)


# ---------------------------------------------------------------------------
# error cases
# ---------------------------------------------------------------------------


_ISIN_BAD_VALUES = [
    lambda: col("A").isin(_lf()),  # bare tileable
    lambda: col("A").isin(col("B")),  # bare expression
    lambda: col("A").isin([col("B"), 1]),  # expression member
    lambda: col("A").isin([[1, 2], 3]),  # nested list member
    lambda: col("A").isin([{"x": 1}]),  # dict member
]


@pytest.mark.parametrize("values", _ISIN_BAD_VALUES)
def test_isin_rejects_non_scalar(values):
    with pytest.raises(NotImplementedError, match="flat list of scalars"):
        values()


def test_isin_query_against_column_raises():
    with pytest.raises(NotImplementedError):
        _lf().query("A in B")


@pytest.mark.parametrize(
    "expr, error, kwargs",
    [
        (123, TypeError, {}),
        ("Z + A", KeyError, {}),
        ("A + B", NotImplementedError, {"engine": "python"}),
        ("A + B", ValueError, {"inplace": True}),  # inplace without assignment
        ("A\nC = A + B", SyntaxError, {}),  # mixed: pure expr then assignment
        ("C = A + B\nC * 2", SyntaxError, {}),  # mixed: assignment then pure
        ("obj.attr = 1", SyntaxError, {}),  # attribute target
        ("df[0] = 1", SyntaxError, {}),  # subscript target
        ("A, B = 1, 2", SyntaxError, {}),  # tuple unpacking
        ("a = b = 1", SyntaxError, {}),  # chained assignment
        ("A ^ B", SyntaxError, {}),  # unsupported bitwise op
        ("A is None", SyntaxError, {}),  # `is` unsupported
        ("@x = A + 1", SyntaxError, {}),  # @local target rejected
        ("", ValueError, {}),  # blank expr rejected
        ("   ", ValueError, {}),  # whitespace-only rejected
    ],
)
def test_eval_errors(expr, error, kwargs):
    with pytest.raises(error):
        _lf().eval(expr, **kwargs)


@pytest.mark.parametrize(
    "expr, error",
    [
        ("A", ValueError),  # non-boolean column
        ("True", ValueError),  # scalar, not a LiteFrame
        ("A > 2\nB < 50", SyntaxError),  # multi-statement
    ],
)
def test_query_errors(expr, error):
    with pytest.raises(error):
        _lf().query(expr)


def test_eval_local_dict_takes_precedence():
    threshold = 1  # noqa: F841  caller scope; must be shadowed
    result = _lf().eval("A > @threshold", local_dict={"threshold": 3})
    inner = _inner_expr(result)
    assert inner.right.value == 3


def test_cross_source_isin_mask_raises():
    left = LiteFrame({"a": [1, 2, 3], "b": [10, 20, 30]})
    right = LiteFrame({"a": [1, 5], "c": [100, 200]})
    with pytest.raises(SourceCompatibilityError):
        left[right.eval("a in [1]")]


def test_eval_local_dict_shadows_global_dict():
    result = _lf().eval("A > @v", local_dict={"v": 1}, global_dict={"v": 2})
    assert _inner_expr(result).right.value == 1


def test_query_not_keyword_matches_invert():
    lf = _lf()
    not_result = lf.query("not (A > 2)")
    invert_result = lf.query("~(A > 2)")
    assert not_result.op.predicate == invert_result.op.predicate


def test_eval_backtick_assign_target_decoded():
    lf = LiteFrame({"D": [1, 2, 3]})
    result = lf.eval("`C C` = D + 1")
    # Column names are normalized to lowercase; the point is that the backtick
    # target decoded to the original text, not the synthetic _backtick_var_<hex> id.
    assert "c c" in result.columns
    assert not any(c.startswith("_backtick_var_") for c in result.columns)


def test_eval_inplace_rejected_on_liteframe_data():
    lf = _lf()
    data = lf.data
    with pytest.raises(NotImplementedError, match="LiteFrameData"):
        data.eval("C = A + B", inplace=True)
    with pytest.raises(NotImplementedError, match="LiteFrameData"):
        data.query("A > 2", inplace=True)
    with pytest.raises(NotImplementedError, match="LiteFrameData"):
        data["C"] = data["A"]
