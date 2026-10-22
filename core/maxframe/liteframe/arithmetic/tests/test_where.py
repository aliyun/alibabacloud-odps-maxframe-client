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

from maxframe.liteframe.core import LiteFrame
from maxframe.liteframe.expressions import (
    LiteFrameColumn,
    LiteFrameGt,
    LiteFrameLiteral,
    LiteFrameWhere,
    SourceCompatibilityError,
)
from maxframe.liteframe.initializer import LiteFrame as LF
from maxframe.liteframe.operators.project import LiteFrameProjection
from maxframe.utils import wrap_arrow_dtype

from .conftest import _eval_projection, _inner_expr

# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------


def test_where_structure():
    lf = LF({"a": [1, 2, 3, 4]})
    result = lf.where(lf > 2, -1)
    assert isinstance(result, LiteFrame)
    expr = _inner_expr(result)
    assert isinstance(expr, LiteFrameWhere)
    # where: true_value is the column, false_value is other
    assert isinstance(expr.condition, LiteFrameGt)
    assert isinstance(expr.false_value, LiteFrameLiteral)
    assert expr.false_value.value == -1


def test_mask_structure():
    lf = LF({"a": [1, 2, 3, 4]})
    result = lf.mask(lf > 2, -1)
    expr = _inner_expr(result)
    assert isinstance(expr, LiteFrameWhere)
    # mask: true_value is other, false_value is the column
    assert isinstance(expr.true_value, LiteFrameLiteral)
    assert expr.true_value.value == -1


@pytest.mark.parametrize(
    "other,expected_dtype",
    [
        (np.nan, wrap_arrow_dtype(pa.float64())),
        (-1, None),
    ],
    ids=["default_nan", "int_other"],
)
def test_where_dtype(other, expected_dtype):
    lf = LF({"a": [1, 2, 3, 4]})
    result = lf.where(lf > 2, other)
    if expected_dtype is None:
        assert result.dtypes["a"] == lf.dtypes["a"]
    else:
        assert result.dtypes["a"] == expected_dtype


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["where", "mask"])
@pytest.mark.parametrize(
    "other",
    [-1, 0, np.nan],
    ids=["neg_one", "zero", "nan"],
)
def test_where_mask_evaluation(method, other):
    df = pd.DataFrame({"a": [1, 2, 3, 4], "b": [10, 20, 30, 40]})
    lf = LF(df)
    result = getattr(lf, method)(lf > 2, other)
    evaluated = _eval_projection(result, df)
    expected = getattr(df, method)(df > 2, other)
    # Compare values (dtype may differ for nan case — see test_where_default_nan_dtype)
    for col in df.columns:
        for r, e in zip(evaluated[col], expected[col]):
            if pd.isna(e):
                assert pd.isna(r)
            else:
                assert r == e


# ---------------------------------------------------------------------------
# Source compatibility
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["where", "mask"])
def test_where_mask_cross_source_cond_raises(method):
    """where/mask with a cond from an unrelated LiteFrame raises
    SourceCompatibilityError at graph-construction time (review #1)."""
    left = LF({"a": [1, 2]})
    right = LF({"a": [100, 0]})
    with pytest.raises(SourceCompatibilityError):
        getattr(left, method)(right > 50, -1)


def test_where_cross_source_other_raises():
    """where with a LiteFrame-typed other from an unrelated source raises
    SourceCompatibilityError at graph-construction time (review #1)."""
    lf = LF({"a": [1, 2]})
    other = LF({"a": [-1, -1]})
    with pytest.raises(SourceCompatibilityError):
        lf.where(lf > 0, other)


# ---------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------


def test_where_after_arithmetic_fuses():
    lf = LF({"a": [1, 2, 3]})
    result = (lf + 5).where(lf + 5 > 7, 0)
    op = result.data.op
    assert isinstance(op, LiteFrameProjection)
    assert result.data.inputs[0] is lf._data


# ---------------------------------------------------------------------------
# case_when
# ---------------------------------------------------------------------------


def test_case_when_structure():
    """case_when builds nested Where: Where(c0, r0, Where(c1, r1, col))."""
    lf = LF({"a": [1, 2, 3, 4, 5]})
    result = lf.case_when([(lf > 3, "high"), (lf > 1, "mid")])
    expr = _inner_expr(result)
    # Outer: Where(c0=lf>3, r0="high", inner)
    assert isinstance(expr, LiteFrameWhere)
    assert expr.true_value.value == "high"
    # Inner: Where(c1=lf>1, r1="mid", col)
    inner = expr.false_value
    assert isinstance(inner, LiteFrameWhere)
    assert inner.true_value.value == "mid"
    assert isinstance(inner.false_value, LiteFrameColumn)


def test_case_when_evaluation():
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
    lf = LF(df)
    result = lf.case_when([(lf > 3, 10), (lf > 1, 20)])
    evaluated = _eval_projection(result, df)
    expected = df["a"].case_when([(df["a"] > 3, 10), (df["a"] > 1, 20)])
    assert evaluated["a"].tolist() == expected.tolist()


@pytest.mark.parametrize(
    "replacements,expected_dtype",
    [
        ((10, 20), pa.int64()),
        ((10.0, 20), pa.float64()),
        (("hi", "lo"), pa.string()),
    ],
    ids=["int", "float", "string"],
)
def test_case_when_dtype(replacements, expected_dtype):
    lf = LF({"a": [1, 2, 3, 4, 5]})
    r0, r1 = replacements
    result = lf.case_when([(lf > 3, r0), (lf > 1, r1)])
    assert result.dtypes["a"] == wrap_arrow_dtype(expected_dtype)


def test_case_when_multi_column():
    df = pd.DataFrame({"a": [1, 4, 5], "b": [10, 20, 30]})
    lf = LF(df)
    result = lf.case_when([(lf > 3, 99)])
    evaluated = _eval_projection(result, df)
    expected_a = df["a"].case_when([(df["a"] > 3, 99)])
    expected_b = df["b"].case_when([(df["b"] > 3, 99)])
    assert evaluated["a"].tolist() == expected_a.tolist()
    assert evaluated["b"].tolist() == expected_b.tolist()


def test_case_when_after_arithmetic_fuses():
    lf = LF({"a": [1, 2, 3]})
    result = (lf + 5).case_when([(lf + 5 > 7, 0)])
    op = result.data.op
    assert isinstance(op, LiteFrameProjection)
    assert result.data.inputs[0] is lf._data


def test_case_when_callable_condition():
    """Callable conditions are invoked with lf, matching pandas."""
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
    lf = LF(df)
    result = lf.case_when([(lambda x: x > 3, 10), (lambda x: x > 1, 20)])
    evaluated = _eval_projection(result, df)
    expected = df["a"].case_when([(df["a"] > 3, 10), (df["a"] > 1, 20)])
    assert evaluated["a"].tolist() == expected.tolist()


def test_case_when_callable_replacement():
    """Callable replacements are invoked with lf."""
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
    lf = LF(df)
    result = lf.case_when([(lf > 3, lambda x: x * 100)])
    evaluated = _eval_projection(result, df)
    # Rows where a > 3 get a*100, others keep original
    assert evaluated["a"].tolist() == [1, 2, 3, 400, 500]


def test_case_when_both_callable():
    """Both condition and replacement can be callables."""
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
    lf = LF(df)
    result = lf.case_when([(lambda x: x > 3, lambda x: x * 100)])
    evaluated = _eval_projection(result, df)
    assert evaluated["a"].tolist() == [1, 2, 3, 400, 500]


@pytest.mark.parametrize(
    "bad_caselist,error",
    [
        ([], ValueError),
        ("not_a_list", TypeError),
        ([(1, 2, 3)], ValueError),
    ],
    ids=["empty", "not_list", "bad_tuple"],
)
def test_case_when_errors(bad_caselist, error):
    lf = LF({"a": [1, 2, 3]})
    with pytest.raises(error):
        lf.case_when(bad_caselist)
