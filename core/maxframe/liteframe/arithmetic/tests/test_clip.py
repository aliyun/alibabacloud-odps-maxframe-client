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
    LiteFrameGt,
    LiteFrameLt,
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


def test_clip_both_bounds_returns_liteframe():
    lf = LF({"a": [1, 2, 3, 4], "b": [10, 20, 30, 40]})
    result = lf.clip(lower=2, upper=3)
    assert isinstance(result, LiteFrame)
    assert list(result.columns) == ["a", "b"]
    assert isinstance(_inner_expr(result), LiteFrameWhere)


def test_clip_both_bounds_structure():
    """clip(lower, upper) → Where(col < lower, lower, Where(col > upper, upper, col))."""
    lf = LF({"a": [1, 2, 3, 4]})
    result = lf.clip(lower=2, upper=3)
    expr = _inner_expr(result)
    # Outer: where(col < lower, lower, inner)
    assert isinstance(expr.condition, LiteFrameLt)
    assert expr.true_value.value == 2
    inner = expr.false_value
    assert isinstance(inner, LiteFrameWhere)
    assert isinstance(inner.condition, LiteFrameGt)
    assert inner.true_value.value == 3


def test_clip_lower_only_structure():
    lf = LF({"a": [1, 2, 3]})
    result = lf.clip(lower=2)
    expr = _inner_expr(result)
    assert isinstance(expr, LiteFrameWhere)
    assert isinstance(expr.condition, LiteFrameLt)
    assert expr.true_value.value == 2


def test_clip_upper_only_structure():
    lf = LF({"a": [1, 2, 3]})
    result = lf.clip(upper=2)
    expr = _inner_expr(result)
    assert isinstance(expr, LiteFrameWhere)
    assert isinstance(expr.condition, LiteFrameGt)
    assert expr.true_value.value == 2


def test_clip_preserves_dtype():
    lf = LF({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
    result = lf.clip(lower=2, upper=5)
    assert result.dtypes["a"] == lf.dtypes["a"]
    assert result.dtypes["b"] == lf.dtypes["b"]


def test_clip_scalar_bound_preserves_float32():
    """A float32 column clipped by a scalar bound stays float32 — the scalar
    is coerced to the column dtype, not the reverse (review follow-up:
    promotion was applied unconditionally, widening to double)."""
    lf = LF(pd.DataFrame({"a": pd.Series([-1.0, 2.0], dtype="float32")}))
    result = lf.clip(lower=0)
    assert str(result.dtypes["a"]) == "float[pyarrow]"
    evaluated = _eval_projection(
        result, pd.DataFrame({"a": pd.Series([-1.0, 2.0], dtype="float32")})
    )
    assert str(evaluated["a"].dtype) == "float32"
    assert evaluated["a"].tolist() == [0.0, 2.0]


def test_clip_int_column_float_scalar_promotes_to_float64():
    """An int64 column clipped by a float scalar promotes to float64 —
    metadata, physical dtype, and values all agree (review follow-up:
    metadata said int64 while the physical result was float64 [0.5, 2.0])."""
    lf = LF(pd.DataFrame({"a": pd.Series([-1, 2], dtype="int64")}))
    result = lf.clip(lower=0.5)
    assert str(result.dtypes["a"]) == "double[pyarrow]"
    evaluated = _eval_projection(
        result, pd.DataFrame({"a": pd.Series([-1, 2], dtype="int64")})
    )
    assert str(evaluated["a"].dtype) == "float64"
    assert evaluated["a"].tolist() == [0.5, 2.0]


def test_clip_integer_valued_float_scalar_keeps_int():
    """An int column clipped by an integer-valued float scalar stays int64
    (0.0 → int64), while a non-integer float promotes to float64 (0.5 →
    float64) — matching pandas value-based casting (review follow-up:
    kind-based rules promoted 0.0 → float64)."""
    lf = LF(pd.DataFrame({"a": pd.Series([-1, 2], dtype="int64")}))
    # 0.0 is integer-valued → int64
    result = lf.clip(lower=0.0)
    assert "int" in str(result.dtypes["a"])
    evaluated = _eval_projection(
        result, pd.DataFrame({"a": pd.Series([-1, 2], dtype="int64")})
    )
    assert str(evaluated["a"].dtype) == "int64"
    assert evaluated["a"].tolist() == [0, 2]


def test_clip_scalar_range_widens_int_dtype():
    """An int8 column clipped by a scalar outside int8 range widens to the
    smallest fitting int (int8 + 1000 → int16), matching pandas (review
    follow-up: metadata stayed int8 while physical was int16)."""
    lf = LF(pd.DataFrame({"a": pd.Series([-1, 2], dtype="int8")}))
    result = lf.clip(lower=1000)
    assert "int16" in str(result.dtypes["a"])
    evaluated = _eval_projection(
        result, pd.DataFrame({"a": pd.Series([-1, 2], dtype="int8")})
    )
    assert str(evaluated["a"].dtype) == "int16"
    assert evaluated["a"].tolist() == [1000, 1000]


def test_clip_nan_scalar_treated_as_absent():
    """A NaN scalar bound is treated as "no bound" — the column dtype and
    values are unchanged, matching pandas (review follow-up: NaN turned
    int64 into float64 [NaN, NaN])."""
    lf = LF(pd.DataFrame({"a": pd.Series([-1, 2], dtype="int64")}))
    result = lf.clip(lower=np.nan)
    assert "int" in str(result.dtypes["a"])
    evaluated = _eval_projection(
        result, pd.DataFrame({"a": pd.Series([-1, 2], dtype="int64")})
    )
    assert str(evaluated["a"].dtype) == "int64"
    assert evaluated["a"].tolist() == [-1, 2]


def test_clip_no_bounds_raises():
    lf = LF({"a": [1, 2, 3]})
    with pytest.raises(ValueError):
        lf.clip()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "df_data,lower,upper",
    [
        ({"a": [1, 2, 3, 4], "b": [10, 20, 30, 40]}, 2, 3),
        ({"a": [1, 2, 3, 4], "b": [10, 20, 30, 40]}, None, 3),
        ({"a": [1, 2, 3, 4], "b": [10, 20, 30, 40]}, 2, None),
        ({"a": [-1.5, 0.0, 2.5]}, 0.0, 2.0),
    ],
    ids=["both", "upper_only", "lower_only", "floats"],
)
def test_clip_evaluation(df_data, lower, upper):
    df = pd.DataFrame(df_data)
    lf = LF(df)
    result = lf.clip(lower=lower, upper=upper)
    evaluated = _eval_projection(result, df)
    expected = df.clip(lower=lower, upper=upper)
    pd.testing.assert_frame_equal(evaluated, expected)


def test_clip_inverted_scalar_bounds_matches_pandas():
    """clip(lower=3, upper=1) swaps the bounds and matches pandas [1,2,3]
    rather than clamping every value to the lower branch (review #3)."""
    df = pd.DataFrame({"a": [0, 2, 4]})
    lf = LF(df)
    result = lf.clip(lower=3, upper=1)
    evaluated = _eval_projection(result, df)
    pd.testing.assert_frame_equal(evaluated, df.clip(lower=3, upper=1))
    assert evaluated["a"].tolist() == [1, 2, 3]


def test_clip_inverted_numpy_scalar_bounds_matches_pandas():
    """clip(lower=np.int64(3), upper=np.int64(1)) normalizes NumPy scalars to
    Python scalars, swaps, and matches pandas [1,2,3] (review #3 follow-up:
    np.int64 raised TypeError building the literal)."""
    df = pd.DataFrame({"a": [0, 2, 4]})
    lf = LF(df)
    result = lf.clip(lower=np.int64(3), upper=np.int64(1))
    evaluated = _eval_projection(result, df)
    assert evaluated["a"].tolist() == [1, 2, 3]


# ---------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------


def test_clip_after_arithmetic_fuses():
    """clip on an arithmetic result should fuse into a single projection."""
    lf = LF({"a": [1, 2, 3]})
    result = (lf + 10).clip(lower=12, upper=13)
    op = result.data.op
    assert isinstance(op, LiteFrameProjection)
    expr = _inner_expr(result)
    # The false_value of the outer Where should reference the fused Add expr
    # rather than a bare LiteFrameColumn.
    assert isinstance(expr, LiteFrameWhere)
    inner = expr.false_value
    assert isinstance(inner, LiteFrameWhere)
    # inner.false_value is the original column or the fused add expression
    assert inner.false_value is not None


# ---------------------------------------------------------------------------
# LiteFrame expression bounds
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("lower", "upper", "pd_lower", "pd_upper"),
    [
        # lower=expr, upper=scalar
        ("expr2", 4, 2, 4),
        # lower=scalar, upper=expr
        (2, "expr4", 2, 4),
        # both expr bounds
        ("expr2", "expr4", 2, 4),
        # lower-only expr bound
        ("expr2", None, 2, None),
    ],
    ids=["lower_expr", "upper_expr", "both_expr", "lower_only_expr"],
)
def test_clip_expr_bounds_matches_pandas(lower, upper, pd_lower, pd_upper):
    """clip accepts same-source LiteFrame bounds; per-column bounds are
    matched by column name and the result matches pandas."""
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [10, 20, 30, 40, 50]})
    lf = LF(df)
    # "expr2"/"expr4" are same-source LiteFrame bounds (all-2s / all-4s).
    const = {"expr2": (lf * 0) + 2, "expr4": (lf * 0) + 4}
    result = lf.clip(lower=const.get(lower, lower), upper=const.get(upper, upper))
    evaluated = _eval_projection(result, df)
    pd.testing.assert_frame_equal(evaluated, df.clip(lower=pd_lower, upper=pd_upper))


@pytest.mark.parametrize("which", ["lower", "upper"])
def test_clip_cross_source_bound_raises(which):
    """clip with a bound from an unrelated LiteFrame raises
    SourceCompatibilityError (no implicit alignment)."""
    lf = LF({"a": [1, 2, 3]})
    other = LF({"a": [2, 2, 2]})
    kwargs = (
        {"lower": other, "upper": 4}
        if which == "lower"
        else {
            "lower": 2,
            "upper": other,
        }
    )
    with pytest.raises(SourceCompatibilityError):
        lf.clip(**kwargs)


def test_clip_liteframe_bound_promotes_dtype():
    """An int column clipped by a float LiteFrame bound yields a float
    LiteFrame (metadata matches the physical result and pandas)."""
    lf = LF({"a": [1, 2]})
    result = lf.clip(lower=lf + 0.5)
    assert result.dtypes["a"] == wrap_arrow_dtype(pa.float64())
    evaluated = _eval_projection(result, pd.DataFrame({"a": [1, 2]}))
    pd.testing.assert_series_equal(evaluated["a"], pd.Series([1.5, 2.5], name="a"))


def test_clip_liteframe_bound_missing_column_raises():
    """A LiteFrame bound whose columns don't include the target raises
    KeyError at build time rather than silently becoming a no-op."""
    lf = LF({"a": [1, 2]})
    lower = (lf + 1).rename(columns={"a": "x"})
    with pytest.raises(KeyError, match="no column named"):
        lf.clip(lower=lower)


def test_clip_float32_out_of_range_scalar_promotes_to_float64():
    """A float32 column clipped by a finite scalar beyond float32 range
    promotes to float64 (pandas: float32 + 1e40 → Float64, not Float32
    with [inf, inf] — review follow-up: out-of-range finite bound
    silently overflowed)."""
    df = pd.DataFrame({"a": pd.Series([-1.0, 2.0], dtype="float32")})
    result = LF(df).clip(lower=1e40)
    assert result.dtypes["a"] == wrap_arrow_dtype(pa.float64())
    evaluated = _eval_projection(result, df)
    assert str(evaluated["a"].dtype) == "float64"
    assert evaluated["a"].tolist() == [1e40, 1e40]


@pytest.mark.parametrize("which", ["lower", "upper"])
def test_clip_infinity_scalar_handled(which):
    """Infinity bounds don't crash at build time (np.isfinite guard before
    int(val)) and match pandas: +inf lower promotes to float64 [inf,inf];
    -inf lower is a no-op keeping int64 [-1,2] (review follow-up: int(col)
    + inf raised OverflowError)."""
    df = pd.DataFrame({"a": [-1, 2]})
    kwargs = {which: np.inf if which == "lower" else -np.inf}
    # Should not raise at build time.
    evaluated = _eval_projection(LF(df).clip(**kwargs), df)
    pd.testing.assert_series_equal(evaluated["a"], df.clip(**kwargs)["a"])


def test_clip_unsigned_widening_preserves_signedness():
    """An unsigned column clipped by a scalar outside its range widens to
    the unsigned counterpart (uint8 + 1000 → uint16, not int16) — matching
    numpy/Polars (review follow-up: np.min_scalar_type always returned
    signed dtypes, so metadata/physical diverged)."""
    df = pd.DataFrame({"a": pd.Series([0, 2], dtype="uint8")})
    result = LF(df).clip(lower=1000)
    assert result.dtypes["a"] == wrap_arrow_dtype(pa.uint16())
    assert _eval_projection(result, df)["a"].tolist() == [1000, 1000]
