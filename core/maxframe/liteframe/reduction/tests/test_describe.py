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

"""Unit tests for LiteFrame.describe() / groupby.describe() compile-time checks."""

import pandas as pd
import pytest

from maxframe.liteframe.initializer import LiteFrame as LF
from maxframe.liteframe.reduction.tests.conftest import _op_types, _walk

_ALLOWED_DESCRIBE_OPS = {
    "LiteFrameFunctional",
    "LiteFrameAgg",
    "LiteFrameProjection",
    "LiteFrameFromLocal",
}

_DT_LF = LF(
    {
        "num": [1, 2, 3],
        "dt": pd.date_range("2020-01-01", periods=3, freq="180D"),
        "s": ["x", "y", "z"],
    }
)


def _make_lf(extra=None):
    cols = {"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]}
    if extra:
        cols.update(extra)
    return LF(cols)


# ------------------------------------------------------------------ #
# Column selection & percentiles                                      #
# ------------------------------------------------------------------


@pytest.mark.parametrize(
    "lf,expected",
    [
        (_make_lf(), ["statistic", "a", "b"]),
        (_make_lf({"s": ["x", "y", "z"]}), ["statistic", "a", "b"]),
        (LF({"a": [1, 2, 3], "s": ["x", "y", "z"]}), ["statistic", "a"]),
        (LF({"s": ["x", "y", "z"]}), ["statistic"]),
    ],
    ids=["default", "with_string", "single_numeric", "no_numeric"],
)
def test_describe_columns(lf, expected):
    r = lf.describe()
    assert list(r.columns) == expected
    assert str(r.dtypes["statistic"]) == "string"
    if len(expected) > 1:
        assert all(str(r.dtypes[c]) == "float64" for c in expected[1:])
        assert r.shape[1] == len(expected)


def test_describe_custom_percentiles():
    r = _make_lf().describe(percentiles=[0.1, 0.9])
    assert list(r.columns) == ["statistic", "a", "b"]


@pytest.mark.parametrize(
    "percentiles,match",
    [
        ([0.0], "0.0 or 1.0"),
        ([1.0], "0.0 or 1.0"),
        ([0.25, 0.25, 0.5], "duplicates"),
    ],
    ids=["zero", "one", "duplicates"],
)
def test_describe_percentile_validation(percentiles, match):
    with pytest.raises(ValueError, match=match):
        _make_lf().describe(percentiles=percentiles)


def test_describe_percentile_unsorted_silently_sorted():
    """Unsorted percentiles are silently sorted; the statistic row order
    (25% before 75%) is observable via the stat_pairs in the flat_apply args."""
    r = _make_lf().describe(percentiles=[0.75, 0.25, 0.5])
    for node in _walk(r.data):
        op = node.op
        if type(op).__name__ == "LiteFrameFunctional":
            stat_pairs = op.args[1]
            labels = [label for label, _ in stat_pairs]
            assert labels == ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]
            break


# ------------------------------------------------------------------ #
# Operator graph & collision avoidance                                #
# ------------------------------------------------------------------


@pytest.mark.parametrize("groupby", [False, True], ids=["plain", "groupby"])
def test_describe_operator_graph(groupby):
    lf = LF({"k": ["x", "x", "y"], "v": [1, 2, 3]})
    r = lf.groupby("k").describe() if groupby else _make_lf().describe()
    ops = _op_types(r)
    assert ops <= _ALLOWED_DESCRIBE_OPS
    assert {"LiteFrameFunctional", "LiteFrameAgg"} <= ops
    assert "LiteFrameMerge" not in ops


@pytest.mark.parametrize(
    "lf,expected",
    [
        (
            LF({"statistic": [1, 2, 3], "b": [4.0, 5.0, 6.0]}),
            ["statistic_1", "statistic", "b"],
        ),
        (
            LF({"statistic": [1, 2, 3], "statistic_1": [4.0, 5.0, 6.0]}),
            ["statistic_2", "statistic", "statistic_1"],
        ),
        (LF({"k": ["x", "x", "y"], "statistic": [1, 2, 3]}), None),
        (LF({"a_count": ["x", "x"], "a": [1, 2]}), None),
    ],
    ids=["single", "double", "groupby", "key_stat_collision"],
)
def test_describe_stat_col_collision(lf, expected):
    if expected is None and "k" in lf.columns:
        r = lf.groupby("k").describe()
        expected = ["k", "statistic_1", "statistic"]
    elif expected is None:
        r = lf.groupby("a_count").describe()
        expected = ["a_count", "statistic", "a"]
    else:
        r = lf.describe()
    assert list(r.columns) == expected


# ------------------------------------------------------------------ #
# Datetime support                                                     #
# ------------------------------------------------------------------


@pytest.mark.parametrize(
    "lf,expected_cols",
    [
        (_DT_LF, ["statistic", "num", "dt"]),
        (
            LF({"dt": pd.date_range("2020-01-01", periods=3, freq="D")}),
            ["statistic", "dt"],
        ),
        (
            LF(
                {
                    "dt": pd.date_range("2020-01-01", periods=3, freq="D").astype(
                        "datetime64[ns]"
                    )
                }
            ),
            ["statistic", "dt"],
        ),
    ],
    ids=["mixed", "datetime_only", "datetime_ns"],
)
def test_describe_datetime(lf, expected_cols):
    r = lf.describe(numeric_only=False)
    assert list(r.columns) == expected_cols
    in_dtype = str(lf.dtypes["dt"])
    expected_unit = "us" if "us" in in_dtype else "ns"
    assert str(r.dtypes["dt"]) == f"datetime64[{expected_unit}]"
    if "num" in expected_cols:
        assert str(r.dtypes["num"]) == "float64"
        assert "s" not in r.columns


@pytest.mark.parametrize(
    "numeric_only,expected",
    [
        (True, ["k", "statistic", "num"]),
        (False, ["k", "statistic", "num", "dt"]),
    ],
)
def test_groupby_describe_columns(numeric_only, expected):
    lf = LF(
        {
            "k": ["x", "x", "y"],
            "num": [1, 2, 3],
            "dt": pd.date_range("2020-01-01", periods=3, freq="180D"),
        }
    )
    r = lf.groupby("k").describe(numeric_only=numeric_only)
    assert list(r.columns) == expected
    if not numeric_only:
        in_dtype = str(lf.dtypes["dt"])
        expected_unit = "us" if "us" in in_dtype else "ns"
        assert str(r.dtypes["dt"]) == f"datetime64[{expected_unit}]"


def test_describe_groupby_selection_intersection():
    lf = LF({"k": ["x", "x", "y"], "a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
    r = lf.groupby("k")[["a"]].describe()
    assert list(r.columns) == ["k", "statistic", "a"]
    assert "b" not in r.columns


def test_describe_groupby_multi_key_stat_col_collision():
    """Regression: two group keys named 'a_count' and 'a_count_1' must both
    be preserved when the generated stat_col 'a_count' (from count(a))
    collides with both.  The result must have columns
    ['a_count', 'a_count_1', 'statistic', 'a']."""
    lf = LF({"a_count": ["x", "x"], "a_count_1": [9, 9], "a": [1, 2]})
    r = lf.groupby(["a_count", "a_count_1"]).describe()
    assert list(r.columns) == ["a_count", "a_count_1", "statistic", "a"]
