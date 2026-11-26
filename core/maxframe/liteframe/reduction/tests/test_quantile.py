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

"""Unit tests for LiteFrame.quantile() / median() compile and sketch primitives."""

import pytest

from maxframe.liteframe.initializer import LiteFrame as LF
from maxframe.liteframe.operators.project import LiteFrameProjection
from maxframe.liteframe.reduction.quantile import _format_q_suffix


def _get_agg_op(result):
    """Get the LiteFrameAgg operator from a result (unwrap projection)."""
    op = result.data.op
    return op.inputs[0].op if isinstance(op, LiteFrameProjection) else op


@pytest.mark.parametrize(
    "q,expected",
    [
        (0.5, "q50"),
        (0.25, "q25"),
        (0.125, "q12.5"),
        (0.0, "q0"),
        (1.0, "q100"),
        (0.1, "q10"),
        (0.025, "q2.5"),
    ],
)
def test_format_q_suffix(q, expected):
    assert _format_q_suffix(q) == expected


@pytest.mark.parametrize(
    "q,expected_cols",
    [
        (0.5, ["a", "b"]),
        ([0.25, 0.5], ["a_q25", "a_q50", "b_q25", "b_q50"]),
        ([0.125, 0.5], ["a_q12.5", "a_q50", "b_q12.5", "b_q50"]),
    ],
)
def test_quantile_compile_columns(q, expected_cols):
    lf = LF({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
    assert list(lf.quantile(q).columns) == expected_cols


@pytest.mark.parametrize("approx", [False, True], ids=["exact", "approx"])
def test_quantile_compile_uniform_agg(approx):
    """quantile always builds a uniform LiteFrameAgg (tiler delegates)."""
    lf = LF({"v": [1.0, 2.0, 3.0]})
    desc = _get_agg_op(lf.quantile(0.5, approx=approx)).agg_funcs[0]
    assert desc["raw_func_name"] == "quantile"
    assert desc.get("kwds", {}).get("approx") == approx
    if approx:
        assert desc["map_func_name"] == "quantile_sketch"
        assert desc.get("sketch") is True
        assert desc.get("passthrough") is False


@pytest.mark.parametrize(
    "interp,expected_dtype",
    [
        ("linear", "double[pyarrow]"),
        ("midpoint", "double[pyarrow]"),
        ("lower", "int64[pyarrow]"),
        ("higher", "int64[pyarrow]"),
        ("nearest", "int64[pyarrow]"),
    ],
)
def test_quantile_dtype(interp, expected_dtype):
    lf = LF({"k": ["x", "y", "z"], "v": [1, 2, 3]})
    dt = lf.groupby("k").quantile(0.5, interpolation=interp).dtypes.to_dict()
    assert str(dt["v"]) == expected_dtype


def test_median_compile_plain_names():
    lf = LF({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
    assert list(lf.median().columns) == ["a", "b"]


def test_groupby_median_compile():
    lf = LF({"k": ["x", "y"], "v": [1, 2]})
    assert list(lf.groupby("k").median().columns) == ["k", "v"]


@pytest.mark.parametrize("groupby", [False, True], ids=["plain", "groupby"])
def test_quantile_invalid_interpolation(groupby):
    lf = LF({"k": ["x", "y"], "v": [1, 2]})
    target = lf.groupby("k") if groupby else lf
    with pytest.raises(ValueError, match="Invalid interpolation"):
        if groupby:
            target.quantile(0.5, interpolation="bad")
        else:
            target[["v"]].quantile(0.5, interpolation="bad")
