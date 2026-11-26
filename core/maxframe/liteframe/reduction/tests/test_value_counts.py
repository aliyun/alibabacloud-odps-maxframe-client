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

"""Unit tests for LiteFrame.value_counts() / groupby.value_counts() compile-time checks."""

import pytest

from maxframe.liteframe.initializer import LiteFrame as LF
from maxframe.liteframe.reduction.tests.conftest import _op_types

_BASE_OPS = {
    "LiteFrameAgg",
    "LiteFrameSort",
    "LiteFrameProjection",
    "LiteFrameFromLocal",
}
# Groupby variant may add a LiteFrameFilter op for the dropna pre-filter
_GB_OPS = _BASE_OPS | {"LiteFrameFilter"}
_NORMALIZE_OPS = _BASE_OPS | {"LiteFrameMerge"}
_GB_NORMALIZE_OPS = _GB_OPS | {"LiteFrameMerge"}

_PLAIN_LF = LF({"a": [1, 2, 2, 3, 3, 3], "b": ["p", "q", "q", "r", "r", "r"]})
_GB_LF = LF({"k": ["x", "x", "y", "y"], "a": [1, 1, 2, 3]})


def _vc(normalize=False, **kw):
    return _PLAIN_LF.value_counts("a", normalize=normalize, **kw)


def _gb_vc(normalize=False, **kw):
    return _GB_LF.groupby("k").value_counts("a", normalize=normalize, **kw)


# ------------------------------------------------------------------ #
# Column naming & default subset                                     #
# ------------------------------------------------------------------ #


@pytest.mark.parametrize("groupby", [False, True], ids=["plain", "groupby"])
@pytest.mark.parametrize("normalize,col", [(False, "count"), (True, "proportion")])
def test_value_counts_column_naming(groupby, normalize, col):
    r = _gb_vc(normalize=normalize) if groupby else _vc(normalize=normalize)
    expected = (
        (["k", "a", col] if groupby else ["a", col])
        if not normalize
        else (["k", "a", col] if groupby else [col, "a"])
    )
    assert set(expected) <= set(r.columns)


def test_value_counts_default_subset_groupby():
    assert list(_GB_LF.groupby("k").value_counts().columns) == ["k", "a", "count"]


# ------------------------------------------------------------------ #
# Operator graph & flags                                             #
# ------------------------------------------------------------------ #


@pytest.mark.parametrize(
    "groupby,normalize,allowed_ops",
    [
        (False, False, _BASE_OPS),
        (False, True, _NORMALIZE_OPS),
        (True, False, _GB_OPS),
        (True, True, _GB_NORMALIZE_OPS),
    ],
    ids=["plain", "plain_normalize", "groupby", "groupby_normalize"],
)
def test_value_counts_operator_graph(groupby, normalize, allowed_ops):
    r = _gb_vc(normalize=normalize) if groupby else _vc(normalize=normalize)
    ops = _op_types(r)
    assert ops <= allowed_ops
    assert ("LiteFrameMerge" in ops) == normalize


@pytest.mark.parametrize("groupby", [False, True], ids=["plain", "groupby"])
@pytest.mark.parametrize("sort,has_sort", [(True, True), (False, False)])
def test_value_counts_sort_op(groupby, sort, has_sort):
    r = _gb_vc(sort=sort) if groupby else _vc(sort=sort)
    assert ("LiteFrameSort" in _op_types(r)) == has_sort


@pytest.mark.parametrize("groupby", [False, True], ids=["plain", "groupby"])
@pytest.mark.parametrize("dropna", [True, False])
def test_value_counts_dropna_compiles(groupby, dropna):
    r = _gb_vc(dropna=dropna) if groupby else _vc(dropna=dropna)
    assert "count" in r.columns


# ------------------------------------------------------------------ #
# Dtypes                                                              #
# ------------------------------------------------------------------ #


@pytest.mark.parametrize(
    "normalize,col,expected_dtype",
    [
        (False, "count", "int"),
        (True, "proportion", "double[pyarrow]"),
    ],
    ids=["count", "proportion"],
)
def test_value_counts_dtype(normalize, col, expected_dtype):
    r = _vc(normalize=normalize)
    assert expected_dtype in str(r.dtypes[col]).lower()


# ------------------------------------------------------------------ #
# Reserved-name validation & selection                               #
# ------------------------------------------------------------------ #


@pytest.mark.parametrize("reserved", ["count", "size"], ids=["count", "size"])
def test_value_counts_reserved_name_in_subset_rejected(reserved):
    lf = LF({"a": [1, 2, 2], reserved: [9, 9, 9]})
    with pytest.raises(ValueError, match="collide"):
        lf.value_counts(["a", reserved])


def test_value_counts_reserved_name_not_in_subset_ok():
    lf = LF({"a": [1, 2, 2], "count": [9, 9, 9]})
    r = lf.value_counts("a")
    assert "count" in r.columns


@pytest.mark.parametrize(
    "lf,selection,subset,match",
    [
        (
            LF({"k": ["x", "x", "y"], "a": [1, 1, 2], "b": [3, 4, 5]}),
            ["a"],
            None,
            None,
        ),
        (
            LF({"k": ["x", "x", "y"], "a": [1, 1, 2], "b": [3, 4, 5]}),
            ["a"],
            "b",
            "do not exist",
        ),
        (
            LF({"k": ["x", "x", "y"], "a": [1, 1, 2]}),
            None,
            "k",
            "cannot be in the groupby column keys",
        ),
    ],
    ids=["selection_ok", "selection_vs_subset_error", "subset_overlap_keys"],
)
def test_groupby_value_counts_selection_rules(lf, selection, subset, match):
    gb = lf.groupby("k")[selection] if selection else lf.groupby("k")
    if match:
        with pytest.raises(ValueError, match=match):
            gb.value_counts(subset)
    else:
        r = gb.value_counts()
        assert "b" not in r.columns
        assert list(r.columns) == ["k", "a", "count"]
