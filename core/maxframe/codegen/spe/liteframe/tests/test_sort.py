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
import pytest

from maxframe.codegen.spe.core import SPECodeContext
from maxframe.codegen.spe.liteframe.sort import LiteFrameSortAdapter
from maxframe.liteframe.operators.sort import LiteFrameSort
from maxframe.tests.utils import require_arrow_dtype


def test_sort_adapter_code_generation(make_liteframe):
    """LiteFrameSortAdapter should generate _run_sort call."""
    df = pd.DataFrame({"a": [3, 1, 2], "b": [30, 10, 20]})
    data = make_liteframe(df)
    op = LiteFrameSort(by=["a"], ascending=True, na_position="last")
    out = op(data)
    op = out.op
    adapter = LiteFrameSortAdapter()
    context = SPECodeContext()
    result = adapter.generate_code(op, context)
    assert len(result) == 1
    assert "LiteFrameSortAdapter._run_sort" in result[0]


@pytest.mark.parametrize(
    "df, ascending, nrows, keep_kind, expected_a, expected_len",
    [
        # basic ascending / descending (full frame)
        (
            pd.DataFrame({"a": [3, 1, 2], "b": [30, 10, 20]}),
            True,
            None,
            "head",
            [1, 2, 3],
            3,
        ),
        (
            pd.DataFrame({"a": [3, 1, 2], "b": [30, 10, 20]}),
            False,
            None,
            "head",
            [3, 2, 1],
            3,
        ),
        # nrows + keep_kind=head/first/last on a 5-row frame
        (
            pd.DataFrame({"a": [3, 1, 2, 5, 4], "b": [30, 10, 20, 50, 40]}),
            True,
            2,
            "head",
            [1, 2],
            2,
        ),
        (
            pd.DataFrame({"a": [3, 1, 2, 5, 4], "b": [30, 10, 20, 50, 40]}),
            True,
            3,
            "first",
            [1, 2, 3],
            3,
        ),
        (
            pd.DataFrame({"a": [3, 1, 2, 5, 4], "b": [30, 10, 20, 50, 40]}),
            False,
            3,
            "last",
            [5, 4, 3],
            3,
        ),
        # keep_kind='all' includes all ties at the boundary value
        (
            pd.DataFrame({"a": [1, 2, 2, 2, 3], "b": [10, 20, 30, 40, 50]}),
            True,
            2,
            "all",
            [1, 2, 2, 2],
            4,
        ),
    ],
    ids=["basic", "descending", "nrows_head", "keep_first", "keep_last", "keep_all"],
)
def test_sort_run_sort(df, ascending, nrows, keep_kind, expected_a, expected_len):
    """_run_sort covers ascending/descending, nrows head/first/last/all-keep."""
    result = LiteFrameSortAdapter._run_sort(
        df,
        by=["a"],
        ascending=ascending,
        na_position="last",
        nrows=nrows,
        keep_kind=keep_kind,
    )
    assert len(result) == expected_len
    assert list(result["a"]) == expected_a


# --- nlargest / nsmallest SPE adapter tests ---


@pytest.mark.parametrize(
    "keep, expected_idx, expected_b",
    [
        ("first", [0, 1], ["a", "b"]),
        ("last", [2, 1], ["c", "b"]),
        ("all", [0, 1, 2], ["a", "b", "c"]),
    ],
)
def test_nlargest_keep_ties(keep, expected_idx, expected_b):
    """nlargest tie-breaking matches pandas: first=original, last=reversed, all=include all."""
    df = pd.DataFrame({"a": [5, 5, 5, 1], "b": list("abcd")})
    result = LiteFrameSortAdapter._run_sort(
        df, by=["a"], ascending=False, na_position="last", nrows=2, keep_kind=keep
    )
    assert list(result["b"]) == expected_b


@pytest.mark.parametrize(
    "keep, expected_b",
    [
        ("first", ["d", "b"]),
        ("last", ["d", "c"]),
        ("all", ["d", "b", "c"]),
    ],
)
def test_nsmallest_keep_ties(keep, expected_b):
    """nsmallest tie-breaking matches pandas."""
    df = pd.DataFrame({"a": [5, 1, 1, 0], "b": list("abcd")})
    result = LiteFrameSortAdapter._run_sort(
        df, by=["a"], ascending=True, na_position="last", nrows=2, keep_kind=keep
    )
    assert list(result["b"]) == expected_b


@pytest.mark.parametrize(
    "nrows,expected_ids,expected_nan",
    [
        pytest.param(2, ["a", "c"], 0, id="nan_excluded"),
        pytest.param(4, ["a", "c", "d", "b"], 1, id="nan_included"),
    ],
)
def test_nlargest_nan_placement(nrows, expected_ids, expected_nan):
    """NaN keys follow pandas: excluded below the non-null count, included - in
    first-occurrence order - once nrows reaches it."""
    df = pd.DataFrame({"a": [3.0, np.nan, 2.0, 1.0, np.nan], "b": list("abcde")})
    result = LiteFrameSortAdapter._run_sort(
        df,
        by=["a"],
        ascending=False,
        na_position="last",
        nrows=nrows,
        keep_kind="first",
    )
    assert list(result["b"]) == expected_ids
    assert result["a"].isna().sum() == expected_nan


def test_pad_short_selection_uses_labels_and_sort_order():
    """A short selection (pandas < 2.0 drops the missing keys) is filled with the
    next rows in sort order, addressed by their original labels: the sorted
    frame's positions are not its labels, so slicing it by position duplicates
    rows."""
    df = pd.DataFrame({"a": [3.0, np.nan, 2.0, 1.0, np.nan], "b": list("abcde")})
    pos_df = df.reset_index(drop=True)
    ordered = pos_df.sort_values(
        by=["a"], ascending=False, na_position="last", kind="stable"
    )
    short = ordered.loc[[0, 2, 3]]  # the rows pandas 1.3 keeps: a, c, d
    padded = LiteFrameSortAdapter._pad_short_selection(ordered, short, 4)
    assert list(padded["b"]) == ["a", "c", "d", "b"]
    assert padded["a"].isna().sum() == 1


def test_nlargest_nan_keep_last_first_occurrence():
    """NaN boundary uses first-occurrence for keep='last' (not reversed)."""
    df = pd.DataFrame({"a": [3.0, np.nan, 3.0, 1.0, np.nan], "b": list("abcde")})
    result = LiteFrameSortAdapter._run_sort(
        df, by=["a"], ascending=False, na_position="last", nrows=4, keep_kind="last"
    )
    # Non-NaN 3.0s reversed → [c, a], NaN first-occurrence → [b] (not [e])
    assert list(result["b"]) == ["c", "a", "d", "b"]


def test_nlargest_all_nan_column():
    df = pd.DataFrame({"a": [np.nan, np.nan, np.nan], "b": list("abc")})
    result = LiteFrameSortAdapter._run_sort(
        df, by=["a"], ascending=False, na_position="last", nrows=2, keep_kind="first"
    )
    assert list(result["b"]) == ["a", "b"]


def test_nlargest_multi_column():
    df = pd.DataFrame({"a": [1, 1, 2, 2], "b": [4, 3, 2, 1], "c": list("abcd")})
    result = LiteFrameSortAdapter._run_sort(
        df,
        by=["a", "b"],
        ascending=False,
        na_position="last",
        nrows=2,
        keep_kind="first",
    )
    assert list(result["c"]) == ["c", "d"]


def _arrow_composite_frame(x, y, ids):
    """Arrow-backed frame with nullable Int64 composite keys."""
    return pd.DataFrame(
        {
            "x": pd.array(x, dtype="int64[pyarrow]"),
            "y": pd.array(y, dtype="int64[pyarrow]"),
            "id": ids,
        }
    )


@require_arrow_dtype
@pytest.mark.parametrize(
    "df,expected_ids,expected_y",
    [
        pytest.param(
            _arrow_composite_frame([None] * 4, [None] * 4, list("abcd")),
            ["a", "b"],
            None,
            id="all_keys_missing",
        ),
        pytest.param(
            _arrow_composite_frame([None] * 3, [1, 3, 2], list("abc")),
            ["b", "c"],
            [3, 2],
            id="sort_order",
        ),
        pytest.param(
            _arrow_composite_frame([2, None, None], [1, 1, 1], list("abc")),
            ["a", "b"],
            [1, 1],
            id="complete_key_max",
        ),
    ],
)
def test_nlargest_arrow_composite_keys_fill_short_selection(
    df, expected_ids, expected_y
):
    """Arrow-backed keys make pandas return an empty selection, which is filled
    from the sort order: complete-key rows stay eligible, so the result matches
    descending ``sort().head(n)`` and never depends on the input order."""
    result = LiteFrameSortAdapter._run_sort(
        df,
        by=["x", "y"],
        ascending=False,
        na_position="last",
        nrows=2,
        keep_kind="first",
    )
    assert list(result["id"]) == expected_ids
    if expected_y is not None:
        assert list(result["y"]) == expected_y


def test_nlargest_n_zero():
    df = pd.DataFrame({"a": [1, 2, 3], "b": list("abc")})
    result = LiteFrameSortAdapter._run_sort(
        df, by=["a"], ascending=False, na_position="last", nrows=0, keep_kind="first"
    )
    assert len(result) == 0


def test_nlargest_n_exceeds_len():
    """n > len(df) returns all rows sorted, keep has no effect."""
    df = pd.DataFrame({"a": [5, 5, 1], "b": list("abc")})
    result = LiteFrameSortAdapter._run_sort(
        df, by=["a"], ascending=False, na_position="last", nrows=10, keep_kind="last"
    )
    assert len(result) == 3
    assert list(result["b"]) == ["a", "b", "c"]


def test_nlargest_empty_frame():
    df = pd.DataFrame(
        {"a": pd.Series([], dtype="int64"), "b": pd.Series([], dtype="object")}
    )
    result = LiteFrameSortAdapter._run_sort(
        df, by=["a"], ascending=False, na_position="last", nrows=3, keep_kind="first"
    )
    assert len(result) == 0


def test_nlargest_keep_all_tied_order_matches_stable_sort():
    """keep='all' tied row order matches sort_values(kind='stable')."""
    df = pd.DataFrame({"a": [3, 1, 2, 1, 3], "b": list("abcde")})
    result = LiteFrameSortAdapter._run_sort(
        df, by=["a"], ascending=False, na_position="last", nrows=2, keep_kind="all"
    )
    # result should contain all rows with a >= boundary (3), in original order
    assert list(result["b"]) == ["a", "e"]


def test_nlargest_first_large_tie_group_preserves_occurrence_order():
    """keep='first' with a large equal-key group: the re-sort must use
    kind='stable' so that occurrence order from nlargest/nsmallest is
    preserved. Unstable quicksort scrambles equal-key order."""
    rng = np.random.RandomState(123)
    n = 1000
    df = pd.DataFrame({"a": [5] * n, "b": rng.permutation(n)})
    result = LiteFrameSortAdapter._run_sort(
        df,
        by=["a"],
        ascending=False,
        na_position="last",
        nrows=100,
        keep_kind="first",
    )
    # nlargest(100, keep='first') returns the first 100 rows in occurrence
    # order. The re-sort must preserve that order (stable), not scramble it.
    expected_b = df.head(100)["b"].tolist()
    assert list(result["b"]) == expected_b
