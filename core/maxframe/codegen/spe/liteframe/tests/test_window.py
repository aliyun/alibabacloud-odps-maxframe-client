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
from maxframe.codegen.spe.liteframe.window import LiteFrameWindowTransformAdapter
from maxframe.liteframe.operators.window import LiteFrameWindowTransform


def test_window_adapter_code_generation(make_liteframe):
    """LiteFrameWindowTransformAdapter should generate _run_window_transform call."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    data = make_liteframe(df)
    op = LiteFrameWindowTransform(
        kind="shift",
        periods=1,
        columns=["a", "b"],
    )
    out = op(data)
    op = out.op
    adapter = LiteFrameWindowTransformAdapter()
    context = SPECodeContext()
    result = adapter.generate_code(op, context)
    assert len(result) == 1
    assert "LiteFrameWindowTransformAdapter._run_window_transform" in result[0]


def test_window_run_shift():
    """Test _run_window_transform with shift."""
    df = pd.DataFrame({"a": [1, 2, 3, 4]})
    result = LiteFrameWindowTransformAdapter._run_window_transform(
        df,
        kind="shift",
        groupby_params=None,
        periods=1,
    )
    assert result["a"].iloc[0] is None or pd.isna(result["a"].iloc[0])
    assert list(result["a"].iloc[1:]) == [1, 2, 3]


def test_window_run_shift_fill_value():
    """Test _run_window_transform with shift and fill_value."""
    df = pd.DataFrame({"a": [1, 2, 3]})
    result = LiteFrameWindowTransformAdapter._run_window_transform(
        df,
        kind="shift",
        groupby_params=None,
        periods=1,
        fill_value=0,
    )
    assert result["a"].iloc[0] == 0
    assert list(result["a"].iloc[1:]) == [1, 2]


def test_window_run_diff():
    """Test _run_window_transform with diff."""
    df = pd.DataFrame({"a": [1, 3, 6, 10]})
    result = LiteFrameWindowTransformAdapter._run_window_transform(
        df,
        kind="diff",
        groupby_params=None,
        periods=1,
    )
    assert pd.isna(result["a"].iloc[0])
    assert list(result["a"].iloc[1:]) == [2, 3, 4]


def test_window_run_pct_change():
    """Test _run_window_transform with pct_change."""
    df = pd.DataFrame({"a": [10, 20, 40]})
    result = LiteFrameWindowTransformAdapter._run_window_transform(
        df,
        kind="pct_change",
        groupby_params=None,
        periods=1,
    )
    assert pd.isna(result["a"].iloc[0])
    assert result["a"].iloc[1] == 1.0
    assert result["a"].iloc[2] == 1.0


@pytest.mark.parametrize(
    "agg_func, values, expected",
    [
        ("sum", [1, 2, 3, 4], [None, 3, 5, 7]),
        ("mean", [2, 4, 6, 8], [None, 3.0, 5.0, 7.0]),
    ],
)
def test_window_run_rolling(agg_func, values, expected):
    """Rolling aggregation over a window of 2 (sum / mean)."""
    df = pd.DataFrame({"a": values})
    result = LiteFrameWindowTransformAdapter._run_window_transform(
        df,
        kind="rolling_agg",
        groupby_params=None,
        window_size=2,
        agg_func=agg_func,
    )
    assert [v if not pd.isna(v) else None for v in result["a"].tolist()] == expected


@pytest.mark.parametrize(
    "selection, expected_cols",
    [(None, ["val"]), (["key", "val"], ["key", "val"])],
    ids=["no_selection", "selection_keeps_keys"],
)
def test_window_run_grouped_shift(selection, expected_cols):
    """Grouped shift: without an explicit ``gb[...]`` selection only the
    transformed columns are emitted (pandas window semantics -- the grouping
    keys never leak); a selection keeps the selected key columns."""
    df = pd.DataFrame({"key": [1, 1, 2, 2], "val": [1, 2, 3, 4]})
    groupby_params = {"by": "key", "dropna": True}
    if selection is not None:
        groupby_params["selection"] = selection
    result = LiteFrameWindowTransformAdapter._run_window_transform(
        df,
        kind="shift",
        groupby_params=groupby_params,
        periods=1,
    )
    assert list(result.columns) == expected_cols
    assert pd.isna(result["val"].iloc[0])
    assert result["val"].iloc[1] == 1
    assert pd.isna(result["val"].iloc[2])
    assert result["val"].iloc[3] == 3


def test_window_shift_excludes_virtual_range_column():
    """A virtual range column must stay virtual, not be transformed.

    Regression: the adapter used to materialize the range column and then
    shift it (corrupting it). Range columns are excluded from transform
    targets, matching the DPE executor which keeps them virtual for
    standalone (non-grouped) transforms.
    """
    df = pd.DataFrame({"a": [10, 20, 30], "b": [1, 2, 3]})
    range_columns = {"__lf_range_index__": (0, 3, 1)}
    result = LiteFrameWindowTransformAdapter._run_window_transform(
        df,
        kind="shift",
        groupby_params=None,
        periods=1,
        fill_value=0,
        range_columns=range_columns,
    )
    # Range column is excluded: only physical columns remain in the output.
    assert list(result.columns) == ["a", "b"]
    assert result["a"].tolist() == [0, 10, 20]
    assert result["b"].tolist() == [0, 1, 2]


def test_window_shift_partial_columns_with_range():
    """Shifting a subset of columns preserves the others and the range col."""
    df = pd.DataFrame({"a": [10, 20, 30], "b": [1, 2, 3]})
    range_columns = {"__lf_range_index__": (0, 3, 1)}
    result = LiteFrameWindowTransformAdapter._run_window_transform(
        df,
        kind="shift",
        groupby_params=None,
        periods=1,
        fill_value=0,
        columns=["a"],
        range_columns=range_columns,
    )
    assert list(result.columns) == ["a", "b"]
    assert result["a"].tolist() == [0, 10, 20]
    assert result["b"].tolist() == [1, 2, 3]


# ---------------------------------------------------------------------------
# expanding_agg (cum* / expanding) — added with the cum*/expanding operators
# ---------------------------------------------------------------------------


def test_window_run_cumsum():
    """cumsum via expanding_agg (remask=True)."""
    df = pd.DataFrame({"a": [1, 2, 3, 4]})
    result = LiteFrameWindowTransformAdapter._run_window_transform(
        df,
        "expanding_agg",
        None,
        agg_func="sum",
        min_periods=1,
        remask_input_nan=True,
    )
    assert result["a"].tolist() == [1, 3, 6, 10]


def test_window_run_cumsum_skipna_false():
    """cumsum with skipna=False poisons the rest of the accumulation."""
    df = pd.DataFrame({"a": [1.0, None, 3.0, 4.0]})
    result = LiteFrameWindowTransformAdapter._run_window_transform(
        df,
        "expanding_agg",
        None,
        agg_func="sum",
        min_periods=1,
        remask_input_nan=True,
        skipna=False,
    )
    vals = result["a"].tolist()
    assert vals[0] == 1.0
    assert all(pd.isna(v) for v in vals[1:])


def test_window_run_expanding_sum_fill():
    """expanding sum uses fill semantics (NaN treated as 0)."""
    df = pd.DataFrame({"a": [1.0, None, 3.0, 4.0]})
    result = LiteFrameWindowTransformAdapter._run_window_transform(
        df,
        "expanding_agg",
        None,
        agg_func="sum",
        min_periods=1,
        remask_input_nan=False,
    )
    assert result["a"].tolist() == [1.0, 1.0, 4.0, 8.0]


def test_window_run_cumcount():
    """cumcount (row_count_mode=True) yields 0-indexed row positions."""
    df = pd.DataFrame({"a": [10, 20, 30]})
    result = LiteFrameWindowTransformAdapter._run_window_transform(
        df,
        "expanding_agg",
        None,
        agg_func="count",
        min_periods=1,
        remask_input_nan=True,
        row_count_mode=True,
    )
    assert result["a"].tolist() == [0, 1, 2]


def test_window_run_expanding_mean():
    """expanding mean on [1,2,3,4]."""
    df = pd.DataFrame({"a": [1, 2, 3, 4]})
    result = LiteFrameWindowTransformAdapter._run_window_transform(
        df,
        "expanding_agg",
        None,
        agg_func="mean",
        min_periods=1,
        remask_input_nan=False,
    )
    assert result["a"].tolist() == [1.0, 1.5, 2.0, 2.5]


def test_window_run_expanding_std():
    """expanding std on [1,2,3,4]."""
    df = pd.DataFrame({"a": [1, 2, 3, 4]})
    result = LiteFrameWindowTransformAdapter._run_window_transform(
        df,
        "expanding_agg",
        None,
        agg_func="std",
        min_periods=1,
        remask_input_nan=False,
    )
    assert result["a"].iloc[0] != result["a"].iloc[0]  # NaN
    assert round(result["a"].iloc[1], 6) == round(0.707107, 6)
    assert round(result["a"].iloc[3], 6) == round(1.290994, 6)


def test_window_run_expanding_min_periods():
    """min_periods=2 masks the first position."""
    df = pd.DataFrame({"a": [1, 2, 3, 4]})
    result = LiteFrameWindowTransformAdapter._run_window_transform(
        df,
        "expanding_agg",
        None,
        agg_func="sum",
        min_periods=2,
        remask_input_nan=False,
    )
    assert result["a"].iloc[0] != result["a"].iloc[0]  # NaN
    assert result["a"].tolist()[1:] == [3.0, 6.0, 10.0]


def test_window_run_grouped_cumsum():
    """Grouped cumsum via expanding_agg."""
    df = pd.DataFrame({"k": ["a", "a", "b", "b"], "v": [1, 2, 3, 4]})
    result = LiteFrameWindowTransformAdapter._run_window_transform(
        df,
        "expanding_agg",
        {"by": "k", "dropna": True},
        agg_func="sum",
        min_periods=1,
        remask_input_nan=True,
        columns=["v"],
    )
    assert result["v"].tolist() == [1, 3, 3, 7]


def test_window_run_cumsum_order_by():
    """order_by sorts, accumulates, then places back to original order."""
    df = pd.DataFrame({"a": [1, 2, 3], "o": [3, 1, 2]})
    result = LiteFrameWindowTransformAdapter._run_window_transform(
        df,
        "expanding_agg",
        None,
        agg_func="sum",
        min_periods=1,
        remask_input_nan=True,
        order_by=["o"],
        order_by_ascending=[True],
    )
    # sorted by o: [2,3,1] -> cumsum [2,5,6]; place back: [6,2,5]
    assert result["a"].tolist() == [6, 2, 5]


def test_window_run_cumsum_reverse():
    """reverse=True accumulates from the last row backward."""
    df = pd.DataFrame({"a": [1, 2, 3, 4]})
    result = LiteFrameWindowTransformAdapter._run_window_transform(
        df,
        "expanding_agg",
        None,
        agg_func="sum",
        min_periods=1,
        remask_input_nan=True,
        reverse=True,
    )
    assert result["a"].tolist() == [10, 9, 7, 4]


# ---------------------------------------------------------------------------
# ffill / bfill — added with the ffill/bfill window transforms
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["ffill", "bfill"])
def test_window_run_fill(kind):
    """ffill/bfill propagate non-null values across nulls."""
    df = pd.DataFrame({"a": [1.0, None, None, 4.0, None]})
    result = LiteFrameWindowTransformAdapter._run_window_transform(df, kind, None)
    if kind == "ffill":
        assert result["a"].tolist()[:3] == [1.0, 1.0, 1.0]
        assert result["a"].tolist()[3:] == [4.0, 4.0]
    else:
        assert result["a"].tolist()[:1] == [1.0]
        assert result["a"].tolist()[1:4] == [4.0, 4.0, 4.0]
        assert pd.isna(result["a"].iloc[4])


@pytest.mark.parametrize(
    "kind, expected",
    [
        # limit=1 fills only the first null after each non-null.
        ("ffill", [1.0, 1.0, None, 4.0, 4.0]),
        ("bfill", [1.0, None, 4.0, 4.0, None]),
    ],
)
def test_window_run_fill_limit(kind, expected):
    """limit caps the number of consecutive nulls filled (ffill/bfill)."""
    df = pd.DataFrame({"a": [1.0, None, None, 4.0, None]})
    result = LiteFrameWindowTransformAdapter._run_window_transform(
        df, kind, None, limit=1
    )
    assert [v if not pd.isna(v) else None for v in result["a"].tolist()] == expected


@pytest.mark.parametrize("kind", ["ffill", "bfill"])
def test_window_run_grouped_fill(kind):
    """Grouped ffill/bfill propagate within each group."""
    df = pd.DataFrame({"k": [1, 1, 2, 2], "v": [1.0, None, None, 4.0]})
    result = LiteFrameWindowTransformAdapter._run_window_transform(
        df, kind, {"by": "k", "dropna": True}, columns=["v"]
    )
    if kind == "ffill":
        assert result["v"].tolist()[:2] == [1.0, 1.0]
        assert pd.isna(result["v"].iloc[2])
        assert result["v"].iloc[3] == 4.0
    else:
        assert result["v"].iloc[0] == 1.0
        assert pd.isna(result["v"].iloc[1])
        assert result["v"].tolist()[2:] == [4.0, 4.0]


def test_window_run_fill_partial_columns():
    """columns= narrows fill targets; other columns pass through."""
    df = pd.DataFrame({"a": [1.0, None], "b": [None, 2.0]})
    result = LiteFrameWindowTransformAdapter._run_window_transform(
        df, "ffill", None, columns=["a"]
    )
    assert result["a"].tolist() == [1.0, 1.0]
    assert pd.isna(result["b"].iloc[0])
    assert result["b"].iloc[1] == 2.0


def test_window_fill_excludes_range_column():
    """Range columns are excluded from fill targets (stay virtual)."""
    df = pd.DataFrame({"a": [1.0, None]})
    result = LiteFrameWindowTransformAdapter._run_window_transform(
        df,
        "ffill",
        None,
        range_columns={"__lf_range_index__": (0, 2, 1)},
    )
    assert list(result.columns) == ["a"]
    assert result["a"].tolist() == [1.0, 1.0]


# ---------------------------------------------------------------------------
# Regressions for AI-review comments C3-C6 (grouped dropna row preservation,
# expanding window_shift direction, rolling order_by, rolling window_shift).
# ---------------------------------------------------------------------------


def test_grouped_expanding_dropna_preserves_null_key_rows():
    """C3: grouped expanding with dropna=True keeps null-key rows (null target)
    instead of dropping them -- row-count-preserving scatter-back."""
    df = pd.DataFrame({"g": ["a", None, "a"], "v": [1, 2, 3]})
    result = LiteFrameWindowTransformAdapter._run_window_transform(
        df,
        "expanding_agg",
        {"by": "g", "dropna": True},
        agg_func="sum",
        min_periods=1,
        remask_input_nan=False,
        columns=["v"],
    )
    assert len(result) == 3  # all original rows preserved
    assert result["v"].iloc[0] == 1  # group "a", first row
    assert pd.isna(result["v"].iloc[1])  # null-key row -> null
    assert result["v"].iloc[2] == 4  # group "a", cumsum 1+3


def test_expanding_window_shift_positive():
    """C4: expanding(window_shift=1).sum() on [1,2,3,4] yields [3,6,10,10]
    (frame [0, i+shift]), not [NaN,1,3,6] from the old shift(+1)."""
    df = pd.DataFrame({"a": [1, 2, 3, 4]})
    result = LiteFrameWindowTransformAdapter._run_window_transform(
        df,
        "expanding_agg",
        None,
        agg_func="sum",
        min_periods=1,
        remask_input_nan=False,
        window_shift=1,
    )
    assert result["a"].tolist() == [3, 6, 10, 10]


def test_expanding_window_shift_negative():
    """C4: expanding(window_shift=-1).sum() on [1,2,3,4] shifts backward."""
    df = pd.DataFrame({"a": [1, 2, 3, 4]})
    result = LiteFrameWindowTransformAdapter._run_window_transform(
        df,
        "expanding_agg",
        None,
        agg_func="sum",
        min_periods=1,
        remask_input_nan=False,
        window_shift=-1,
    )
    # raw cumsum [1,3,6,10]; shift +1 (backward) -> [NaN,1,3,6]
    assert result["a"].iloc[0] != result["a"].iloc[0]  # NaN
    assert result["a"].iloc[1:].tolist() == [1, 3, 6]


def test_rolling_order_by_places_back():
    """C5: rolling(2, order_by='ord').sum() sorts by ord, rolls, places back
    to original row order."""
    df = pd.DataFrame({"a": [1, 2, 3, 4], "ord": [4, 1, 3, 2]})
    result = LiteFrameWindowTransformAdapter._run_window_transform(
        df,
        "rolling_agg",
        None,
        window_size=2,
        agg_func="sum",
        min_periods=1,
        order_by=["ord"],
        order_by_ascending=[True],
    )
    # sorted by ord: a=[2,4,3,1]; rolling(2).sum=[2,6,7,4]; place back to
    # original order [1,2,3,4] -> [4,2,7,6]
    assert result["a"].tolist() == [4, 2, 7, 6]


def test_rolling_window_shift_positive():
    """C6: rolling(2, window_shift=1).sum() on [1,2,3,4] pre-shifts by -1 then
    rolls, yielding [NaN,5,7,NaN] (min_periods=window_size so edge windows
    are null, matching the reviewer's expected output)."""
    df = pd.DataFrame({"a": [1, 2, 3, 4]})
    result = LiteFrameWindowTransformAdapter._run_window_transform(
        df,
        "rolling_agg",
        None,
        window_size=2,
        agg_func="sum",
        min_periods=2,
        window_shift=1,
    )
    assert pd.isna(result["a"].iloc[0])
    assert result["a"].iloc[1:3].tolist() == [5, 7]
    assert pd.isna(result["a"].iloc[3])


# ---------------------------------------------------------------------------
# Second-pass regressions: duplicate-index scatter, min_periods>1 shift,
# grouped rolling source preservation, combined order_by+window_shift,
# plain grouped rolling fast-path coverage.
# ---------------------------------------------------------------------------


def test_grouped_expanding_duplicate_index_scatter():
    """Second-pass ①: scatter by positional ordinal, not index label.

    With a duplicate index [0, 0, 1], grouped expanding sum must update only
    the row belonging to each group, not every row sharing the label."""
    df = pd.DataFrame({"g": ["a", "b", "a"], "v": [1, 10, 3]}, index=[0, 0, 1])
    result = LiteFrameWindowTransformAdapter._run_window_transform(
        df,
        "expanding_agg",
        {"by": "g", "dropna": True},
        agg_func="sum",
        min_periods=1,
    )
    assert result["v"].tolist() == [1.0, 10.0, 4.0]
    assert result.index.tolist() == [0, 0, 1]


def test_expanding_window_shift_min_periods_gt_1_positive():
    """Second-pass ②: min_periods gates on the shifted raw count, not the
    already-gated output.  v=[1,2,3,4], mp=2, shift=+1 → [3,6,10,10]."""
    df = pd.DataFrame({"a": [1, 2, 3, 4]})
    result = LiteFrameWindowTransformAdapter._run_window_transform(
        df,
        "expanding_agg",
        None,
        agg_func="sum",
        min_periods=2,
        window_shift=1,
    )
    assert result["a"].tolist() == [3.0, 6.0, 10.0, 10.0]


def test_expanding_window_shift_min_periods_gt_1_negative():
    """Second-pass ②: negative shift with min_periods=2.  v=[1,2,3,4],
    mp=2, shift=-1 → [NaN, NaN, 3, 6] (leading edge shrinks before row 0)."""
    df = pd.DataFrame({"a": [1, 2, 3, 4]})
    result = LiteFrameWindowTransformAdapter._run_window_transform(
        df,
        "expanding_agg",
        None,
        agg_func="sum",
        min_periods=2,
        window_shift=-1,
    )
    assert pd.isna(result["a"].iloc[0])
    assert pd.isna(result["a"].iloc[1])
    assert result["a"].iloc[2:].tolist() == [3.0, 6.0]


def test_grouped_rolling_preserves_source_data():
    """Second-pass ③: grouped rolling must not clear the source frame before
    grouping.  g=["a","a","b","b"], v=[1,2,3,4], rolling(2,mp=1).sum()
    → [1, 3, 3, 7]."""
    df = pd.DataFrame({"g": ["a", "a", "b", "b"], "v": [1, 2, 3, 4]})
    result = LiteFrameWindowTransformAdapter._run_window_transform(
        df,
        "rolling_agg",
        {"by": "g", "dropna": True},
        window_size=2,
        min_periods=1,
        agg_func="sum",
    )
    assert result["v"].tolist() == [1.0, 3.0, 3.0, 7.0]


def test_grouped_rolling_order_by_preserves_source_data():
    """Second-pass ③: grouped rolling with order_by keeps source intact
    (MF path: sort → roll → place-back by positional ordinal)."""
    df = pd.DataFrame(
        {
            "g": ["a", "a", "b", "b"],
            "v": [1, 2, 3, 4],
            "ord": [2, 1, 4, 3],
        }
    )
    result = LiteFrameWindowTransformAdapter._run_window_transform(
        df,
        "rolling_agg",
        {"by": "g", "dropna": True},
        window_size=2,
        min_periods=1,
        agg_func="sum",
        order_by=["ord"],
        order_by_ascending=[True],
    )
    # group a: sorted by ord v=[2,1], rolling(2,mp1).sum=[2,3] → place back [3,2]
    # group b: sorted by ord v=[4,3], rolling(2,mp1).sum=[4,7] → place back [7,4]
    assert result["v"].tolist() == [3.0, 2.0, 7.0, 4.0]


def test_rolling_order_by_and_window_shift_combined():
    """Second-pass ③: sort BEFORE shift (DPE order).  v=[10,20,30,40],
    ord=[4,1,3,2], window=2, shift=1, mp=2 → [NaN, NaN, 40, 70]."""
    df = pd.DataFrame({"a": [10, 20, 30, 40], "ord": [4, 1, 3, 2]})
    result = LiteFrameWindowTransformAdapter._run_window_transform(
        df,
        "rolling_agg",
        None,
        window_size=2,
        min_periods=2,
        agg_func="sum",
        order_by=["ord"],
        order_by_ascending=[True],
        window_shift=1,
    )
    assert pd.isna(result["a"].iloc[0])
    assert pd.isna(result["a"].iloc[1])
    assert result["a"].iloc[2:].tolist() == [40.0, 70.0]


def test_grouped_rolling_order_by_and_window_shift_combined():
    """Second-pass ③: grouped + order_by + window_shift combined uses the
    same sort → shift → roll → place-back ordering."""
    df = pd.DataFrame(
        {
            "g": ["a", "a", "a", "a"],
            "v": [10, 20, 30, 40],
            "ord": [4, 1, 3, 2],
        }
    )
    result = LiteFrameWindowTransformAdapter._run_window_transform(
        df,
        "rolling_agg",
        {"by": "g", "dropna": True},
        window_size=2,
        min_periods=2,
        agg_func="sum",
        order_by=["ord"],
        order_by_ascending=[True],
        window_shift=1,
    )
    assert pd.isna(result["v"].iloc[0])
    assert pd.isna(result["v"].iloc[1])
    assert result["v"].iloc[2:].tolist() == [40.0, 70.0]


def test_plain_grouped_rolling_fast_path_coverage():
    """Second-pass ④: plain grouped rolling (no order_by/window_shift) uses
    the vectorized groupby.rolling fast path and stays correct on a
    non-trivial group count."""
    g = np.repeat(np.arange(1000), 100)
    v = np.arange(100000)
    df = pd.DataFrame({"g": g, "v": v})
    result = LiteFrameWindowTransformAdapter._run_window_transform(
        df,
        "rolling_agg",
        {"by": "g", "dropna": True},
        window_size=5,
        min_periods=1,
        agg_func="sum",
    )
    # group 0: [0,1,2,3,4,...] rolling(5,mp1).sum → [0,1,3,6,10]
    assert result["v"].iloc[:5].tolist() == [0.0, 1.0, 3.0, 6.0, 10.0]
    # last group (999) values 99900..99999; first window sum is just 99900
    assert result["v"].iloc[-100] == 99900.0


# ---------------------------------------------------------------------------
# Third-pass regressions: __mf_pos__ column-name collision, count min_periods
# gate under shift, plain grouped rolling duplicate-index crash.
# ---------------------------------------------------------------------------


def test_grouped_expanding_pos_col_name_collision():
    """Third-pass ①: the positional-ordinal temp column must not collide with
    a legitimate user column of the same name.  The user's __mf_pos__ column
    must survive untouched, and the grouped expanding result must be correct."""
    df = pd.DataFrame(
        {"g": ["a", "a", "b"], "v": [1, 2, 3], "__mf_pos__": [101, 102, 103]}
    )
    result = LiteFrameWindowTransformAdapter._run_window_transform(
        df,
        "expanding_agg",
        {"by": "g", "dropna": True},
        agg_func="sum",
        min_periods=1,
        columns=["v"],
    )
    assert result["v"].tolist() == [1.0, 3.0, 3.0]
    assert result["__mf_pos__"].tolist() == [101, 102, 103]


def test_grouped_rolling_mf_path_pos_col_name_collision():
    """Third-pass ①: the grouped-rolling MF path (order_by/window_shift) also
    uses the collision-safe temp column name."""
    df = pd.DataFrame(
        {
            "g": ["a", "a", "b"],
            "v": [1, 2, 3],
            "ord": [2, 1, 1],
            "__mf_pos__": [101, 102, 103],
        }
    )
    result = LiteFrameWindowTransformAdapter._run_window_transform(
        df,
        "rolling_agg",
        {"by": "g", "dropna": True},
        window_size=2,
        min_periods=1,
        agg_func="sum",
        columns=["v"],
        order_by=["ord"],
        order_by_ascending=[True],
    )
    # group a sorted by ord: v=[2,1], rolling(2,mp1).sum=[2,3] → place back [3,2]
    assert result["v"].tolist() == [3.0, 2.0, 3.0]
    assert result["__mf_pos__"].tolist() == [101, 102, 103]


def test_expanding_window_shift_count_min_periods_gate():
    """Third-pass ②: count/cumcount must gate on the shifted row-count under
    window_shift.  v=[1,2,3,4], count, mp=3, shift=+1 → [NaN, 3, 4, 4]."""
    df = pd.DataFrame({"v": [1, 2, 3, 4]})
    result = LiteFrameWindowTransformAdapter._run_window_transform(
        df,
        "expanding_agg",
        None,
        agg_func="count",
        min_periods=3,
        window_shift=1,
    )
    assert pd.isna(result["v"].iloc[0])
    assert result["v"].iloc[1:].tolist() == [3.0, 4.0, 4.0]


def test_expanding_window_shift_count_min_periods_gate_grouped():
    """Third-pass ②: grouped count path also gates on shifted row-count.
    One group g=["a","a","a","a"], same expectation as standalone."""
    df = pd.DataFrame({"g": ["a", "a", "a", "a"], "v": [1, 2, 3, 4]})
    result = LiteFrameWindowTransformAdapter._run_window_transform(
        df,
        "expanding_agg",
        {"by": "g", "dropna": True},
        agg_func="count",
        min_periods=3,
        window_shift=1,
    )
    assert pd.isna(result["v"].iloc[0])
    assert result["v"].iloc[1:].tolist() == [3.0, 4.0, 4.0]


def test_plain_grouped_rolling_duplicate_index():
    """Third-pass ③: plain grouped rolling fast path survives duplicate index
    labels and scatters by position.  g=["a","b","a"], v=[1,10,3],
    index=[0,0,1], rolling(2,mp=1).sum() → [1, 10, 4]."""
    df = pd.DataFrame({"g": ["a", "b", "a"], "v": [1, 10, 3]}, index=[0, 0, 1])
    result = LiteFrameWindowTransformAdapter._run_window_transform(
        df,
        "rolling_agg",
        {"by": "g", "dropna": True},
        window_size=2,
        min_periods=1,
        agg_func="sum",
    )
    assert result["v"].tolist() == [1.0, 10.0, 4.0]
    assert result.index.tolist() == [0, 0, 1]
