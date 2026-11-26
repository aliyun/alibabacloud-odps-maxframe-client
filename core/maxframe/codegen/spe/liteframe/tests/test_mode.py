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

"""SPE-level tests for LiteFrame.mode() — ModeFreqAgg → flat_apply extraction.

Tests exercise the full SPE pipeline:
1. ``LiteFrameAggAdapter._run_agg`` compiles and executes ``ModeFreqAgg``
   through the custom-agg descriptor path (descriptor built to mirror
   ``compile_custom_agg`` output).
2. ``LiteFrameFunctionalAdapter._run_functional`` runs ``_mode_extract``
   in ``flat_apply`` mode on the packed frequency table.

A multi-chunk combine test verifies the tree-reduction merge stage.
"""

import copy

import pandas as pd
import polars as pl
import pyarrow as pa
import pytest

from maxframe.codegen.spe.liteframe.functional import LiteFrameFunctionalAdapter
from maxframe.codegen.spe.liteframe.reduction import LiteFrameAggAdapter
from maxframe.liteframe.functional.type_infer import infer_output_dtypes
from maxframe.liteframe.reduction.mode import (
    ModeFreqAgg,
    _cnt_col,
    _is_mode_numeric_dtype,
    _mode_extract,
    _val_col,
)
from maxframe.utils import is_arrow_dtype_supported, wrap_arrow_dtype


def _to_arrow_df(df_data):
    """Build an Arrow-backed pandas DataFrame from a dict of lists."""
    cols = {}
    for col, vals in df_data.items():
        s = pd.Series(vals)
        if pd.api.types.is_integer_dtype(s):
            pa_t = pa.int64()
        elif pd.api.types.is_float_dtype(s):
            pa_t = pa.float64()
        elif pd.api.types.is_bool_dtype(s):
            pa_t = pa.bool_()
        else:
            pa_t = pa.string()
            # Don't convert None to 'None'; Arrow string handles NA natively.
            s = pd.Series(
                [None if v is None or (v != v) else str(v) for v in vals],
                dtype=object,
            )
        cols[col] = pd.array(s, dtype=wrap_arrow_dtype(pa_t))
    return pd.DataFrame(cols)


def _run_mode_spe(df_data, dropna=True, numeric_only=False):
    """Run mode() through the full SPE pipeline.

    Builds a custom-agg descriptor mirroring ``compile_custom_agg`` output,
    executes it via ``LiteFrameAggAdapter._run_agg``, then runs
    ``_mode_extract`` via ``LiteFrameFunctionalAdapter._run_functional``
    in ``flat_apply`` mode.
    """
    if not is_arrow_dtype_supported():
        pytest.skip("pandas ArrowDtype is not supported")

    df = _to_arrow_df(df_data)
    all_cols = list(df.columns)

    if numeric_only:
        dtypes = pd.Series({c: df[c].dtype for c in all_cols})
        data_cols = [c for c in all_cols if _is_mode_numeric_dtype(dtypes[c])]
        df = df[data_cols]
    else:
        data_cols = all_cols

    cols = list(df.columns)
    dtypes = pd.Series({c: df[c].dtype for c in cols})

    # Build descriptor (mirrors compile_custom_agg output)
    agg = ModeFreqAgg(dropna=dropna, columns=cols)
    infer_instance = copy.deepcopy(agg)
    inferred = infer_output_dtypes(
        infer_instance.agg, dtypes, input_frame_type="polars"
    )
    agg_output_dtypes = inferred[0]
    agg_output_cols = [_val_col(c) for c in cols] + [_cnt_col(c) for c in cols]

    desc = {
        "raw_func_name": "ModeFreqAgg",
        "map_func_name": "custom_agg",
        "agg_func_name": "custom_agg",
        "cols": cols,
        "agg_output_cols": agg_output_cols,
        "output_key": "agg_0",
        "kwds": {},
        "custom_func": agg,
        "agg_output_dtypes": agg_output_dtypes,
        "input_frame_type": "polars",
        "needs_row_pos": False,
    }

    # Step 1: CustomAgg → packed frequency lists (SPE path)
    freq_df = LiteFrameAggAdapter._run_agg(df, [desc], None, agg_output_cols)

    # Step 2: flat_apply → max + filter + sort + coalesce (SPE path)
    out_dtypes = {col: dtypes[col] for col in cols}
    result = LiteFrameFunctionalAdapter._run_functional(
        freq_df,
        _mode_extract,
        "flat_apply",
        0,
        (cols,),
        None,
        output_dtypes=[pd.Series(out_dtypes)],
        input_frame_type="pandas",
    )
    return result[data_cols]


def _assert_match(result, df_data, dropna=True, numeric_only=False):
    """Assert result matches pandas DataFrame.mode()."""
    expected = pd.DataFrame(df_data).mode(dropna=dropna, numeric_only=numeric_only)
    assert list(result.columns) == list(
        expected.columns
    ), f"columns: {list(result.columns)} != {list(expected.columns)}"
    assert len(result) == len(expected), f"rows: {len(result)} != {len(expected)}"
    for col in expected.columns:
        for i in range(len(expected)):
            rv = result[col].iloc[i]
            ev = expected[col].iloc[i]
            rv_na = pd.isna(rv)
            ev_na = pd.isna(ev)
            if rv_na and ev_na:
                continue
            assert not (rv_na or ev_na), (
                f"col={col} row={i}: one is NA, other is not "
                f"(result={rv!r}, expected={ev!r})"
            )
            try:
                assert float(rv) == float(ev), f"col={col} row={i}: {rv} != {ev}"
            except (ValueError, TypeError):
                assert str(rv) == str(ev), f"col={col} row={i}: {rv} != {ev}"


# ── basic mode ─────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "df_data, expected_a, expected_b",
    [
        pytest.param(
            {"a": [1, 2, 2, 3], "b": [4, 4, 5, 5]},
            [2],
            [4, 5],
            id="basic_ties",
        ),
        pytest.param(
            {"a": [1, 1, 1, 2], "b": [3, 3, 3, 4]},
            [1],
            [3],
            id="single_mode",
        ),
        pytest.param(
            {"a": [5, 5, 5], "b": [1, 2, 3]},
            [5],
            [1, 2, 3],
            id="all_tied_b",
        ),
    ],
)
def test_mode_basic(df_data, expected_a, expected_b):
    result = _run_mode_spe(df_data)
    assert list(result.columns) == ["a", "b"]
    modes_a = [v for v in result["a"] if not pd.isna(v)]
    modes_b = [v for v in result["b"] if not pd.isna(v)]
    assert modes_a == expected_a
    assert modes_b == expected_b
    assert len(result) == max(len(expected_a), len(expected_b))


def test_mode_mixed_dtypes():
    """Per-column processing preserves dtype (no cross-column promotion)."""
    result = _run_mode_spe({"a": [1, 2, 2, 3], "c": ["x", "x", "y", "z"]})
    _assert_match(result, {"a": [1, 2, 2, 3], "c": ["x", "x", "y", "z"]})
    assert len(result) == 1
    assert int(result["a"].iloc[0]) == 2
    assert str(result["c"].iloc[0]) == "x"


def test_mode_dropna_true():
    result = _run_mode_spe({"a": [1, None, 1, None, 2]}, dropna=True)
    assert int(result["a"].iloc[0]) == 1
    assert len(result) == 1


def test_mode_dropna_false():
    """null is a valid candidate; sorted with nulls last (pandas-compatible)."""
    result = _run_mode_spe({"a": [1, None, 1, None, 2]}, dropna=False)
    # null count=2, value 1 count=2 → both are modes; null goes last
    assert len(result) == 2
    assert int(result["a"].iloc[0]) == 1
    assert pd.isna(result["a"].iloc[1])


def test_mode_all_null_column():
    result = _run_mode_spe({"a": [1, 1, 2], "b": [None, None, None]}, dropna=True)
    assert list(result.columns) == ["a", "b"]
    assert int(result["a"].iloc[0]) == 1
    assert pd.isna(result["b"].iloc[0])


def test_mode_numeric_only():
    """numeric_only includes boolean columns (matching pandas)."""
    result = _run_mode_spe(
        {"a": [1, 2, 2, 3], "b": ["x", "x", "y", "z"]},
        numeric_only=True,
    )
    assert list(result.columns) == ["a"]
    assert int(result["a"].iloc[0]) == 2


def test_mode_numeric_only_includes_bool():
    """pandas includes boolean columns in mode(numeric_only=True)."""
    result = _run_mode_spe(
        {"b": [True, False], "i": [1, 2], "s": ["x", "y"]},
        numeric_only=True,
    )
    expected = pd.DataFrame({"b": [True, False], "i": [1, 2], "s": ["x", "y"]}).mode(
        numeric_only=True
    )
    assert set(result.columns) == set(expected.columns)
    assert "b" in result.columns
    assert "i" in result.columns
    assert "s" not in result.columns


def test_mode_empty_input():
    result = _run_mode_spe({"a": []})
    assert list(result.columns) == ["a"]
    assert len(result) == 0


def test_mode_single_value():
    result = _run_mode_spe({"a": [42, 42, 42]})
    assert len(result) == 1
    assert int(result["a"].iloc[0]) == 42


def test_mode_matches_pandas():
    """Verify mode output matches pandas DataFrame.mode()."""
    df_data = {
        "a": [1, 2, 2, 3],
        "b": [4, 4, 5, 5],
        "c": ["x", "x", "y", "z"],
    }
    result = _run_mode_spe(df_data)
    _assert_match(result, df_data)


def test_mode_dropna_false_matches_pandas():
    """dropna=False with null + non-null tied modes matches pandas."""
    df_data = {"a": [1, None, 1, None, 2]}
    result = _run_mode_spe(df_data, dropna=False)
    _assert_match(result, df_data, dropna=False)


# ── internal column-name collisions ────────────────────────────────


def test_mode_column_named_like_count_alias():
    """A user column named ``__cnt`` must not collide with the map-stage
    count alias (regression: polars ``DuplicateError`` at planning time)."""
    df_data = {"__cnt": [1, 1, 2]}
    result = _run_mode_spe(df_data)
    assert int(result["__cnt"].iloc[0]) == 1
    _assert_match(result, df_data)


# ── multi-chunk combine stage ──────────────────────────────────────


def test_mode_combine_stage_correctness():
    """Verify that merge (combine stage) produces correct global counts.

    Simulates two map chunks, then the combine stage, and asserts the
    merged frequency table yields the correct modes.  This exercises
    the tree-reduction merge path (lock-step explode + re-aggregate).
    """
    if not is_arrow_dtype_supported():
        pytest.skip("pandas ArrowDtype is not supported")

    cols = ["a", "b"]
    agg = ModeFreqAgg(columns=cols)

    # Two map chunks
    chunk1 = pl.DataFrame({"a": [1, 2, 2], "b": [4, 4, 5]})
    chunk2 = pl.DataFrame({"a": [1, 3, 3], "b": [4, 5, 6]})
    freq1 = agg.agg(chunk1)
    freq2 = agg.agg(chunk2)

    # SPE stores intermediate frames as pandas; convert and concat
    from maxframe.lib.dtypes_extension.blob import polars_to_pandas_with_blobs

    pdf1 = polars_to_pandas_with_blobs(freq1)
    pdf2 = polars_to_pandas_with_blobs(freq2)
    combined = pd.concat([pdf1, pdf2], ignore_index=True)

    # Convert back to polars for combine stage
    combined_pl = pl.from_pandas(combined)
    merged = agg.agg(combined_pl)

    # Run _mode_extract on the merged result
    merged_pd = polars_to_pandas_with_blobs(merged)
    row = merged_pd.iloc[0]
    result_rows = _mode_extract(row, all_cols=cols)

    # Combined: a=[1,2,2,1,3,3] → counts: 1→2, 2→2, 3→2 → all three are modes
    # Combined: b=[4,4,5,4,5,6] → counts: 4→3, 5→2, 6→1 → only 4 is mode
    modes_a = sorted(r["a"] for r in result_rows if r["a"] is not None)
    assert modes_a == [1, 2, 3]
    # b has 1 mode (4), a has 3 modes → result has 3 rows, b padded with None
    assert len(result_rows) == 3
    assert result_rows[0]["b"] == 4 or result_rows[1]["b"] == 4


def test_mode_combine_stage_matches_pandas():
    """Multi-chunk combine produces same result as pandas on full data."""
    if not is_arrow_dtype_supported():
        pytest.skip("pandas ArrowDtype is not supported")

    cols = ["a", "b"]
    agg = ModeFreqAgg(columns=cols)

    chunk1 = pl.DataFrame({"a": [1, 2, 2], "b": [4, 4, 5]})
    chunk2 = pl.DataFrame({"a": [1, 3, 3], "b": [4, 5, 6]})
    freq1 = agg.agg(chunk1)
    freq2 = agg.agg(chunk2)

    from maxframe.lib.dtypes_extension.blob import polars_to_pandas_with_blobs

    pdf1 = polars_to_pandas_with_blobs(freq1)
    pdf2 = polars_to_pandas_with_blobs(freq2)
    combined = pd.concat([pdf1, pdf2], ignore_index=True)
    combined_pl = pl.from_pandas(combined)
    merged = agg.agg(combined_pl)

    merged_pd = polars_to_pandas_with_blobs(merged)
    row = merged_pd.iloc[0]
    result_rows = _mode_extract(row, all_cols=cols)

    # Compare with pandas on full data
    full_data = {"a": [1, 2, 2, 1, 3, 3], "b": [4, 4, 5, 4, 5, 6]}
    expected = pd.DataFrame(full_data).mode()

    modes_a = sorted(
        r["a"] for r in result_rows if r["a"] is not None and not pd.isna(r["a"])
    )
    expected_a = sorted(expected["a"].dropna().tolist())
    assert modes_a == expected_a


def test_mode_dropna_false_sort_with_null():
    """With dropna=False, null modes are sorted last (pandas-compatible)."""
    result = _run_mode_spe({"a": ["x", None]}, dropna=False)
    # x count=1, null count=1 → both are modes; null goes last
    assert len(result) == 2
    assert str(result["a"].iloc[0]) == "x"
    assert pd.isna(result["a"].iloc[1])
