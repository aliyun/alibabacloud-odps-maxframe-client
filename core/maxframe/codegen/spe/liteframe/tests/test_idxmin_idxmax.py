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

"""SPE adapter tests for idxmin / idxmax."""

import pandas as pd
import pyarrow as pa
import pytest

from maxframe.codegen.spe.liteframe.reduction import LiteFrameAggAdapter
from maxframe.liteframe.reduction.compile import _IDX_SUFFIX, _VALUE_SUFFIX
from maxframe.liteframe.reduction.custom_agg import IdxMaxAgg, IdxMinAgg
from maxframe.utils import wrap_arrow_dtype

_INT64 = wrap_arrow_dtype(pa.int64())
_FLOAT64 = wrap_arrow_dtype(pa.float64())


def _idx_agg_desc(custom_func, cols, value_dtypes=None, key="agg_0"):
    """Build a custom-agg descriptor for idxmin/idxmax.

    The ``agg`` method returns ``{col}__value`` (double[pyarrow], NA-safe)
    and ``{col}__idx`` (int64[pyarrow], nullable) columns, so
    ``agg_output_dtypes`` must describe those intermediate columns.  The
    ``post`` method (compiled as a separate projection) then extracts the
    ``__idx`` columns as the final result.
    """
    if value_dtypes is None:
        value_dtypes = pd.Series({c: _FLOAT64 for c in cols})
    agg_cols = []
    for c in cols:
        agg_cols.extend([f"{c}{_VALUE_SUFFIX}", f"{c}{_IDX_SUFFIX}"])
    agg_output_dtypes = pd.Series()
    for c in cols:
        agg_output_dtypes[f"{c}{_VALUE_SUFFIX}"] = value_dtypes[c]
        agg_output_dtypes[f"{c}{_IDX_SUFFIX}"] = _INT64
    return {
        "raw_func_name": getattr(custom_func, "name", "<custom>"),
        "map_func_name": "custom_agg",
        "agg_func_name": "custom_agg",
        "cols": cols,
        "agg_output_cols": agg_cols,
        "output_key": key,
        "kwds": {},
        "custom_func": custom_func,
        "agg_output_dtypes": agg_output_dtypes,
        "input_frame_type": "pandas",
        "needs_row_pos": True,
    }


def _finalize(result, agg, cols, by=None):
    """Apply ``post`` to the agg-stage result to get final idxmin/idxmax output.

    For groupby results, group key columns (``by``) are preserved alongside
    the extracted index columns, mirroring the compiled post projection.
    """
    final = agg.post(result)
    if by is not None:
        by_cols = [by] if isinstance(by, str) else list(by)
        for c in by_cols:
            if c in result.columns:
                final[c] = result[c].values
        # Reorder: group keys first, then value columns
        final = final[by_cols + [c for c in final.columns if c not in by_cols]]
    return final


# ── whole-frame idxmin/idxmax via SPE adapter ──────────────────────


@pytest.mark.parametrize(
    "method, df_data, cols, expected",
    [
        pytest.param(
            "idxmin",
            {"a": [3, 1, 2], "b": [6, 5, 4]},
            ["a", "b"],
            {"a": 1, "b": 2},
            id="idxmin_basic",
        ),
        pytest.param(
            "idxmax",
            {"a": [3, 1, 2], "b": [6, 5, 4]},
            ["a", "b"],
            {"a": 0, "b": 0},
            id="idxmax_basic",
        ),
        pytest.param(
            "idxmin",
            {"a": [5, 3, 4, 1, 2]},
            ["a"],
            {"a": 3},
            id="idxmin_single_col",
        ),
        pytest.param(
            "idxmax",
            {"a": [5, 3, 4, 1, 2]},
            ["a"],
            {"a": 0},
            id="idxmax_single_col",
        ),
    ],
)
def test_spe_idxmin_idxmax_whole_frame(method, df_data, cols, expected):
    """SPE adapter runs idxmin/idxmax over a whole frame."""
    df = pd.DataFrame(df_data)
    agg = IdxMinAgg() if method == "idxmin" else IdxMaxAgg()
    desc = _idx_agg_desc(agg, cols)
    result = LiteFrameAggAdapter._run_agg(
        df, [desc], None, list(desc["agg_output_dtypes"].index)
    )
    final = _finalize(result, agg, cols)
    assert len(final) == 1
    for col, val in expected.items():
        assert final[col].iloc[0] == val


# ── groupby idxmin/idxmax via SPE adapter ──────────────────────────


@pytest.mark.parametrize(
    "method, df_data, by, cols, expected",
    [
        pytest.param(
            "idxmin",
            {"k": ["x", "x", "y", "y"], "a": [3, 1, 5, 2]},
            "k",
            ["a"],
            {"x": 1, "y": 3},
            id="idxmin_grouped",
        ),
        pytest.param(
            "idxmax",
            {"k": ["x", "x", "y", "y"], "a": [3, 1, 5, 2]},
            "k",
            ["a"],
            {"x": 0, "y": 2},
            id="idxmax_grouped",
        ),
        pytest.param(
            "idxmin",
            {"k": ["a", "a", "b", "b", "b"], "v": [5, 3, 8, 1, 6]},
            "k",
            ["v"],
            {"a": 1, "b": 3},
            id="idxmin_grouped_uneven",
        ),
        pytest.param(
            "idxmax",
            {"k": ["a", "a", "b", "b", "b"], "v": [5, 3, 8, 1, 6]},
            "k",
            ["v"],
            {"a": 0, "b": 2},
            id="idxmax_grouped_uneven",
        ),
    ],
)
def test_spe_idxmin_idxmax_groupby(method, df_data, by, cols, expected):
    """SPE adapter runs idxmin/idxmax per group, returning global row positions."""
    df = pd.DataFrame(df_data)
    agg = IdxMinAgg() if method == "idxmin" else IdxMaxAgg()
    desc = _idx_agg_desc(agg, cols)
    groupby_params = {"by": by, "sort": True, "dropna": True}
    result = LiteFrameAggAdapter._run_agg(df, [desc], groupby_params, None)
    final = _finalize(result, agg, cols, by=by)
    final = final.sort_values(by).reset_index(drop=True)
    for row in final[[by] + cols].to_dict("records"):
        assert row[by] in expected
        assert row[cols[0]] == expected[row[by]]


# ── edge cases: NA handling, all-NA, tie-breaking ─────────────────


@pytest.mark.parametrize(
    "method, df_data, value_dtype, skipna, expected",
    [
        # skipna=True: NA values are skipped.
        pytest.param(
            "idxmin",
            pd.array([3, None, 2], dtype="Int64"),
            "Int64",
            True,
            2,
            id="idxmin_skipna_true",
        ),
        pytest.param(
            "idxmax",
            pd.array([3, None, 2], dtype="Int64"),
            "Int64",
            True,
            0,
            id="idxmax_skipna_true",
        ),
        # skipna=False: if any NA, result is NA.
        pytest.param(
            "idxmin",
            pd.array([3, None, 2], dtype="Int64"),
            "Int64",
            False,
            None,
            id="idxmin_skipna_false",
        ),
        pytest.param(
            "idxmax",
            pd.array([3, None, 2], dtype="Int64"),
            "Int64",
            False,
            None,
            id="idxmax_skipna_false",
        ),
        # All-NA column returns NA.
        pytest.param(
            "idxmin",
            pd.array([None, None, None], dtype="Int64"),
            "Int64",
            None,
            None,
            id="idxmin_all_na",
        ),
        pytest.param(
            "idxmax",
            pd.array([None, None, None], dtype="Int64"),
            "Int64",
            None,
            None,
            id="idxmax_all_na",
        ),
        # Ties are broken by first occurrence.
        pytest.param(
            "idxmin",
            [5, 5, 5],
            None,
            None,
            0,
            id="idxmin_tie",
        ),
        pytest.param(
            "idxmax",
            [5, 5, 5],
            None,
            None,
            0,
            id="idxmax_tie",
        ),
    ],
)
def test_spe_idxmin_idxmax_edge_cases(method, df_data, value_dtype, skipna, expected):
    """skipna, all-NA, and tie-breaking edge cases for idxmin/idxmax."""
    df = pd.DataFrame({"a": df_data})
    if skipna is not None:
        agg = (
            IdxMinAgg(skipna=skipna) if method == "idxmin" else IdxMaxAgg(skipna=skipna)
        )
    else:
        agg = IdxMinAgg() if method == "idxmin" else IdxMaxAgg()
    kwargs = {}
    if value_dtype is not None:
        kwargs["value_dtypes"] = pd.Series({"a": value_dtype})
    desc = _idx_agg_desc(agg, ["a"], **kwargs)
    result = LiteFrameAggAdapter._run_agg(
        df, [desc], None, list(desc["agg_output_dtypes"].index)
    )
    final = _finalize(result, agg, ["a"])
    if expected is None:
        assert pd.isna(final["a"].iloc[0])
    else:
        assert final["a"].iloc[0] == expected
