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

"""Compile-time tests for LiteFrame.mode()."""

import pandas as pd
import polars as pl
import pyarrow as pa
import pytest

from maxframe.liteframe.initializer import LiteFrame as LF
from maxframe.liteframe.operators.reduction import LiteFrameAgg
from maxframe.liteframe.reduction.custom_agg import CustomAgg
from maxframe.liteframe.reduction.mode import (
    ModeFreqAgg,
    _cnt_col,
    _is_mode_numeric_dtype,
    _mode_extract,
    _val_col,
)
from maxframe.utils.datatypes import is_arrow_dtype_supported


def _walk(node):
    yield node
    for inp in getattr(node.op, "inputs", []):
        yield from _walk(inp)


def _op_types(lf):
    return {type(node.op).__name__ for node in _walk(lf.data)}


# ── operator graph ─────────────────────────────────────────────────


def test_mode_operator_graph():
    lf = LF({"a": [1, 2, 2, 3], "b": [4, 4, 5, 5]})
    result = lf.mode()
    ops = _op_types(result)
    assert "LiteFrameAgg" in ops
    assert "LiteFrameFunctional" in ops


def test_mode_agg_takes_isolation_path():
    """The configured ``ModeFreqAgg`` instance is classified as custom code.

    The agg descriptor carries an *instance* of the
    ``@builtin_function``-decorated class (it must, to carry
    ``dropna``/``columns``), which is pickled into the op, so
    ``has_custom_code()`` is ``True`` and the agg runs in isolation —
    same classification as ``IdxMinAgg``/``IdxMaxAgg``.  Only the
    ``BuiltinFunction`` *wrapper* shape skips isolation; see
    ``docs/designs/liteframe/operators/agg.md``.
    """
    lf = LF({"a": [1, 2, 2, 3]})
    agg_ops = [n.op for n in _walk(lf.mode().data) if isinstance(n.op, LiteFrameAgg)]
    assert len(agg_ops) == 1
    assert type(agg_ops[0].agg_funcs[0]["custom_func"]).__name__ == "ModeFreqAgg"
    assert agg_ops[0].has_custom_code() is True


def test_mode_columns_preserved():
    lf = LF({"a": [1, 2, 2, 3], "b": [4, 4, 5, 5]})
    result = lf.mode()
    assert list(result.columns) == ["a", "b"]


def test_mode_shape_unknown_rows():
    lf = LF({"a": [1, 2, 2, 3]})
    result = lf.mode()
    assert pd.isna(result.shape[0])
    assert result.shape[1] == 1


# ── ModeFreqAgg stage detection ────────────────────────────────────


def test_mode_freq_agg_map_stage():
    """agg() on original columns produces packed per-column list columns."""
    df = pl.DataFrame({"a": [1, 2, 2, 3], "b": [4, 4, 5, 5]})
    agg = ModeFreqAgg(dropna=True, columns=["a", "b"])
    result = agg.agg(df)
    assert set(result.columns) == {
        _val_col("a"),
        _cnt_col("a"),
        _val_col("b"),
        _cnt_col("b"),
    }
    # Single row (packed)
    assert len(result) == 1
    a_vals = result[_val_col("a")].to_list()[0]
    a_cnts = result[_cnt_col("a")].to_list()[0]
    assert sorted(a_vals) == [1, 2, 3]
    # value 2 has count 2; 1 and 3 have count 1 each
    val_to_cnt = dict(zip(a_vals, a_cnts))
    assert val_to_cnt[2] == 2
    assert val_to_cnt[1] == 1
    assert val_to_cnt[3] == 1


def test_mode_freq_agg_combine_stage():
    """agg() on packed list columns merges them (sums counts)."""
    freq = pl.DataFrame(
        {
            _val_col("a"): [[1, 2]],
            _cnt_col("a"): [[1, 2]],
            _val_col("b"): [[4]],
            _cnt_col("b"): [[2]],
        }
    )
    agg = ModeFreqAgg(columns=["a", "b"])
    result = agg.agg(freq)
    assert len(result) == 1
    a_vals = result[_val_col("a")].to_list()[0]
    a_cnts = result[_cnt_col("a")].to_list()[0]
    assert sorted(a_vals) == [1, 2]
    assert dict(zip(a_vals, a_cnts)) == {1: 1, 2: 2}


def test_mode_freq_agg_combine_merges_duplicates():
    """agg() on a packed freq table with duplicate values sums the counts."""
    freq = pl.DataFrame(
        {
            _val_col("a"): [[1, 1, 2, 2]],
            _cnt_col("a"): [[1, 2, 1, 1]],
        }
    )
    agg = ModeFreqAgg(columns=["a"])
    result = agg.agg(freq)
    assert len(result) == 1
    a_vals = result[_val_col("a")].to_list()[0]
    a_cnts = result[_cnt_col("a")].to_list()[0]
    val_to_cnt = dict(zip(a_vals, a_cnts))
    assert val_to_cnt[1] == 3
    assert val_to_cnt[2] == 2


def test_mode_freq_agg_combine_empty_lists():
    """Combining empty packed rows preserves empty (no synthetic null/0)."""
    freq = pl.DataFrame(
        {
            _val_col("a"): [[], []],
            _cnt_col("a"): [[], []],
        },
        schema={
            _val_col("a"): pl.List(pl.Int64),
            _cnt_col("a"): pl.List(pl.Int64),
        },
    )
    agg = ModeFreqAgg(columns=["a"])
    result = agg.agg(freq)
    assert len(result) == 1
    a_vals = result[_val_col("a")].to_list()[0]
    a_cnts = result[_cnt_col("a")].to_list()[0]
    assert a_vals == []
    assert a_cnts == []


def test_mode_freq_agg_dropna_true():
    """dropna=True filters out null values in per-column counting."""
    df = pl.DataFrame({"a": [1, None, 1, None]})
    agg = ModeFreqAgg(dropna=True, columns=["a"])
    result = agg.agg(df)
    a_vals = result[_val_col("a")].to_list()[0]
    a_cnts = result[_cnt_col("a")].to_list()[0]
    assert a_vals == [1]
    assert a_cnts == [2]


def test_mode_freq_agg_dropna_false():
    """dropna=False keeps null as a distinct value."""
    df = pl.DataFrame({"a": [1, None, 1, None]})
    agg = ModeFreqAgg(dropna=False, columns=["a"])
    result = agg.agg(df)
    a_vals = result[_val_col("a")].to_list()[0]
    a_cnts = result[_cnt_col("a")].to_list()[0]
    # null has count 2, 1 has count 2
    assert len(a_vals) == 2
    assert sum(a_cnts) == 4


def test_mode_freq_agg_is_custom_agg():
    assert issubclass(ModeFreqAgg, CustomAgg)


def test_mode_freq_agg_name():
    agg = ModeFreqAgg()
    assert agg.name == "mode_freq"


# ── _mode_extract UDF ──────────────────────────────────────────────


def test_mode_extract_basic():
    """_mode_extract finds max-count values, sorts, and coalesces."""
    import pandas as pd

    row = pd.Series(
        {
            _val_col("a"): [1, 2, 3],
            _cnt_col("a"): [1, 2, 1],
            _val_col("b"): [4, 5],
            _cnt_col("b"): [2, 2],
        }
    )
    result = _mode_extract(row, all_cols=["a", "b"])
    assert len(result) == 2
    assert result[0]["a"] == 2
    assert result[0]["b"] == 4
    assert result[1]["b"] == 5
    assert result[1]["a"] is None


def test_mode_extract_single_mode():
    row = pd.Series(
        {
            _val_col("a"): [1, 2],
            _cnt_col("a"): [3, 1],
        }
    )
    result = _mode_extract(row, all_cols=["a"])
    assert len(result) == 1
    assert result[0]["a"] == 1


def test_mode_extract_missing_column():
    """Empty freq for a column produces None padding."""
    row = pd.Series(
        {
            _val_col("a"): [1],
            _cnt_col("a"): [2],
            _val_col("b"): [],
            _cnt_col("b"): [],
        }
    )
    result = _mode_extract(row, all_cols=["a", "b"])
    assert len(result) == 1
    assert result[0]["a"] == 1
    assert result[0]["b"] is None


def test_mode_extract_empty_input():
    """All empty freq → returns None (zero rows)."""
    row = pd.Series(
        {
            _val_col("a"): [],
            _cnt_col("a"): [],
            _val_col("b"): [],
            _cnt_col("b"): [],
        }
    )
    result = _mode_extract(row, all_cols=["a", "b"])
    assert result is None


def test_mode_extract_sorts_values():
    """Tied modes are sorted in ascending order."""
    row = pd.Series(
        {
            _val_col("a"): [3, 1, 2],
            _cnt_col("a"): [2, 2, 1],
        }
    )
    result = _mode_extract(row, all_cols=["a"])
    assert [r["a"] for r in result] == [1, 3]


def test_mode_extract_null_last():
    """Null modes are sorted after non-null (pandas-compatible)."""
    row = pd.Series(
        {
            _val_col("a"): ["x", None],
            _cnt_col("a"): [1, 1],
        }
    )
    result = _mode_extract(row, all_cols=["a"])
    assert len(result) == 2
    assert result[0]["a"] == "x"
    assert result[1]["a"] is None


def test_mode_extract_unsortable_struct():
    """Tied struct modes cannot be sorted — fall back to unsorted with
    a warning, matching pandas' 'Unable to sort modes' behavior."""
    row = pd.Series(
        {
            _val_col("a"): [{"x": 1}, {"y": 2}],
            _cnt_col("a"): [1, 1],
        }
    )
    with pytest.warns(UserWarning, match="Unable to sort modes"):
        result = _mode_extract(row, all_cols=["a"])
    assert len(result) == 2
    # Both modes present, order unspecified
    modes = [r["a"] for r in result]
    assert len(modes) == 2
    assert {"x": 1} in modes
    assert {"y": 2} in modes


# ── numeric_only ───────────────────────────────────────────────────


@pytest.mark.parametrize(
    "numeric_only, expected_cols",
    [
        pytest.param(True, ["a"], id="numeric_only"),
        pytest.param(False, ["a", "b"], id="all"),
    ],
)
def test_mode_numeric_only(numeric_only, expected_cols):
    lf = LF({"a": [1, 2, 2, 3], "b": ["x", "x", "y", "z"]})
    result = lf.mode(numeric_only=numeric_only)
    assert list(result.columns) == expected_cols


def test_mode_numeric_only_includes_bool():
    """numeric_only includes boolean columns (matching pandas)."""
    lf = LF({"b": [True, False], "i": [1, 2], "s": ["x", "y"]})
    result = lf.mode(numeric_only=True)
    assert "b" in result.columns
    assert "i" in result.columns
    assert "s" not in result.columns


def test_is_mode_numeric_dtype_includes_bool():
    """_is_mode_numeric_dtype includes booleans."""
    if not is_arrow_dtype_supported():
        pytest.skip("pandas ArrowDtype is not supported")
    assert _is_mode_numeric_dtype(pd.ArrowDtype(pa.bool_()))


# ── dropna parameter ───────────────────────────────────────────────


@pytest.mark.parametrize("dropna", [True, False])
def test_mode_dropna(dropna):
    lf = LF({"a": [1, None, 1, None]})
    result = lf.mode(dropna=dropna)
    assert list(result.columns) == ["a"]
