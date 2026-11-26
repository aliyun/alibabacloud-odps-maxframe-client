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

"""Unit tests for idxmin / idxmax on LiteFrame and LiteFrameGroupBy."""

import pandas as pd
import pyarrow as pa
import pytest

from maxframe.liteframe.core import LiteFrame
from maxframe.liteframe.initializer import LiteFrame as LF
from maxframe.liteframe.operators.reduction import LiteFrameAgg
from maxframe.liteframe.reduction.custom_agg import IdxMaxAgg, IdxMinAgg
from maxframe.utils import wrap_arrow_dtype

_IDX_AGGS = [IdxMinAgg, IdxMaxAgg]
_IDX_IDS = ["idxmin", "idxmax"]
_IDX_METHODS = list(zip(_IDX_AGGS, _IDX_IDS))


def _get_agg_op(result):
    """Get the LiteFrameAgg operator from a result."""
    op = result.data.op
    if op.__class__.__name__ == "LiteFrameProjection":
        return result.data.inputs[0].op
    return op


# ── compile / op-creation ──────────────────────────────────────────


@pytest.mark.parametrize("agg_cls, method", _IDX_METHODS, ids=_IDX_IDS)
def test_idxmin_idxmax_creates_op_and_columns(agg_cls, method):
    """lf.idxmin()/idxmax() creates LiteFrameAgg with custom_func and preserves columns."""
    lf = LF({"a": [3, 1, 2], "b": [6, 5, 4]})
    result = getattr(lf, method)()
    assert isinstance(result, LiteFrame)
    op = _get_agg_op(result)
    assert isinstance(op, LiteFrameAgg)
    desc = op.agg_funcs[0]
    assert desc.get("custom_func") is not None
    assert desc.get("needs_row_pos") is True
    assert list(result.columns) == ["a", "b"]


@pytest.mark.parametrize("agg_cls, method", _IDX_METHODS, ids=_IDX_IDS)
def test_groupby_idxmin_idxmax_creates_op_and_columns(agg_cls, method):
    """groupby.idxmin()/idxmax() creates LiteFrameAgg with groupby_params and correct columns."""
    lf = LF({"k": ["x", "x", "y"], "a": [3, 1, 2]})
    gb = lf.groupby("k")
    result = getattr(gb, method)()
    op = _get_agg_op(result)
    assert isinstance(op, LiteFrameAgg)
    assert op.groupby_params is not None
    desc = op.agg_funcs[0]
    assert desc.get("custom_func") is not None
    assert desc.get("needs_row_pos") is True
    assert "k" in list(result.columns)
    assert "a" in list(result.columns)


@pytest.mark.parametrize("agg_cls, method", _IDX_METHODS, ids=_IDX_IDS)
@pytest.mark.parametrize(
    "skipna_kw, expected",
    [({}, True), ({"skipna": False}, False)],
    ids=["default", "false"],
)
def test_idxmin_idxmax_skipna(agg_cls, method, skipna_kw, expected):
    """skipna parameter is passed through to the CustomAgg."""
    lf = LF({"a": [3, 1, 2]})
    result = getattr(lf, method)(**skipna_kw)
    op = _get_agg_op(result)
    desc = op.agg_funcs[0]
    assert desc["custom_func"].skipna is expected


# ── CustomAgg.agg / post ────────────────────────────────────────────


@pytest.mark.parametrize(
    "agg_cls, data, iloc_offset, expected",
    [
        (IdxMinAgg, {"a": [3, 1, 2], "b": [6, 5, 4]}, 0, {"a": (1, 1), "b": (4, 2)}),
        (IdxMaxAgg, {"a": [3, 1, 2], "b": [6, 5, 4]}, 0, {"a": (3, 0), "b": (6, 0)}),
        (IdxMinAgg, {"a": [5, 3, 4]}, 100, {"a": (3, 101)}),
        (IdxMaxAgg, {"a": [5, 3, 4]}, 100, {"a": (5, 100)}),
        (
            IdxMinAgg,
            {"a": [5, 3, 4], "__mf_agg_row_pos__": [10, 11, 12]},
            0,
            {"a": (3, 11)},
        ),
        (
            IdxMaxAgg,
            {"a": [5, 3, 4], "__mf_agg_row_pos__": [10, 11, 12]},
            0,
            {"a": (5, 10)},
        ),
        (IdxMinAgg, {"a__value": [1, 3], "a__idx": [5, 2]}, 0, {"a": (1, 5)}),
        (IdxMaxAgg, {"a__value": [1, 3], "a__idx": [5, 2]}, 0, {"a": (3, 2)}),
    ],
    ids=[
        "idxmin-basic",
        "idxmax-basic",
        "idxmin-offset",
        "idxmax-offset",
        "idxmin-rowpos",
        "idxmax-rowpos",
        "idxmin-combine",
        "idxmax-combine",
    ],
)
def test_idx_agg(agg_cls, data, iloc_offset, expected):
    """agg produces (value, idx) pairs at map, iloc-offset, row-pos, and combine stages."""
    agg = agg_cls()
    df = pd.DataFrame(data)
    df.attrs["iloc_offset"] = iloc_offset
    result = agg.agg(df)
    for col, (val, idx) in expected.items():
        assert result[f"{col}__value"].iloc[0] == val
        assert result[f"{col}__idx"].iloc[0] == idx


@pytest.mark.parametrize(
    "agg_cls, data, all_na",
    [
        (IdxMinAgg, {"a": pd.array([None, None, None], dtype="Int64")}, True),
        (IdxMaxAgg, {"a": pd.array([None, None, None], dtype="Int64")}, True),
        (IdxMinAgg, {"a": [1, 1, 1]}, False),
        (IdxMaxAgg, {"a": [5, 5, 5]}, False),
    ],
    ids=["idxmin-all-na", "idxmax-all-na", "idxmin-tie", "idxmax-tie"],
)
def test_idx_agg_na_and_tie(agg_cls, data, all_na):
    """agg returns NA for all-NA columns; first occurrence on ties."""
    agg = agg_cls()
    df = pd.DataFrame(data)
    df.attrs["iloc_offset"] = 0
    result = agg.agg(df)
    if all_na:
        assert pd.isna(result["a__value"].iloc[0])
        assert pd.isna(result["a__idx"].iloc[0])
    else:
        assert result["a__idx"].iloc[0] == 0


@pytest.mark.parametrize("agg_cls", _IDX_AGGS, ids=_IDX_IDS)
def test_idx_post_extracts_idx_columns(agg_cls):
    """post extracts __idx columns as the final result."""
    agg = agg_cls()
    df = pd.DataFrame({"a__value": [1], "a__idx": [3], "b__value": [5], "b__idx": [7]})
    result = agg.post(df)
    assert list(result.columns) == ["a", "b"]
    assert result["a"].iloc[0] == 3
    assert result["b"].iloc[0] == 7


@pytest.mark.parametrize(
    "agg_cls, skipna, expected_val, expected_idx",
    [
        (IdxMinAgg, False, pd.NA, pd.NA),
        (IdxMaxAgg, False, pd.NA, pd.NA),
        (IdxMinAgg, True, 2, 2),
        (IdxMaxAgg, True, 3, 0),
    ],
    ids=[
        "idxmin-skipna-false",
        "idxmax-skipna-false",
        "idxmin-skipna-true",
        "idxmax-skipna-true",
    ],
)
def test_idx_agg_skipna_with_na(agg_cls, skipna, expected_val, expected_idx):
    """skipna=False → NA result; skipna=True → NA values skipped."""
    agg = agg_cls(skipna=skipna)
    df = pd.DataFrame({"a": pd.array([3, None, 2], dtype="Int64")})
    df.attrs["iloc_offset"] = 0
    result = agg.agg(df)
    if pd.isna(expected_val):
        assert pd.isna(result["a__value"].iloc[0])
        assert pd.isna(result["a__idx"].iloc[0])
    else:
        assert result["a__value"].iloc[0] == expected_val
        assert result["a__idx"].iloc[0] == expected_idx


@pytest.mark.parametrize("agg_cls", _IDX_AGGS, ids=_IDX_IDS)
def test_idx_agg_empty_chunk_neutral_and_post_passthrough(agg_cls):
    """Empty chunk → 0-row neutral element; all-empty combine → 0-row post()."""
    agg = agg_cls(skipna=False)
    empty = pd.DataFrame({"a": pd.array([], dtype="int64")})
    assert len(agg.agg(empty)) == 0  # not (NA, NA)
    post_in = pd.DataFrame({"a__idx": pd.array([], dtype=wrap_arrow_dtype(pa.int64()))})
    assert len(agg.post(post_in)) == 0  # transform_chunk preserves row count


@pytest.mark.parametrize("agg_cls", _IDX_AGGS, ids=_IDX_IDS)
def test_idx_agg_fake_arrow_int64_precision(agg_cls):
    """FakeArrow int64 near 2**53 must not narrow to float64 (wrong winner)."""
    from maxframe.lib.dtypes_extension._fake_arrow_dtype import FakeArrowDtype

    big = 2**53
    data = [big + 1, big] if agg_cls is IdxMinAgg else [big, big + 1]
    df = pd.DataFrame({"a": pd.Series(data, dtype=FakeArrowDtype(pa.int64()))})
    assert agg_cls().post(agg_cls().agg(df))["a"].iloc[0] == 1


@pytest.mark.parametrize("agg_cls", _IDX_AGGS, ids=_IDX_IDS)
def test_idx_agg_empty_chunk_preserves_value_dtype(agg_cls):
    """Empty chunk __value column must match the source dtype, not int64."""
    agg = agg_cls(skipna=False)
    for dtype in ("bool", "float64"):
        empty = pd.DataFrame({"a": pd.array([], dtype=dtype)})
        result = agg.agg(empty)
        assert result["a__value"].dtype == dtype
