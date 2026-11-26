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

"""Tests for LiteFrame sample() and groupby.sample()."""

import numpy as np
import pandas as pd
import pytest

from maxframe.codegen.spe.core import SPECodeContext
from maxframe.codegen.spe.liteframe.sample import LiteFrameSampleAdapter
from maxframe.liteframe.core import RANGE_COL_NAME
from maxframe.liteframe.datasource.from_local import from_local_df
from maxframe.liteframe.initializer import LiteFrame as LF
from maxframe.liteframe.operators.functional import LiteFrameFunctional
from maxframe.liteframe.operators.project import LiteFrameProjection
from maxframe.liteframe.operators.sample import LiteFrameSample
from maxframe.protocol import DefaultIndexType

_PDF = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [10, 20, 30, 40, 50]})


def _get_sample_op(result):
    """Get the LiteFrameSample operator from a result (walks past projection)."""
    op = result.data.op
    if isinstance(op, LiteFrameSample):
        return op
    for inp in getattr(op, "inputs", []):
        if isinstance(inp.op, LiteFrameSample):
            return inp.op
    return op


def _spe_sample(df, **kwargs):
    return LiteFrameSampleAdapter._run_sample(df, **kwargs)


# ── Construction tests ─────────────────────────────────────────────


@pytest.fixture
def lf():
    return LF({"a": [1, 2, 3, 4, 5], "b": [10, 20, 30, 40, 50]})


@pytest.mark.parametrize(
    "kwargs, attr_checks",
    [
        pytest.param(
            {"frac": 0.5, "random_state": 42},
            {"frac": 0.5, "n": None, "replace": False, "exact": False},
            id="frac",
        ),
        pytest.param({"n": 3, "random_state": 42}, {"n": 3, "frac": None}, id="n"),
        pytest.param(
            {"n": 3, "replace": True, "random_state": 42},
            {"replace": True},
            id="replace",
        ),
        pytest.param(
            {"n": 3, "exact": True, "random_state": 42}, {"exact": True}, id="exact"
        ),
        pytest.param({"random_state": 42}, {"n": 1}, id="default_n_is_1"),
    ],
)
def test_sample_construction(lf, kwargs, attr_checks):
    op = _get_sample_op(lf.sample(**kwargs))
    for attr, expected in attr_checks.items():
        assert getattr(op, attr) == expected, f"{attr} should be {expected}"


@pytest.mark.parametrize(
    "rs, expected_type",
    [
        pytest.param(42, np.random.RandomState, id="int_seed"),
        pytest.param(None, type(None), id="none"),
    ],
)
def test_sample_random_state(lf, rs, expected_type):
    op = _get_sample_op(lf.sample(n=2, random_state=rs))
    assert isinstance(op.random_state, expected_type)


def test_sample_weights_string(lf):
    lf_w = LF({"a": [1, 2, 3], "w": [0.5, 0.3, 0.2]})
    result = lf_w.sample(n=1, weights="w", random_state=42)
    assert not isinstance(result.data.op, LiteFrameProjection)


def test_sample_weights_liteframe(lf):
    w_lf = LF({"w": [0.5, 0.3, 0.2]})
    result = lf.sample(n=1, weights=w_lf, random_state=42)
    assert isinstance(result.data.op, LiteFrameProjection)


# ── Validation tests ────────────────────────────────────────────────


@pytest.mark.parametrize(
    "kwargs, error, match",
    [
        pytest.param({"n": 1, "frac": 0.5}, ValueError, "not both", id="n_and_frac"),
        pytest.param({"frac": -0.5}, ValueError, "non-negative", id="negative_frac"),
        pytest.param({"n": -1}, ValueError, "non-negative", id="negative_n"),
        pytest.param(
            {"frac": 1.5}, ValueError, "frac.*>.*1", id="frac_gt_1_no_replace"
        ),
        pytest.param(
            {"n": 10, "replace": False}, ValueError, "larger sample", id="n_gt_N"
        ),
        pytest.param(
            {"n": 1, "random_state": "bad"}, TypeError, "random_state", id="bad_rs"
        ),
        pytest.param(
            {"n": 1, "weights": "nonexistent"}, KeyError, None, id="bad_weights"
        ),
        pytest.param(
            {"n": 1, "weights": [1.0] * 5},
            TypeError,
            "weights must be",
            id="array_like_weights",
        ),
    ],
)
def test_sample_validation(lf, kwargs, error, match):
    with pytest.raises(error, match=match):
        lf.sample(**kwargs)


@pytest.mark.parametrize(
    "kwargs, attr",
    [
        pytest.param(
            {"frac": 1.5, "replace": True, "random_state": 42},
            "replace",
            id="frac_gt_1_replace",
        ),
        pytest.param(
            {"n": 10, "replace": True, "random_state": 42},
            "replace",
            id="n_gt_N_replace",
        ),
    ],
)
def test_sample_replace_ok(lf, kwargs, attr):
    op = _get_sample_op(lf.sample(**kwargs))
    assert getattr(op, attr) is True


# ── SPE adapter tests ───────────────────────────────────────────────


@pytest.mark.parametrize(
    "kwargs, expected_len",
    [
        pytest.param({"frac": 0.5, "replace": False, "seed": 42}, 2, id="frac"),
        pytest.param({"replace": False, "seed": 42, "chunk_n": 3}, 3, id="n"),
        pytest.param({"replace": True, "seed": 42, "chunk_n": 10}, 10, id="replace"),
        pytest.param({"chunk_n": 0, "replace": False, "seed": 42}, 0, id="n_zero"),
        pytest.param({"n": 3, "replace": False, "seed": 42}, 3, id="spe_n"),
    ],
)
def test_spe_sample(kwargs, expected_len):
    result = _spe_sample(_PDF, **kwargs)
    assert len(result) == expected_len
    assert list(result.columns) == ["a", "b"]


@pytest.mark.parametrize(
    "kwargs, check",
    [
        pytest.param(
            {"replace": False, "seed": 42, "chunk_n": 3},
            lambda r1, r2: pd.testing.assert_frame_equal(r1, r2),
            id="reproducible",
        ),
        pytest.param(
            {"replace": True, "seed": 42, "chunk_n": 10},
            lambda r1, r2: None,
            id="replace_has_dups",
        ),
    ],
)
def test_spe_reproducibility_and_replace(kwargs, check):
    r1 = _spe_sample(_PDF, **kwargs)
    r2 = _spe_sample(_PDF, **kwargs)
    check(r1, r2)
    if "replace" in kwargs and kwargs["replace"]:
        assert r1.index.duplicated().any()


def test_spe_weights():
    df_w = _PDF.copy()
    df_w["w"] = [0.0, 0.0, 1.0, 0.0, 0.0]
    result = _spe_sample(df_w, replace=False, seed=42, chunk_n=1, weights_col="w")
    assert result["a"].iloc[0] == 3


@pytest.mark.parametrize(
    "row_ids, expected_vals",
    [
        pytest.param([0, 2, 4], [1, 3, 5], id="sorted_ids"),
        pytest.param([], [], id="empty_ids"),
    ],
)
def test_spe_selected_row_ids(row_ids, expected_vals):
    result = _spe_sample(_PDF, selected_row_ids=row_ids)
    assert len(result) == len(expected_vals)
    if expected_vals:
        assert list(result["a"]) == expected_vals


def test_spe_generate_code():
    """generate_code produces compilable code for the sample SPE adapter."""
    data = from_local_df(pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}))
    op = LiteFrameSample(n=2, replace=False, random_state=42)
    out = op(data)
    code = LiteFrameSampleAdapter().generate_code(out.op, SPECodeContext())
    assert len(code) == 1
    assert "LiteFrameSampleAdapter._run_sample" in code[0]


# ── GroupBy sample tests ───────────────────────────────────────────


@pytest.fixture
def gb_lf():
    return LF({"k": ["a", "a", "b", "b"], "v": [1, 2, 3, 4]})


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"n": 1, "random_state": 42}, id="n"),
        pytest.param({"frac": 0.5, "random_state": 42}, id="frac"),
        pytest.param({"n": 3, "replace": True, "random_state": 42}, id="replace"),
        pytest.param({"n": 1, "exact": True, "random_state": 42}, id="exact_accepted"),
    ],
)
def test_groupby_sample_construction(gb_lf, kwargs):
    op = gb_lf.groupby("k").sample(**kwargs).data.op
    assert isinstance(op, LiteFrameFunctional)
    assert op.mode == "apply_chunk"
    assert op.groupby_params is not None


def test_groupby_sample_validation(gb_lf):
    with pytest.raises(ValueError, match="not both"):
        gb_lf.groupby("k").sample(n=1, frac=0.5)


# ── Range columns materialization ──────────────────────────────────


def test_sample_range_columns_materialized():
    """Sampling changes the row set, so range columns must be materialized
    into physical Int64 columns — same as LiteFrameFilter."""
    lf = from_local_df(
        pd.DataFrame({"a": range(20), "b": range(20)}),
        chunk_size=5,
        default_index_type=DefaultIndexType.range,
    )
    assert RANGE_COL_NAME in lf.frame_metadata.range_columns
    assert RANGE_COL_NAME not in lf._data._physical_dtypes.index
    assert lf.shape[1] == 3

    result = lf.sample(frac=0.5, random_state=42)
    assert RANGE_COL_NAME in result._data._physical_dtypes.index
    assert not result.frame_metadata.range_columns
    assert result.shape[1] == 3
    assert len(result.dtypes) == 3
    assert list(result.dtypes.index) == ["a", "b", RANGE_COL_NAME]
