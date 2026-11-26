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
import polars as pl
import pytest

from maxframe.liteframe.datasource.from_local import from_local_df
from maxframe.liteframe.initializer import LiteFrame as LF
from maxframe.liteframe.operators.sort import LiteFrameSort


@pytest.fixture
def lf():
    return LF({"a": [3, 1, 2, 1, 5], "b": list("abcde")})


# --- operator construction ---


@pytest.mark.parametrize(
    "method,ascending",
    [
        pytest.param("nlargest", False, id="nlargest"),
        pytest.param("nsmallest", True, id="nsmallest"),
    ],
)
def test_creates_sort_op(lf, method, ascending):
    """Both wrappers build LiteFrameSort with their direction, nrows and
    defaults (keep='first', na_position='last')."""
    op = getattr(lf, method)(3, "a").op
    assert isinstance(op, LiteFrameSort)
    assert op.ascending is ascending
    assert op.nrows == 3
    assert op.keep_kind == "first"
    assert op.na_position == "last"


@pytest.mark.parametrize("method", ["nlargest", "nsmallest"])
@pytest.mark.parametrize("keep", ["last", "all"])
def test_keep_propagates(lf, method, keep):
    assert getattr(lf, method)(3, "a", keep=keep).op.keep_kind == keep


@pytest.mark.parametrize("method", ["nlargest", "nsmallest"])
def test_columns_as_list_numeric(method):
    lf = LF({"a": [3, 1, 2, 1, 5], "c": [10, 20, 30, 40, 50]})
    result = getattr(lf, method)(3, ["a", "c"])
    assert result.op.by == ["a", "c"]


@pytest.mark.parametrize("method", ["nlargest", "nsmallest"])
def test_n_zero(lf, method):
    result = getattr(lf, method)(0, "a")
    assert result.op.nrows == 0


# --- validation ---


@pytest.mark.parametrize("method", ["nlargest", "nsmallest"])
def test_invalid_keep_raises(lf, method):
    with pytest.raises(ValueError, match="Invalid keep"):
        getattr(lf, method)(3, "a", keep="invalid")


@pytest.mark.parametrize("method", ["nlargest", "nsmallest"])
def test_negative_n_normalizes_to_zero(lf, method):
    """pandas treats negative n as empty selection; LiteFrame matches."""
    result = getattr(lf, method)(-1, "a")
    assert result.op.nrows == 0


@pytest.mark.parametrize("bad_n", [3.0, "3", True])
@pytest.mark.parametrize("method", ["nlargest", "nsmallest"])
def test_non_int_n_raises(lf, method, bad_n):
    with pytest.raises(TypeError, match="n must be an int"):
        getattr(lf, method)(bad_n, "a")


@pytest.mark.parametrize("np_int", [np.int64(2), np.int8(2), np.int32(2)])
@pytest.mark.parametrize("method", ["nlargest", "nsmallest"])
def test_numpy_integer_n_accepted(lf, method, np_int):
    """NumPy integer scalars are accepted and normalized to Python int."""
    result = getattr(lf, method)(np_int, "a")
    assert result.op.nrows == 2
    assert isinstance(result.op.nrows, int)


@pytest.mark.parametrize("method", ["nlargest", "nsmallest"])
def test_nonexistent_column_raises_keyerror(lf, method):
    """Early column validation — KeyError at call time, not execution time."""
    with pytest.raises(KeyError, match="not found"):
        getattr(lf, method)(3, "nope")


@pytest.mark.parametrize("method", ["nlargest", "nsmallest"])
def test_nonexistent_column_in_list_raises_keyerror(lf, method):
    with pytest.raises(KeyError, match="not found"):
        getattr(lf, method)(3, ["a", "nope"])


@pytest.mark.parametrize("method", ["nlargest", "nsmallest"])
def test_non_numeric_column_raises_typeerror(lf, method):
    """Non-numeric columns (strings, etc.) raise TypeError at call time,
    matching pandas nlargest/nsmallest dtype restrictions."""
    with pytest.raises(TypeError, match="unsupported dtype"):
        getattr(lf, method)(3, "b")


@pytest.mark.parametrize("method", ["nlargest", "nsmallest"])
def test_duplicate_columns_raises_valueerror(method):
    """Duplicate column labels are rejected at call time."""
    lf = LF({"a": [3, 1, 2, 1, 5], "c": [10, 20, 30, 40, 50]})
    with pytest.raises(ValueError, match="Duplicate column labels"):
        getattr(lf, method)(3, ["a", "a"])


# --- shape & dtypes ---


def test_output_dtypes_match_input(lf):
    result = lf.nlargest(3, "a")
    assert list(result._data._physical_dtypes.index) == ["a", "b"]


def test_keep_all_shape_unknown(lf):
    """For keep_kind='all', __call__ publishes unknown row count (np.nan)
    because the output may exceed nrows due to boundary ties."""
    result = lf.nlargest(3, "a", keep="all")
    assert np.isnan(result.shape[0])


def test_keep_first_shape_capped(lf):
    result = lf.nlargest(3, "a", keep="first")
    assert result.shape[0] == 3


@pytest.mark.parametrize("keep", ["first", "last"])
def test_multi_col_keep_shape_unknown(keep):
    """Multi-column keep='first'/'last' publishes unknown row count because
    pandas SelectNFrame can return more or fewer than nrows on composite ties."""
    lf = LF({"a": [3, 1, 2, 1, 5], "c": [10, 20, 30, 40, 50]})
    result = lf.nlargest(3, ["a", "c"], keep=keep)
    assert np.isnan(result.shape[0])


@pytest.mark.parametrize("method", ["nlargest", "nsmallest"])
def test_polars_backed_frame_validates_numeric_keys(method):
    """A LiteFrame built from a polars frame declares polars dtypes, which carry
    the numeric check in type predicates rather than ``kind``."""
    lf = from_local_df(pl.DataFrame({"a": [1.0, 2.0], "b": ["x", "y"]}))
    assert getattr(lf, method)(1, "a").op.nrows == 1
    with pytest.raises(TypeError, match="unsupported dtype"):
        getattr(lf, method)(1, "b")
