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

import pandas as pd
import pyarrow as pa
import pytest

from maxframe.liteframe import LiteFrame
from maxframe.liteframe.operators.functional import LiteFrameFunctional
from maxframe.utils import wrap_arrow_dtype

_LIST = wrap_arrow_dtype(pa.list_(pa.int64()))


def _list_df():
    raw = pd.DataFrame({"id": ["x", "y"]})
    raw["lst"] = pd.Series([[1, 2], [3]]).astype(_LIST)
    return LiteFrame(raw)


def test_explode_op_structure():
    """explode builds a flat_apply op with value_type dtype, no custom code."""
    result = _list_df().explode("lst")

    op = result.data.op
    assert isinstance(op, LiteFrameFunctional)
    assert op.mode == "flat_apply"
    assert op.has_custom_code() is False
    assert list(result.columns) == ["id", "lst"]
    assert result.dtypes["lst"] == wrap_arrow_dtype(pa.int64())


@pytest.mark.parametrize("column", ["lst", ["lst"]])
def test_explode_accepts_str_and_list(column):
    assert _list_df().explode(column).dtypes["lst"] == wrap_arrow_dtype(pa.int64())


def test_explode_rejects_non_list_column():
    with pytest.raises(ValueError, match="list dtype"):
        LiteFrame({"a": [1, 2, 3]}).explode("a")


def test_explode_multi_column_single_flat_apply():
    """Multiple columns explode in one flat_apply op (paired, value_type dtypes)."""
    raw = pd.DataFrame({"a": [[1, 2], [3]], "b": [[4, 5], [6]]})
    raw["a"] = raw["a"].astype(_LIST)
    raw["b"] = raw["b"].astype(_LIST)
    result = LiteFrame(raw).explode(["a", "b"])

    op = result.data.op
    assert isinstance(op, LiteFrameFunctional)
    assert op.mode == "flat_apply"
    assert op.has_custom_code() is False
    assert list(result.columns) == ["a", "b"]
    assert result.dtypes["a"] == wrap_arrow_dtype(pa.int64())
    assert result.dtypes["b"] == wrap_arrow_dtype(pa.int64())
    # single flat_apply: the input is the source frame, not another flat_apply
    assert not isinstance(op.inputs[0].op, LiteFrameFunctional)


def test_explode_non_contiguous_preserves_column_order():
    """explode of non-contiguous columns preserves original column order.

    Regression: flat_apply groups func-out cols into one block at the first
    subset position, so explode(["a","c"]) on [a,b,c] would yield [a,c,b].
    The appended projection restores pandas-compatible [a,b,c].
    """
    raw = pd.DataFrame({"a": [[1, 2], [3]], "b": [9, 10], "c": [[4, 5], [6]]})
    raw["a"] = raw["a"].astype(_LIST)
    raw["c"] = raw["c"].astype(_LIST)
    result = LiteFrame(raw).explode(["a", "c"])

    assert list(result.columns) == ["a", "b", "c"]
    assert result.dtypes["a"] == wrap_arrow_dtype(pa.int64())
    assert str(result.dtypes["b"]) in ("int64", "int64[pyarrow]")
    assert result.dtypes["c"] == wrap_arrow_dtype(pa.int64())


def test_explode_rejects_unknown_column():
    with pytest.raises(KeyError):
        LiteFrame({"a": [1, 2]}).explode("missing")


def test_explode_rejects_empty_columns():
    """explode([]) must not silently yield an empty frame."""
    with pytest.raises(ValueError, match="at least one column"):
        _list_df().explode([])


@pytest.mark.parametrize("columns", [["lst", "lst"], ["Lst", "lst"]])
def test_explode_rejects_duplicate_columns(columns):
    """Duplicate column names (incl. case-only duplicates) must be rejected
    after normalization, mirroring pandas and LiteFrame's selection APIs."""
    with pytest.raises(ValueError, match="unique after normalization"):
        _list_df().explode(columns)
