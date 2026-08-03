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
import pytest

from maxframe.liteframe.core import RANGE_COL_NAME, FrameMetadata
from maxframe.liteframe.datasource.from_local import from_local_df
from maxframe.liteframe.merge.merge import _normalize_keys
from maxframe.protocol import DefaultIndexType


def _make_range_lf(pdf_data):
    """Create a LiteFrame with a virtual range column."""
    pdf = pd.DataFrame(pdf_data)
    return from_local_df(pdf, default_index_type=DefaultIndexType.range)


def _make_hidden_lf(pdf_data, hidden_names):
    """Create a LiteFrame with hidden columns."""
    pdf = pd.DataFrame(pdf_data)
    lf = from_local_df(pdf)
    fm = FrameMetadata(hidden_columns=hidden_names)
    lf._data.frame_metadata = fm
    lf._data.__dict__.pop("dtypes", None)
    lf._data.__dict__.pop("columns", None)
    return lf


@pytest.mark.parametrize(
    "keys, expected",
    [
        (None, None),
        ("KEY", "key"),
        ("key", "key"),
        (["KEY", "Id"], ["key", "id"]),
        (["a", "b"], ["a", "b"]),
    ],
)
def test_normalize_keys(keys, expected):
    assert _normalize_keys(keys) == expected


@pytest.mark.parametrize(
    "merge_kwargs, expected_op_attr, expected_op_value",
    [
        ({"on": "KEY"}, "on", "key"),
        ({"left_on": "K1", "right_on": "K2"}, "left_on", "k1"),
        ({"left_on": "K1", "right_on": "K2"}, "right_on", "k2"),
    ],
)
def test_merge_key_normalization(merge_kwargs, expected_op_attr, expected_op_value):
    left = from_local_df(pd.DataFrame({"key": [1, 2], "k1": [1, 2], "a": [10, 20]}))
    right = from_local_df(pd.DataFrame({"key": [2, 3], "k2": [2, 3], "b": [200, 300]}))
    merged = left.merge(right, **merge_kwargs)
    assert getattr(merged.op, expected_op_attr) == expected_op_value


def test_merge_key_normalization_on_list():
    left = from_local_df(
        pd.DataFrame({"key1": [1, 2], "key2": ["a", "b"], "v": [10, 20]})
    )
    right = from_local_df(
        pd.DataFrame({"key1": [2, 3], "key2": ["b", "c"], "w": [200, 300]})
    )
    merged = left.merge(right, on=["KEY1", "KEY2"])
    assert merged.op.on == ["key1", "key2"]


@pytest.mark.parametrize(
    "check_description, check_fn",
    [
        (
            "range columns not in output physical dtypes",
            lambda m: RANGE_COL_NAME not in m._data._physical_dtypes.index,
        ),
        (
            "range columns not in output range_columns",
            lambda m: not (
                m.frame_metadata
                and m.frame_metadata.range_columns
                and RANGE_COL_NAME in m.frame_metadata.range_columns
            ),
        ),
        (
            "range columns not in visible dtypes",
            lambda m: RANGE_COL_NAME not in m.dtypes.index,
        ),
        (
            "physical columns preserved in visible dtypes",
            lambda m: {"key", "a", "b"}.issubset(m.dtypes.index),
        ),
    ],
)
def test_merge_drops_range_columns(check_description, check_fn):
    left = _make_range_lf({"key": [1, 2], "a": [10, 20]})
    right = _make_range_lf({"key": [2, 3], "b": [200, 300]})

    # Inputs have __index__ as a virtual range column
    assert RANGE_COL_NAME in left.frame_metadata.range_columns
    assert RANGE_COL_NAME in right.frame_metadata.range_columns

    merged = left.merge(right, on="key")
    assert check_fn(merged), f"Failed: {check_description}"


def test_merge_auto_key_uses_on_not_left_right_on():
    """Default common-key case should set on=, not left_on/right_on.

    Pandas treats the default case as on=common_cols. Setting left_on/right_on
    makes the executor take the different-name-key path which can create
    duplicate key columns.
    """
    left = _make_range_lf({"key": [1, 2], "a": [10, 20]})
    right = _make_range_lf({"key": [2, 3], "b": [200, 300]})

    merged = left.merge(right)
    assert merged.op.on == ["key"]
    assert merged.op.left_on is None
    assert merged.op.right_on is None


@pytest.mark.parametrize(
    "check_description, check_fn",
    [
        (
            "hidden columns in output physical dtypes",
            lambda m: "_hc" in m._data._physical_dtypes.index,
        ),
        (
            "hidden columns not in visible dtypes",
            lambda m: "_hc" not in m.dtypes.index,
        ),
        (
            "hidden columns tracked in frame_metadata",
            lambda m: "_hc" in m._hidden_columns,
        ),
    ],
)
def test_merge_preserves_hidden_columns(check_description, check_fn):
    left = _make_hidden_lf(
        {"key": [1, 2], "a": [10, 20], "_hc": [1.0, 2.0]}, hidden_names=["_hc"]
    )
    right = from_local_df(pd.DataFrame({"key": [2, 3], "b": [200, 300]}))

    merged = left.merge(right, on="key")
    assert check_fn(merged), f"Failed: {check_description}"
