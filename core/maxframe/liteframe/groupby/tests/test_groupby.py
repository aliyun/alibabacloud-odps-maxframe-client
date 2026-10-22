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

import pytest

from maxframe.liteframe.core import LiteFrame
from maxframe.liteframe.groupby.core import LiteFrameGroupBy
from maxframe.liteframe.initializer import LiteFrame as LF
from maxframe.liteframe.operators.reduction import LiteFrameAgg


def test_groupby_returns_api_object():
    """lf.groupby("key") returns a LiteFrameGroupBy API object."""
    lf = LF({"key": ["a", "b", "a"], "a": [1, 2, 3]})
    gb = lf.groupby("key")
    assert isinstance(gb, LiteFrameGroupBy)
    assert gb.groupby_params["by"] == "key"
    assert gb.groupby_params["sort"] is True
    assert gb.groupby_params["dropna"] is True
    assert gb.groupby_params["group_keys"] is True


def test_groupby_group_keys_param():
    """group_keys propagates through ``groupby()`` / ``__getitem__``."""
    lf = LF({"k": ["a", "a", "b"], "v": [1, 2, 3]})
    assert lf.groupby("k", group_keys=False).groupby_params["group_keys"] is False
    assert lf.groupby("k")[["v"]].groupby_params["group_keys"] is True
    assert (
        lf.groupby("k", group_keys=False)[["v"]].groupby_params["group_keys"] is False
    )


@pytest.mark.parametrize(
    "group_keys, expected_cols", [(True, ["k", "0"]), (False, ["0"])], ids=["on", "off"]
)
def test_groupby_group_keys_gates_apply(group_keys, expected_cols):
    """group_keys gates apply output: keys prepended by default (reset_index
    shape), omitted with False -- the declared schema matches."""
    lf = LF({"k": ["a", "a", "b"], "v": [1, 2, 3]})
    res = lf.groupby("k", group_keys=group_keys).apply(lambda g: g["v"].sum())
    assert list(res.dtypes.index) == expected_cols


def test_groupby_sum():
    """lf.groupby("key").sum() creates LiteFrameAgg with groupby_params."""
    lf = LF({"key": ["a", "b", "a"], "val": [1, 2, 3]})
    result = lf.groupby("key").sum()
    assert isinstance(result, LiteFrame)
    op = result.data.op
    assert isinstance(op, LiteFrameAgg)
    assert op.groupby_params is not None
    assert op.groupby_params["by"] == "key"
    # Single func with groupby: group key + original column names
    assert "key" in list(result.columns)
    assert "val" in list(result.columns)


def test_groupby_multiple_funcs():
    """lf.groupby("key").agg(["sum", "mean"]) produces suffixed names."""
    lf = LF({"key": ["a", "b", "a"], "val": [1, 2, 3]})
    result = lf.groupby("key").agg(["sum", "mean"])
    assert isinstance(result, LiteFrame)
    assert "key" in list(result.columns)
    assert "val_sum" in list(result.columns)
    assert "val_mean" in list(result.columns)


def test_groupby_named_agg():
    """lf.groupby("key").agg(total=("val", "sum")) uses the provided name."""
    lf = LF({"key": ["a", "b", "a"], "val": [1, 2, 3]})
    result = lf.groupby("key").agg(total=("val", "sum"))
    assert isinstance(result, LiteFrame)
    cols = list(result.columns)
    assert "key" in cols
    assert "total" in cols
