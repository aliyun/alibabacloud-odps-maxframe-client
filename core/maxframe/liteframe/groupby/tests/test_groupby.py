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

from maxframe.liteframe.core import LiteFrame
from maxframe.liteframe.datasource.from_local import from_local_df
from maxframe.liteframe.expressions import (
    LiteFrameColumn,
    LiteFrameLiteral,
    LiteFrameWhere,
)
from maxframe.liteframe.groupby.core import LiteFrameGroupBy
from maxframe.liteframe.initializer import LiteFrame as LF
from maxframe.liteframe.operators.functional import LiteFrameFunctional
from maxframe.liteframe.operators.groupby import LiteFrameGroupByOp
from maxframe.liteframe.operators.project import LiteFrameProjection
from maxframe.liteframe.operators.reduction import LiteFrameAgg
from maxframe.liteframe.reduction.compile import AggCall
from maxframe.utils import is_arrow_dtype_supported, wrap_arrow_dtype


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


@pytest.mark.parametrize(
    "conflicting_col",
    ["__mf_gb_ci__", "__mf_gb_ri__"],
    ids=["ci_collision", "ri_collision"],
)
def test_groupby_placeback_collision_raises(conflicting_col):
    """A frame already carrying a routing-column name must raise ValueError
    when placeback is requested, so the user column is not silently lost."""

    pdf = pd.DataFrame({"key": ["a", "b"], "val": [1, 2], conflicting_col: [10, 20]})
    lf = from_local_df(pdf)
    op = LiteFrameGroupByOp(groupby_params={"by": "key", "sort": False, "dropna": True})
    op.extra_params["placeback"] = True
    with pytest.raises(ValueError, match="reserves internal column names"):
        op(lf)


def test_groupby_placeback_no_collision_succeeds():
    """Without a naming collision, placeback=True adds routing columns
    without error."""

    pdf = pd.DataFrame({"key": ["a", "b"], "val": [1, 2]})
    lf = from_local_df(pdf)
    op = LiteFrameGroupByOp(groupby_params={"by": "key", "sort": False, "dropna": True})
    op.extra_params["placeback"] = True
    result = op(lf)
    assert result.shape[1] == lf.shape[1] + 2


# --- min_count tests ---


@pytest.mark.parametrize("func_name", ["sum", "prod", "min", "max", "first", "last"])
def test_groupby_min_count_default_no_projection(func_name):
    """Default min_count does not trigger projection."""
    lf = LF({"key": ["a", "b", "a"], "val": [1, 2, 3]})
    result = getattr(lf.groupby("key"), func_name)()
    assert isinstance(result.data.op, LiteFrameAgg)


@pytest.mark.parametrize(
    "func_name,min_count",
    [("sum", 3), ("prod", 2), ("min", 2), ("max", 2), ("first", 2), ("last", 2)],
)
def test_groupby_min_count_triggers_where_guard(func_name, min_count):
    """Positive min_count triggers projection with LiteFrameWhere + None."""
    lf = LF({"key": ["a", "b", "a"], "val": [1, 2, 3]})
    result = getattr(lf.groupby("key"), func_name)(min_count=min_count)

    proj_op = result.data.op
    assert isinstance(proj_op, LiteFrameProjection)
    val_proj = next(p for p in proj_op.projections if p.name == "val")
    assert isinstance(val_proj.expr, LiteFrameWhere)
    assert isinstance(val_proj.expr.false_value, LiteFrameLiteral)
    assert val_proj.expr.false_value.value is None


@pytest.mark.parametrize("func_name", ["min", "max", "first", "last"])
def test_groupby_min_count_one_noop(func_name):
    """min_count=1 is a no-op for min/max/first/last — no projection."""
    lf = LF({"key": ["a", "b", "a"], "val": [1, 2, 3]})
    result = getattr(lf.groupby("key"), func_name)(min_count=1)
    assert isinstance(result.data.op, LiteFrameAgg)


@pytest.mark.skipif(not is_arrow_dtype_supported(), reason="Arrow dtype not supported")
@pytest.mark.parametrize("func_name", ["sum", "min", "first", "last"])
def test_groupby_min_count_preserves_int64_dtype(func_name):
    """min_count preserves Int64 output dtype."""
    lf = LF({"key": ["a", "b", "a"], "val": [1, 2, 3]})
    result = getattr(lf.groupby("key"), func_name)(min_count=2)
    assert result.dtypes["val"] == wrap_arrow_dtype(pa.int64())


@pytest.mark.skipif(not is_arrow_dtype_supported(), reason="Arrow dtype not supported")
@pytest.mark.parametrize("pa_dtype", [pa.int8(), pa.int32(), pa.float32(), pa.bool_()])
@pytest.mark.parametrize("func_name", ["sum", "prod", "min", "max", "first", "last"])
def test_groupby_min_count_preserves_widened_dtype(func_name, pa_dtype):
    """Guarded min_count produces the same dtype as the unguarded aggregation
    for narrow input types (int8/int32/float32/bool)."""
    data = pd.Series([1, 2, 3], dtype=wrap_arrow_dtype(pa_dtype))
    lf = LF({"key": ["a", "b", "a"], "val": data})
    unguarded = getattr(lf.groupby("key"), func_name)()
    guarded = getattr(lf.groupby("key"), func_name)(min_count=2)
    assert guarded.dtypes["val"] == unguarded.dtypes["val"]


@pytest.mark.parametrize("func_name", ["first", "last"])
def test_groupby_min_count_skipna_false_true_value_is_column(func_name):
    """first/last(skipna=False, min_count=2) produces a Where whose
    true_value is a LiteFrameColumn (skipna-aware intermediate)."""
    lf = LF({"key": ["a", "b", "a"], "val": [1, 2, 3]})
    result = getattr(lf.groupby("key"), func_name)(skipna=False, min_count=2)

    val_proj = next(p for p in result.data.op.projections if p.name == "val")
    assert isinstance(val_proj.expr, LiteFrameWhere)
    assert isinstance(val_proj.expr.true_value, LiteFrameColumn)


@pytest.mark.parametrize("func_name", ["sum", "first", "last"])
def test_groupby_min_count_shares_count(func_name):
    """func(min_count=2) and sum(min_count=2) share a single count intermediate."""
    lf = LF({"key": ["a", "b", "a"], "val": [1.0, 2.0, 3.0]})
    result = lf.groupby("key").agg(
        [AggCall(func_name, min_count=2), AggCall("sum", min_count=2)]
    )
    assert isinstance(result.data.op, LiteFrameProjection)
    agg_op = result.data.inputs[0].op
    counts = [d for d in agg_op.agg_funcs if d["map_func_name"] == "count"]
    assert len(counts) == 1  # shared, not duplicated
    assert any(d["map_func_name"] == func_name for d in agg_op.agg_funcs)
    assert any(d["map_func_name"] == "sum" for d in agg_op.agg_funcs)


# ---------------------------------------------------------------------------
# groupby.filter tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "data,call,expected_cols",
    [
        pytest.param(
            {"key": ["a", "a", "b", "b"], "val": [1, 2, 3, 4]},
            lambda lf: lf.groupby("key").filter(lambda g: g["val"].sum() > 3),
            ["key", "val"],
            id="scalar_true",
        ),
        pytest.param(
            {"key": ["a", "a", "b", "b"], "val": [1, 2, 3, 4]},
            lambda lf: lf.groupby("key").filter(lambda g: g["val"].sum() < 3),
            ["key", "val"],
            id="scalar_false",
        ),
        pytest.param(
            {"key": ["a", "a", "b", "b"], "val": [1, 2, 3, 4]},
            lambda lf: lf.groupby("key").filter(lambda g: False),
            ["key", "val"],
            id="all_groups_dropped",
        ),
        pytest.param(
            {"val": [1, 2, 3, 4], "key": ["a", "a", "b", "b"]},
            lambda lf: lf.groupby("key").filter(lambda g: True),
            ["val", "key"],
            id="preserves_column_order",
        ),
        pytest.param(
            {"key": ["a", "a", "b", "b"], "val": [1, 2, 3, 4]},
            lambda lf: lf.groupby("key").filter(lambda g, t: g["val"].sum() > t, t=3),
            ["key", "val"],
            id="forward_kwargs",
        ),
        pytest.param(
            {"key": ["a", "a", "b", "b"], "val": [1, 2, 3, 4]},
            # pandas positional order: filter(func, dropna, *args)
            lambda lf: lf.groupby("key").filter(lambda g: g["val"].sum() > 3, False),
            ["key", "val"],
            id="dropna_positional",
        ),
        pytest.param(
            {"key": ["a", "a", "b", "b"], "val": [1, 2, 3, 4]},
            lambda lf: lf.groupby("key").filter(
                lambda g: g["val"].sum() > 3, dropna=False
            ),
            ["key", "val"],
            id="dropna_false_keeps_columns",
        ),
    ],
)
def test_groupby_filter(data, call, expected_cols):
    result = call(LF(data))
    assert isinstance(result, LiteFrame)
    assert list(result.columns) == expected_cols


def test_groupby_filter_uses_apply():
    """groupby.filter delegates to groupby_apply (LiteFrameFunctional)."""
    result = (
        LF({"key": ["a", "a", "b", "b"], "val": [1, 2, 3, 4]})
        .groupby("key")
        .filter(lambda g: g["val"].sum() > 3)
    )
    op = result.data.op
    while isinstance(op, LiteFrameProjection):
        op = op.inputs[0].op
    assert isinstance(op, LiteFrameFunctional)


@pytest.mark.parametrize(
    "selection,group_keys,expected_cols",
    [
        pytest.param(None, True, ["key", "val", "w"], id="default"),
        pytest.param(["val"], True, ["val"], id="selected"),
        pytest.param(None, False, ["key", "val", "w"], id="group_keys_false"),
        pytest.param(["val"], False, ["val"], id="selected_group_keys_false"),
    ],
)
def test_groupby_filter_column_output(selection, group_keys, expected_cols):
    """filter returns correct columns for selected/group_keys variants (pandas)."""
    lf = LF({"key": ["a", "a", "b", "b"], "val": [1, 2, 3, 4], "w": [10, 20, 30, 40]})
    gb = lf.groupby("key", group_keys=group_keys)
    if selection is not None:
        gb = gb[selection]
    result = gb.filter(lambda g: g["val"].sum() > 3)
    assert list(result.columns) == expected_cols
