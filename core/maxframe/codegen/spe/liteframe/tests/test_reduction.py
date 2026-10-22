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

from maxframe.codegen.spe.core import SPECodeContext
from maxframe.codegen.spe.liteframe.reduction import LiteFrameAggAdapter
from maxframe.liteframe.operators.reduction import LiteFrameAgg


def _agg(func, cols, key="agg_0"):
    """Build a single agg_func entry for _run_agg."""
    return {"map_func_name": func, "cols": cols, "output_key": key, "kwds": {}}


def test_agg_adapter_code_generation(make_liteframe):
    """LiteFrameAggAdapter should generate _run_agg call."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    data = make_liteframe(df)
    op = LiteFrameAgg(raw_func="sum")
    out = op(data)
    op = out.op
    adapter = LiteFrameAggAdapter()
    context = SPECodeContext()
    result = adapter.generate_code(op, context)
    assert len(result) == 1
    assert "LiteFrameAggAdapter._run_agg" in result[0]


@pytest.mark.parametrize(
    "df_data, agg_funcs, output_cols, expected",
    [
        pytest.param(
            {"a": [1, 2, 3], "b": [4, 5, 6]},
            [_agg("sum", ["a"]), _agg("sum", ["b"], "agg_1")],
            ["a", "b"],
            {"a": 6, "b": 15},
            id="sum",
        ),
        pytest.param(
            {"a": [1, 2, None], "b": [4, 5, 6]},
            [_agg("count", ["a"]), _agg("count", ["b"], "agg_1")],
            ["a", "b"],
            {"a": 2, "b": 3},
            id="count",
        ),
        pytest.param(
            {"a": [1, 2, 3]},
            [_agg("size", ["a"])],
            ["count"],
            {"count": 3},
            id="size",
        ),
    ],
)
def test_agg_run_whole_frame(df_data, agg_funcs, output_cols, expected):
    """Test _run_agg over a whole (non-grouped) frame."""
    df = pd.DataFrame(df_data)
    result = LiteFrameAggAdapter._run_agg(df, agg_funcs, None, output_cols)
    assert len(result) == 1
    for col, val in expected.items():
        assert result[col].iloc[0] == val


@pytest.mark.parametrize(
    "df_data, func_name, output_col, expected_keys, expected_vals",
    [
        pytest.param(
            {"key": [1, 1, 2, 2], "val": [1, 2, 3, 4]},
            "sum",
            "val",
            [1, 2],
            [3, 7],
            id="sum",
        ),
        pytest.param(
            {"key": ["a", "a", "b"], "val": [1, 2, 3]},
            "size",
            "__size__",
            ["a", "b"],
            [2, 1],
            id="size",
        ),
    ],
)
def test_agg_run_grouped(df_data, func_name, output_col, expected_keys, expected_vals):
    """Test _run_agg with a groupby."""
    df = pd.DataFrame(df_data)
    agg_funcs = [_agg(func_name, ["val"])]
    groupby_params = {"by": "key", "sort": False, "dropna": True}
    result = LiteFrameAggAdapter._run_agg(df, agg_funcs, groupby_params, None)
    result = result.sort_values("key").reset_index(drop=True)
    assert list(result["key"]) == expected_keys
    assert list(result[output_col]) == expected_vals


def test_agg_run_whole_frame_min_max():
    """Test _run_agg with min and max."""
    df = pd.DataFrame({"a": [3, 1, 2]})
    agg_funcs = [
        {"map_func_name": "min", "cols": ["a"], "output_key": "agg_0", "kwds": {}},
        {"map_func_name": "max", "cols": ["a"], "output_key": "agg_1", "kwds": {}},
    ]
    result = LiteFrameAggAdapter._run_agg(df, agg_funcs, None, ["min_a", "max_a"])
    assert result["min_a"].iloc[0] == 1
    assert result["max_a"].iloc[0] == 3


import pyarrow as pa

from maxframe.liteframe.reduction.custom_agg import CustomAgg
from maxframe.utils.datatypes import is_arrow_dtype_supported, wrap_arrow_dtype


def _run_custom_grouped(df, agg, cols, dtypes):
    """Run a custom agg in standard groupby mode (by="key", sort/dropna
    defaults as in production LiteFrame groupby)."""
    desc = _custom_agg_desc(agg, cols, dtypes)
    groupby_params = {"by": "key", "sort": False, "dropna": True}
    return LiteFrameAggAdapter._run_agg(df, [desc], groupby_params, None)


def _custom_agg_desc(custom_func, cols, agg_output_dtypes, key="agg_0"):
    """Build a custom-agg agg_func entry mirroring compile_custom_agg output."""
    return {
        "raw_func_name": getattr(custom_func, "__name__", "<custom>"),
        "map_func_name": "custom_agg",
        "agg_func_name": "custom_agg",
        "cols": cols,
        "agg_output_cols": list(agg_output_dtypes.index),
        "output_key": key,
        "kwds": {},
        "custom_func": custom_func,
        "agg_output_dtypes": agg_output_dtypes,
        "input_frame_type": "pandas",
    }


class _SumABAgg(CustomAgg):
    """Sum columns a and b into a single 'total' column."""

    def agg(self, df):
        return pd.DataFrame({"total": [df["a"].sum() + df["b"].sum()]})


class _StatefulCountAgg(CustomAgg):
    """Count rows using internal state to verify per-call isolation.

    The result depends on the accumulated counter: with per-call isolation
    (fresh deepcopy per group) two groups of sizes 2 and 1 yield [2, 1];
    a reused instance would report [2, 3], so the isolation assertion has a
    real regression oracle.
    """

    def __init__(self):
        self._count = 0

    def agg(self, df):
        self._count += len(df)
        return pd.DataFrame({"count": [self._count]})


class _KeyCheckerAgg(CustomAgg):
    """Record the UDF-visible columns (group key must be excluded)."""

    def agg(self, df):
        return pd.DataFrame({"cols": [",".join(sorted(df.columns))]})


def _double_first(df):
    """Plain callable (not a CustomAgg subclass); called directly."""
    return pd.DataFrame({"doubled": [df["a"].sum() * 2]})


@pytest.mark.parametrize(
    "agg, cols, dtypes, output_cols, expected",
    [
        pytest.param(
            _SumABAgg(),
            ["a", "b"],
            pd.Series({"total": wrap_arrow_dtype(pa.int64())}),
            ["total"],
            {"total": 21},
            id="custom_agg_instance",
        ),
        pytest.param(
            _double_first,
            ["a"],
            pd.Series({"doubled": wrap_arrow_dtype(pa.int64())}),
            ["doubled"],
            {"doubled": 12},
            id="plain_callable",
        ),
    ],
)
def test_custom_agg_whole_frame(agg, cols, dtypes, output_cols, expected):
    """Custom agg over a whole (non-grouped) frame: class instances are
    deep-copied per call, plain callables invoked directly."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    desc = _custom_agg_desc(agg, cols, dtypes)
    result = LiteFrameAggAdapter._run_agg(df, [desc], None, output_cols)
    for col, val in expected.items():
        assert result[col].iloc[0] == val


@pytest.mark.parametrize(
    "agg, cols, dtypes, expected",
    [
        pytest.param(
            _SumABAgg(),
            ["a", "b"],
            pd.Series({"total": wrap_arrow_dtype(pa.int64())}),
            {"key": ["x", "y"], "total": [12, 9]},
            id="preserves_group_keys_and_totals",
        ),
        pytest.param(
            _KeyCheckerAgg(),
            ["a"],
            pd.Series({"cols": wrap_arrow_dtype(pa.string())}),
            {"key": ["x", "y"], "cols": ["a", "a"]},
            id="excludes_group_key_from_input",
        ),
        pytest.param(
            _StatefulCountAgg(),
            ["a"],
            pd.Series({"count": wrap_arrow_dtype(pa.int64())}),
            {"key": ["x", "y"], "count": [2, 1]},
            id="isolates_state_across_groups",
        ),
    ],
)
def test_custom_agg_grouped(agg, cols, dtypes, expected):
    """Custom agg per group: group keys re-materialized, input narrowed to the
    declared cols (group-key-excluded), per-group state isolated."""
    df = pd.DataFrame({"key": ["x", "x", "y"], "a": [1, 2, 3], "b": [4, 5, 6]})
    result = _run_custom_grouped(df, agg, cols, dtypes)
    result = result.sort_values("key").reset_index(drop=True)
    for col, vals in expected.items():
        assert list(result[col]) == vals


@pytest.mark.parametrize(
    "key_dtype",
    [
        pytest.param(None, id="plain_object_key"),
        pytest.param(
            "arrow",
            id="arrow_string_key",
            marks=pytest.mark.skipif(
                not is_arrow_dtype_supported(),
                reason="pandas ArrowDtype is not supported",
            ),
        ),
    ],
)
def test_custom_agg_grouped_key_first_and_dtype(key_dtype):
    """Group keys are prepended (declared key-first order), not appended
    after the UDF result columns, and keep the source key dtype instead of
    being re-inferred as object by scalar assignment."""
    if key_dtype == "arrow":
        key = pd.Series(["x", "x", "y"], dtype=pd.ArrowDtype(pa.string()))
    else:
        key = ["x", "x", "y"]
    df = pd.DataFrame({"key": key, "a": [1, 2, 3], "b": [4, 5, 6]})
    result = _run_custom_grouped(
        df, _SumABAgg(), ["a", "b"], pd.Series({"total": wrap_arrow_dtype(pa.int64())})
    )
    assert list(result.columns) == ["key", "total"]
    assert str(result["key"].dtype) == str(df["key"].dtype)


@pytest.mark.parametrize(
    "df_data",
    [
        pytest.param({"key": [], "a": [], "b": []}, id="empty_input"),
        pytest.param(
            {"key": [None, None], "a": [1, 2], "b": [3, 4]},
            id="all_null_keys_dropna",
        ),
    ],
)
def test_custom_agg_grouped_no_groups(df_data):
    """Empty input / all-null keys (dropna=True) yield a zero-row frame with
    the declared key + agg schema instead of raising on pd.concat([])."""
    df = pd.DataFrame(df_data)
    result = _run_custom_grouped(
        df, _SumABAgg(), ["a", "b"], pd.Series({"total": wrap_arrow_dtype(pa.int64())})
    )
    assert result.empty
    assert list(result.columns) == ["key", "total"]
    assert str(result["key"].dtype) == str(df["key"].dtype)


def test_custom_agg_close_called():
    """close() is released after agg() even on success."""
    df = pd.DataFrame({"a": [1, 2, 3]})
    closed = []

    class _CloseAgg(CustomAgg):
        def agg(self, d):
            return pd.DataFrame({"v": [d["a"].sum()]})

        def close(self):
            closed.append(True)

    desc = _custom_agg_desc(_CloseAgg(), ["a"], pd.Series({"v": "int64"}))
    LiteFrameAggAdapter._run_agg(df, [desc], None, ["v"])
    assert closed == [True]


def test_custom_agg_code_generation(make_liteframe):
    """LiteFrameAggAdapter generates _run_agg for a custom-agg op."""
    if not is_arrow_dtype_supported():
        pytest.skip("pandas ArrowDtype is not supported")
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    data = make_liteframe(df)
    op = LiteFrameAgg(raw_func=_SumABAgg())
    out = op(data)
    adapter = LiteFrameAggAdapter()
    context = SPECodeContext()
    code = adapter.generate_code(out.op, context)
    assert len(code) == 1
    assert "LiteFrameAggAdapter._run_agg" in code[0]


def test_custom_agg_end_to_end(make_liteframe):
    """Full codegen + exec produces the correct result for a custom agg."""
    if not is_arrow_dtype_supported():
        pytest.skip("pandas ArrowDtype is not supported")
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    data = make_liteframe(df)
    op = LiteFrameAgg(raw_func=_SumABAgg())
    out = op(data)

    adapter = LiteFrameAggAdapter()
    context = SPECodeContext()
    code = adapter.generate_code(out.op, context)

    exec_ns = {"LiteFrameAggAdapter": LiteFrameAggAdapter}
    for cname, cval in context.constants.items():
        exec_ns[cname] = cval
    exec_ns["var_0"] = df
    exec(code[0], exec_ns)
    assert exec_ns["var_1"]["total"].iloc[0] == 21
