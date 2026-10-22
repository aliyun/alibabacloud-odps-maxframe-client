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

import pyarrow as pa
import pytest

from maxframe.liteframe.core import LiteFrame
from maxframe.liteframe.expressions import LiteFrameColumn, LiteFramePow, NamedExpr
from maxframe.liteframe.initializer import LiteFrame as LF
from maxframe.liteframe.operators.project import LiteFrameProjection
from maxframe.liteframe.operators.reduction import LiteFrameAgg
from maxframe.liteframe.reduction.compile import (
    AggCall,
    _infer_output_column_names,
    _normalize_func,
    _normalize_func_kw,
    _resolve_callable_names,
    compile_agg,
)
from maxframe.utils import wrap_arrow_dtype


def _get_agg_op(result):
    """Get the LiteFrameAgg operator from a result.

    When projection is chained, the top op is LiteFrameProjection and the
    agg is its input. Otherwise the top op is LiteFrameAgg directly.
    """
    op = result.data.op
    if isinstance(op, LiteFrameProjection):
        # The input to projection is the agg result
        return result.data.inputs[0].op
    return op


def test_agg_single_func():
    """lf.sum() creates a LiteFrameAgg operator with correct fields."""
    lf = LF({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
    result = lf.sum()
    assert isinstance(result, LiteFrame)
    op = result.data.op
    assert isinstance(op, LiteFrameAgg)
    assert op.raw_func == "sum"
    assert op.groupby_params is None
    # Single func: output column names should match input physical columns
    assert list(result.columns) == ["a", "b"]


def test_agg_multiple_funcs():
    """lf.agg(["sum", "mean"]) produces suffixed column names."""
    lf = LF({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
    result = lf.agg(["sum", "mean"])
    assert isinstance(result, LiteFrame)
    # mean is recipe-based, so the top op is a projection
    op = result.data.op
    assert isinstance(op, LiteFrameProjection)
    # The underlying agg op should store the raw_func
    agg_op = _get_agg_op(result)
    assert isinstance(agg_op, LiteFrameAgg)
    assert agg_op.raw_func == ["sum", "mean"]
    # Multiple funcs: suffixed column names
    assert list(result.columns) == [
        "a_sum",
        "a_mean",
        "b_sum",
        "b_mean",
    ]


@pytest.mark.parametrize(
    "func_spec,expected_columns",
    [
        ({"a": "sum"}, ["a"]),
        ({"a": ["sum", "mean"]}, ["a_sum", "a_mean"]),
    ],
)
def test_agg_dict_func(func_spec, expected_columns):
    """lf.agg(dict) applies to specific column(s) with correct naming."""
    lf = LF({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
    result = lf.agg(func_spec)
    assert isinstance(result, LiteFrame)
    assert list(result.columns) == expected_columns


def test_agg_named_agg():
    """lf.agg(total=("a", "sum")) uses the provided name."""
    lf = LF({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
    result = lf.agg(total=("a", "sum"))
    assert isinstance(result, LiteFrame)
    assert "total" in list(result.columns)


def test_agg_duplicate_direct_named_agg():
    """lf.agg(t1=("a","sum"), t2=("a","sum")) fans out via projection."""
    lf = LF({"a": [1, 2, 3]})
    result = lf.agg(t1=("a", "sum"), t2=("a", "sum"))
    assert isinstance(result, LiteFrame)
    cols = list(result.columns)
    assert "t1" in cols
    assert "t2" in cols
    # Underlying agg should have a single sum(a) descriptor (deduped),
    # and a projection fans it out to two output names.
    agg_op = _get_agg_op(result)
    assert isinstance(agg_op, LiteFrameAgg)
    sum_descs = [d for d in agg_op.agg_funcs if d["map_func_name"] == "sum"]
    assert len(sum_descs) == 1
    assert result.data.op.__class__.__name__ == "LiteFrameProjection"


def test_compile_sum_and_mean_share_intermediate():
    """sum and mean on same column should share the sum intermediate."""
    lf = LF({"a": [1.0, 2.0, 3.0]})
    result = lf.agg({"a": ["sum", "mean"]})
    agg_op = _get_agg_op(result)

    # With deduplication: sum(a) + count(a) = 2 descriptors, not 3
    assert len(agg_op.agg_funcs) == 2
    # Both should be decomposable (no passthrough)
    assert not any(d.get("passthrough") for d in agg_op.agg_funcs)


def test_compile_mean_var_std_share_intermediates():
    """mean, var, std on same column should produce only 3 intermediates."""
    lf = LF({"a": [1.0, 2.0, 3.0]})
    result = lf.agg({"a": ["mean", "var", "std"]})
    agg_op = _get_agg_op(result)

    # sum + sum_sq + count = 3 intermediates (shared across all three)
    assert len(agg_op.agg_funcs) == 3
    func_names = sorted(d["map_func_name"] for d in agg_op.agg_funcs)
    assert func_names == ["count", "sum", "sum_sq"]


def test_compile_mean_produces_projection_exprs():
    """mean should produce a LiteFrameProjection with NamedExpr."""
    lf = LF({"a": [1.0, 2.0, 3.0]})
    result = lf.mean()

    # The result should be produced by a LiteFrameProjection op
    op = result.data.op
    assert isinstance(op, LiteFrameProjection)
    # Projections should be NamedExpr instances
    assert all(isinstance(p, NamedExpr) for p in op.projections)
    # Output name should be "a"
    assert op.projections[0].name == "a"


# --- Error path tests for compile_agg ---


def test_normalize_func_unsupported_type():
    """Unsupported aggregation function type raises TypeError."""
    with pytest.raises(TypeError, match="Unsupported aggregation function type"):
        _normalize_func(12345)  # int is not supported


@pytest.mark.parametrize(
    "bad_spec",
    [
        "sum",  # not a tuple
        ("a", "sum", "extra"),  # wrong tuple length
    ],
)
def test_normalize_func_kw_invalid_spec(bad_spec):
    """Named aggregation with invalid spec raises ValueError."""
    with pytest.raises(ValueError, match="Named aggregation requires"):
        _normalize_func_kw({"total": bad_spec})


# --- Multi-key groupby at operator level ---


def test_groupby_multi_key():
    """Multi-key groupby creates correct operator."""
    lf = LF({"k1": ["a", "a", "b"], "k2": ["x", "y", "x"], "v": [1, 2, 3]})
    result = lf.groupby(["k1", "k2"]).sum()
    assert isinstance(result, LiteFrame)
    agg_op = _get_agg_op(result)
    assert isinstance(agg_op, LiteFrameAgg)
    assert agg_op.groupby_params["by"] == ["k1", "k2"]
    # Output should include key columns and value columns
    assert "k1" in list(result.columns)
    assert "k2" in list(result.columns)
    assert "v" in list(result.columns)


# --- Shortcut method tests ---


@pytest.mark.parametrize(
    "func",
    [
        "sum",
        "mean",
        "min",
        "max",
        "var",
        "std",
        "count",
        "prod",
        "sem",
        "skew",
        "kurt",
        "nunique",
        "median",
    ],
)
def test_shortcut_methods_create_agg_op(func):
    """Shortcut methods (lf.sum(), lf.mean(), etc.) create LiteFrameAgg with correct raw_func."""
    lf = LF({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    result = getattr(lf, func)()
    assert isinstance(result, LiteFrame)
    agg_op = _get_agg_op(result)
    assert isinstance(agg_op, LiteFrameAgg)
    assert agg_op.raw_func == func


@pytest.mark.parametrize(
    "func,expected_exponents",
    [
        ("var", [2]),
        ("std", [2]),
        ("sem", [2]),
        ("skew", [2, 3]),
        ("kurt", [2, 3, 4]),
    ],
)
def test_compile_pre_agg_projection(func, expected_exponents):
    """Recipe funcs should produce pre-agg projection exprs with correct exponents."""
    lf = LF({"a": [1.0, 2.0, 3.0, 4.0, 5.0]})
    op = LiteFrameAgg(raw_func=func)
    compiled = compile_agg(op, lf.data)

    assert len(compiled.pre_agg_projection_exprs) == len(expected_exponents)
    for named in compiled.pre_agg_projection_exprs:
        assert isinstance(named, NamedExpr)
        assert isinstance(named.expr, LiteFramePow)
    exponents = sorted(
        named.expr.right.value for named in compiled.pre_agg_projection_exprs
    )
    assert exponents == expected_exponents

    # Power-sum descriptors should have pre_agg_col set
    for desc in compiled.agg_funcs:
        if desc["raw_func_name"] in ("sum_sq", "sum_cube", "sum_fourth"):
            assert "pre_agg_col" in desc
            assert desc["pre_agg_col"].startswith("agg_")
            assert "_input" in desc["pre_agg_col"]
        else:
            assert "pre_agg_col" not in desc


# --- Pre-agg projection DAG structure tests ---


def test_var_dag_has_pre_agg_projection():
    """lf.var() should produce: pre-agg projection -> agg -> post-agg projection."""
    lf = LF({"a": [1.0, 2.0, 3.0]})
    result = lf.var()

    # Top op: post-agg projection
    post_proj_op = result.data.op
    assert isinstance(post_proj_op, LiteFrameProjection)

    # Input to post-agg projection: agg
    agg_op = result.data.inputs[0].op
    assert isinstance(agg_op, LiteFrameAgg)

    # Input to agg: pre-agg projection
    pre_proj_op = result.data.inputs[0].inputs[0].op
    assert isinstance(pre_proj_op, LiteFrameProjection)

    # Pre-agg projection should have pass-through columns + power-sum columns
    pre_proj_names = [p.name for p in pre_proj_op.projections]
    assert "a" in pre_proj_names  # pass-through
    assert any("sum_sq_input" in n for n in pre_proj_names)  # power-sum


def test_sum_dag_has_no_pre_agg_projection():
    """lf.sum() should NOT produce a pre-agg projection (no power-sum needed)."""
    lf = LF({"a": [1.0, 2.0, 3.0]})
    result = lf.sum()

    # Top op is LiteFrameAgg directly (no post-agg projection either)
    op = result.data.op
    assert isinstance(op, LiteFrameAgg)


# --- Passthrough in post-agg projection tests ---


def test_mixed_recipe_and_passthrough_includes_passthrough_in_projection():
    """Mixed mean+median should include median in the post-agg projection.

    Previously, passthrough functions (like median) were silently dropped
    from the projection when recipe-based functions (like mean) were present.
    This test verifies the projection now includes both.
    """
    lf = LF({"a": [1.0, 2.0, 3.0, 4.0, 5.0]})
    result = lf.agg({"a": ["mean", "median"]})

    # Top op should be a projection (because mean is recipe-based)
    proj_op = result.data.op
    assert isinstance(proj_op, LiteFrameProjection)

    # Output columns should include both mean and median
    assert list(result.columns) == ["a_mean", "a_median"]

    # Projection should have an entry for each output column
    proj_names = [p.name for p in proj_op.projections]
    assert "a_mean" in proj_names
    assert "a_median" in proj_names

    # The median projection should be a simple column reference (passthrough)
    median_proj = next(p for p in proj_op.projections if p.name == "a_median")
    assert isinstance(median_proj.expr, LiteFrameColumn)


@pytest.mark.parametrize(
    "dropna,expected", [(True, {"dropna": True}), (False, {"dropna": False})]
)
def test_nunique_dropna(dropna, expected):
    """nunique(dropna=...) propagates to agg_funcs descriptor kwds."""
    lf = LF({"a": [1, None, 1]})
    result = lf.nunique(dropna=dropna)
    agg_op = _get_agg_op(result)
    nunique_descs = [d for d in agg_op.agg_funcs if d["map_func_name"] == "nunique"]
    assert len(nunique_descs) == 1
    assert nunique_descs[0]["kwds"] == expected


def test_nunique_dropna_mixed_named_agg():
    """lf.agg with two AggCall("nunique", ...) with different dropna preserves per-call kwargs."""
    lf = LF({"a": [1, None, 1]})
    result = lf.agg(
        a_drop=("a", AggCall("nunique", dropna=True)),
        a_keep=("a", AggCall("nunique", dropna=False)),
    )
    agg_op = _get_agg_op(result)
    nunique_descs = [d for d in agg_op.agg_funcs if d["map_func_name"] == "nunique"]
    assert len(nunique_descs) == 2
    kwds_set = {tuple(sorted(d["kwds"].items())) for d in nunique_descs}
    assert (("dropna", False),) in kwds_set
    assert (("dropna", True),) in kwds_set


# --- size naming (frame-level, pandas "size" convention) ---

_SIZE_FRAME_LEVEL_CASES = [
    ({"key": ["a", "b", "a"], "val": [1, 2, 3]}, True, ["key", "size"], ["val"]),
    (
        {"key": ["a", "b", "a"], "v1": [1, 2, 3], "v2": [4, 5, 6]},
        True,
        ["key", "size"],
        ["v1", "v2"],
    ),
    ({"a": [1, 2], "b": [3, 4]}, False, ["size"], ["a", "b"]),
]


@pytest.mark.parametrize(
    "data,grouped,expected_cols,size_cols",
    _SIZE_FRAME_LEVEL_CASES,
    ids=["grouped-single-col", "grouped-multi-col", "whole-frame"],
)
def test_size_single_frame_level_output(data, grouped, expected_cols, size_cols):
    """size is frame-level: one "size" int64 column, emitted from a single
    frame-level descriptor, regardless of the target column count."""
    lf = LF(data)
    result = (lf.groupby("key") if grouped else lf).size()
    agg_op = _get_agg_op(result)

    assert list(result.columns) == expected_cols
    size_descs = [d for d in agg_op.agg_funcs if d["map_func_name"] == "size"]
    assert len(size_descs) == 1
    assert size_descs[0]["cols"] == size_cols
    assert result.dtypes["size"] == wrap_arrow_dtype(pa.int64())


@pytest.mark.parametrize(
    "funcs,expected_cols,size_key",
    [
        (["size", "mean"], ["key", "size", "val_mean"], "agg_0____size__"),
        (["mean", "size"], ["key", "val_mean", "size"], "agg_2____size__"),
    ],
    ids=["size-first", "mean-first"],
)
def test_groupby_size_mixed_with_recipe(funcs, expected_cols, size_key):
    """size combined with a recipe function compiles and names correctly.

    Regression: after size became a single frame-level descriptor (registry
    key ``(size, None)``), the post-agg projection looked up ``(size, col)``
    and raised KeyError during compile.
    """
    lf = LF({"key": ["a", "b", "a"], "val": [1, 2, 3]})
    result = lf.groupby("key").agg(funcs)

    assert list(result.columns) == expected_cols
    assert result.dtypes["size"] == wrap_arrow_dtype(pa.int64())
    proj_op = result.data.op
    assert isinstance(proj_op, LiteFrameProjection)
    size_proj = next(p for p in proj_op.projections if p.name == "size")
    # References the executor's frame-level size intermediate (one per group).
    assert isinstance(size_proj.expr, LiteFrameColumn)
    assert size_proj.expr.name == size_key


# --- Recipe output dtype binding (int input -> float output) ---


@pytest.mark.parametrize("grouped", [True, False], ids=["grouped", "whole-frame"])
def test_mean_int_input_infers_float(grouped):
    """mean over int input compiles to float64 output.

    The agg post-projection (sum/count) had dtype=None before the projection
    binding fix, so int/int division was planned as int instead of float.
    """
    data = {"key": ["a", "b", "a"], "val": [1, 2, 3]} if grouped else {"a": [1, 2, 3]}
    target = "val" if grouped else "a"
    lf = LF(data)
    result = (lf.groupby("key") if grouped else lf).mean()

    proj_op = result.data.op
    assert isinstance(proj_op, LiteFrameProjection)
    assert result.dtypes[target] == wrap_arrow_dtype(pa.float64())
    # The recipe expression itself resolves to float64 (int/int division).
    mean_proj = next(p for p in proj_op.projections if p.name == target)
    assert mean_proj.expr.dtype == wrap_arrow_dtype(pa.float64())


# --- Callable column naming ---


def _named(name):
    """Create a function with a specific __name__."""

    def f(x):
        return x.sum()

    f.__name__ = name
    return f


class _CallableInstance:
    """Callable instance without __name__ — resolves to class name."""

    def __call__(self, x):
        return x.sum()


@pytest.mark.parametrize(
    "funcs,expected_names",
    [
        # Single callable — no index
        ([lambda x: x.sum()], ["<lambda>"]),
        ([_named("my_sum")], ["my_sum"]),
        ([_CallableInstance()], ["_CallableInstance"]),
        # Two different-named callables — no index
        ([_named("my_sum"), _named("my_mean")], ["my_sum", "my_mean"]),
        # Two lambdas — indexed inside angle brackets
        (
            [lambda x: x.sum(), lambda x: x.mean()],
            ["<lambda_0>", "<lambda_1>"],
        ),
        # Three lambdas
        (
            [lambda x: x.sum(), lambda x: x.mean(), lambda x: x.max()],
            ["<lambda_0>", "<lambda_1>", "<lambda_2>"],
        ),
        # Two same-named functions — suffix index
        ([_named("shared"), _named("shared")], ["shared_0", "shared_1"]),
        # Two callable instances of same class — suffix index
        (
            [_CallableInstance(), _CallableInstance()],
            ["_CallableInstance_0", "_CallableInstance_1"],
        ),
    ],
    ids=[
        "single-lambda",
        "single-named",
        "single-callable-instance",
        "two-different-named",
        "two-lambdas",
        "three-lambdas",
        "two-same-named",
        "two-same-class-instances",
    ],
)
def test_resolve_callable_names(funcs, expected_names):
    """_resolve_callable_names disambiguates callables sharing the same name.

    Single callable keeps its original name. Multiple callables with the
    same name are numbered: angle-bracket names use inside-bracket index
    (<lambda_0>), plain names use suffix (shared_0).
    """
    assert _resolve_callable_names(funcs) == expected_names


@pytest.mark.parametrize(
    "raw_func,expected_columns",
    [
        # Single callable — no suffix
        ([lambda x: x.sum()], ["a", "b"]),
        # Two named callables — {col}_{func} suffix
        (
            [_named("my_sum"), _named("my_mean")],
            ["a_my_sum", "a_my_mean", "b_my_sum", "b_my_mean"],
        ),
        # Two lambdas — {col}_{<lambda_N>}
        (
            [lambda x: x.sum(), lambda x: x.mean()],
            ["a_<lambda_0>", "a_<lambda_1>", "b_<lambda_0>", "b_<lambda_1>"],
        ),
        # String + single lambda — no index on lambda (no collision)
        (["sum", lambda x: x.mean()], ["a_sum", "a_<lambda>", "b_sum", "b_<lambda>"]),
    ],
    ids=[
        "single-callable-no-suffix",
        "two-named-callables-suffixed",
        "two-lambdas-indexed",
        "string-plus-single-lambda",
    ],
)
def test_infer_output_column_names_callables(raw_func, expected_columns):
    """_infer_output_column_names produces {col}_{func} for multiple callables.

    Single callable per column keeps the column name. Multiple callables
    get the callable's resolved name as the func component.
    """
    func_dict, func_rename = _normalize_func(raw_func)
    cols = _infer_output_column_names(func_dict, func_rename, ["a", "b"], None)
    assert cols == expected_columns
