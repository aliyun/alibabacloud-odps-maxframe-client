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

"""Tests for LiteFrame cov / corr (including GroupBy).

Compile-time and DAG-structure tests live here; value-level verification
uses the DPE executor directly (see
``dpe/dpe_execute/.../reduction/tests/test_cov_corr.py``).
"""

import pytest

from maxframe.liteframe.initializer import LiteFrame as LF
from maxframe.liteframe.operators.functional import LiteFrameFunctional
from maxframe.liteframe.operators.project import LiteFrameProjection
from maxframe.liteframe.operators.reduction import LiteFrameAgg
from maxframe.liteframe.reduction.compile import AggCall, compile_agg

_DATA_2 = {"a": [1, 2, 3, 4, 5], "b": [2, 4, 6, 8, 10]}
_DATA_3 = {"a": [1, 2, 3, 4, 5], "b": [2, 4, 6, 8, 10], "c": [5, 4, 3, 2, 1]}
_GB_DATA = {"key": ["x", "x", "y", "y"], "a": [1, 2, 3, 4], "b": [4, 3, 2, 1]}
_PAIRWISE = ["cov", "corr"]
_VAR_FUNCS = ["var", "std", "sem"]


# --- compile-time ----------------------------------------------------------


@pytest.mark.parametrize("func", _PAIRWISE)
def test_compile_basic(func):
    """Compile produces agg_funcs with sum, count, sum_xy, and projection."""
    lf = LF(_DATA_3)
    compiled = compile_agg(LiteFrameAgg(raw_func=AggCall(func, ddof=1)), lf.data)

    names = {d["raw_func_name"] for d in compiled.agg_funcs}
    assert {"sum", "count", "sum_xy"} <= names
    # Pairwise mask primitives must be present for NaN-correct cov/corr
    assert {"sum_x_pair", "sum_y_pair", "count_pair"} <= names
    if func == "corr":
        assert "sum_sq" in names
        assert {"sum_x_pair_sq", "sum_y_pair_sq"} <= names
    assert compiled.projection_exprs
    assert compiled.pre_agg_projection_exprs


@pytest.mark.parametrize("func", _PAIRWISE)
def test_compile_output_shape(func):
    """3 columns → 6 unique pairs (upper triangle incl. diagonal)."""
    lf = LF(_DATA_3)
    compiled = compile_agg(LiteFrameAgg(raw_func=AggCall(func)), lf.data)
    assert len(compiled.agg_output_columns) == 6
    assert compiled.out_shape == (1, 6)


def test_compile_diagonal_and_ddof():
    """Diagonal sum_xy(A,A) reuses sum_sq; ddof reaches the projection."""
    compiled_cov = compile_agg(LiteFrameAgg(raw_func=AggCall("cov")), LF(_DATA_2).data)
    assert (
        len([d for d in compiled_cov.agg_funcs if d["raw_func_name"] == "sum_xy"]) == 1
    )
    assert (
        len([d for d in compiled_cov.agg_funcs if d["raw_func_name"] == "sum_sq"]) == 2
    )

    compiled_ddof = compile_agg(
        LiteFrameAgg(raw_func=AggCall("cov", ddof=2)), LF(_DATA_2).data
    )
    assert compiled_ddof.projection_exprs


def test_compile_pair_descriptors_stored_as_tuples():
    """sum_xy and mask primitives store (col_x, col_y) in desc['pair'], not
    a string that would be lossy on split('__')."""
    compiled = compile_agg(LiteFrameAgg(raw_func=AggCall("cov")), LF(_DATA_3).data)
    for desc in compiled.agg_funcs:
        prim = desc["raw_func_name"]
        if prim in ("sum_xy", "sum_x_pair", "sum_y_pair", "count_pair"):
            assert "pair" in desc
            assert isinstance(desc["pair"], tuple)
            assert len(desc["pair"]) == 2


def test_compile_output_names_use_double_underscore():
    """Output column names use __ joiner to avoid collisions when column
    names contain underscores."""
    data = {"b_c": [1, 2, 3], "a_b": [4, 5, 6], "a": [7, 8, 9]}
    compiled = compile_agg(LiteFrameAgg(raw_func=AggCall("cov")), LF(data).data)
    # All output names must be unique
    names = compiled.agg_output_columns
    assert len(names) == len(set(names))
    # Names use __ joiner
    assert all("__" in n for n in names)


def test_compile_sorted_numeric_cols():
    """Numeric cols are sorted before pair generation so frame order
    matches the lexicographic lookup in _pairwise_matrix_expand."""
    data = {"b": [1, 2, 3], "a": [4, 5, 6]}
    compiled = compile_agg(LiteFrameAgg(raw_func=AggCall("cov")), LF(data).data)
    # First pair should be (a, a) — sorted, not (b, b)
    assert compiled.agg_output_columns[0] == "a__a__cov"


def test_compile_min_periods_no_crash():
    """min_periods must not crash at compile time (LiteFrameWhere needs dtype)."""
    compiled = compile_agg(
        LiteFrameAgg(raw_func=AggCall("cov", min_periods=2)), LF(_DATA_2).data
    )
    assert compiled.projection_exprs


# --- DAG structure ---------------------------------------------------------


@pytest.mark.parametrize("func", _PAIRWISE)
def test_dag_and_output_columns(func):
    """cov/corr → pre-agg proj → agg → post-agg proj → flat_apply."""
    result = getattr(LF(_DATA_3), func)()
    assert isinstance(result.data.op, LiteFrameFunctional)
    assert isinstance(result.data.inputs[0].op, LiteFrameProjection)
    agg_op = result.data.inputs[0].inputs[0].op
    assert isinstance(agg_op, LiteFrameAgg)
    assert agg_op.raw_func == func
    pre_proj = result.data.inputs[0].inputs[0].inputs[0].op
    assert isinstance(pre_proj, LiteFrameProjection)
    assert any("sum_xy" in p.name for p in pre_proj.projections)

    # Output columns: 'columns' (string) + one float64 per numeric input
    assert list(result.columns) == ["columns", "a", "b", "c"]
    assert str(result.dtypes["columns"]) == "string"
    for c in ["a", "b", "c"]:
        assert str(result.dtypes[c]) == "float64"


@pytest.mark.parametrize("func", _PAIRWISE)
def test_groupby_dag(func):
    """groupby().cov()/corr() has agg with groupby_params set."""
    lf = LF(_GB_DATA)
    result = getattr(lf.groupby("key"), func)()
    assert isinstance(result.data.op, LiteFrameFunctional)
    agg_op = result.data.inputs[0].inputs[0].op
    assert isinstance(agg_op, LiteFrameAgg)
    assert agg_op.groupby_params is not None
    assert agg_op.groupby_params["by"] == "key"


# --- ddof for var / std / sem ----------------------------------------------


@pytest.mark.parametrize("func", _VAR_FUNCS)
def test_var_std_sem_uses_aggcall_ddof(func):
    """var/std/sem use AggCall carrying ddof."""
    result = getattr(LF(_DATA_2), func)(ddof=2)
    op = result.data.op
    if isinstance(op, LiteFrameProjection):
        op = result.data.inputs[0].op
    assert isinstance(op, LiteFrameAgg)
    assert op.raw_func == func
    assert op.raw_func.kwds == {"ddof": 2}


@pytest.mark.parametrize("func", _VAR_FUNCS)
@pytest.mark.parametrize("ddof", [0, 1, 2])
def test_var_std_sem_ddof_compile(func, ddof):
    """Compile succeeds with ddof from AggCall."""
    lf = LF(_DATA_2)
    compiled = compile_agg(LiteFrameAgg(raw_func=AggCall(func, ddof=ddof)), lf.data)
    assert compiled.projection_exprs


# --- numeric_only / list agg error (C8, C12) -------------------------------


def test_numeric_only_false_raises_on_non_numeric():
    """cov(numeric_only=False) raises TypeError when non-numeric columns
    are present (pandas 2.0 semantics)."""
    data = {"a": [1, 2, 3], "s": ["x", "y", "z"]}
    with pytest.raises(TypeError, match="non-numeric"):
        LF(data).cov(numeric_only=False)


def test_numeric_only_true_excludes_non_numeric():
    """cov(numeric_only=True) silently excludes non-numeric columns."""
    data = {"a": [1, 2, 3], "b": [4, 5, 6], "s": ["x", "y", "z"]}
    result = LF(data).cov(numeric_only=True)
    assert list(result.columns) == ["columns", "a", "b"]


def test_agg_list_with_cov_raises_targeted_error():
    """lf.agg(['cov']) raises a clear error pointing to lf.cov()."""
    with pytest.raises(ValueError, match="pairwise functions.*call lf.cov"):
        LF(_DATA_2).agg(["cov"])


@pytest.mark.parametrize("func", _PAIRWISE)
def test_agg_bare_string_cov_corr_raises(func):
    """Bare agg('cov'/'corr') raises, not leak internal projection fields."""
    with pytest.raises(ValueError, match="pairwise function.*call lf\\."):
        LF(_DATA_2).agg(func)


@pytest.mark.parametrize("func", _PAIRWISE)
def test_groupby_agg_bare_string_cov_corr_raises(func):
    """groupby().agg('cov') error points to groupby().cov(), not lf.cov()."""
    with pytest.raises(ValueError, match="pairwise function.*call groupby\\(\\)\\."):
        LF(_GB_DATA).groupby("key").agg(func)


# --- pair identity and numeric_only compile tests -------------------------


def test_compile_pair_identity_no_double_underscore_collision():
    """Pairs (a, b__c) and (a__b, c) must have distinct routing keys —
    the pair_<N> index prevents the ``__``-join collision (Fix 7)."""
    data = {
        "a": [1, 2, 3, 4, 5],
        "b__c": [5, 4, 3, 2, 1],
        "a__b": [2, 4, 6, 8, 10],
        "c": [10, 8, 6, 4, 2],
    }
    compiled = compile_agg(LiteFrameAgg(raw_func=AggCall("cov")), LF(data).data)
    names = compiled.agg_output_columns
    assert len(names) == len(set(names))


def test_compile_numeric_only_false_raises():
    """_compile_pairwise_agg rejects non-numeric cols when numeric_only=False."""
    data = {"a": [1, 2], "s": ["x", "y"]}
    with pytest.raises(TypeError, match="non-numeric"):
        compile_agg(
            LiteFrameAgg(raw_func=AggCall("cov", numeric_only=False)),
            LF(data).data,
        )


def test_compile_numeric_only_true_excludes():
    """_compile_pairwise_agg silently excludes non-numeric cols when
    numeric_only=True."""
    data = {"a": [1, 2], "s": ["x", "y"]}
    compiled = compile_agg(
        LiteFrameAgg(raw_func=AggCall("cov", numeric_only=True)),
        LF(data).data,
    )
    assert compiled.agg_output_columns == ["a__a__cov"]


def test_compile_min_periods_in_aggcall_kwds():
    """min_periods is carried via AggCall.kwds to _compile_pairwise_agg."""
    compiled = compile_agg(
        LiteFrameAgg(raw_func=AggCall("cov", min_periods=3)), LF(_DATA_2).data
    )
    assert compiled.projection_exprs
