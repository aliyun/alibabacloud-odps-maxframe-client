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

"""Compile-time tests for LiteFrame.melt() (no session / no execution).

Covers the operator graph, output schema, dtype handling and the §3 error
contract. End-to-end behaviour is covered by
``dpe/.../reshape/tests/test_melt_execution.py``.
"""

import math

import pandas as pd
import pyarrow as pa
import pytest

from maxframe.liteframe.core import FrameMetadata
from maxframe.liteframe.datasource.from_local import from_local_df
from maxframe.liteframe.expressions import LiteFrameAstype, LiteFrameLiteral, NamedExpr
from maxframe.liteframe.initializer import LiteFrame as LF
from maxframe.liteframe.operators.merge import LiteFrameConcat
from maxframe.liteframe.reshape.melt import _promoted_value_dtype
from maxframe.liteframe.reshape.tests.conftest import _count_expr_type, _op_types
from maxframe.utils import is_arrow_dtype_supported, wrap_arrow_dtype

_requires_arrow_dtype = pytest.mark.skipif(
    not is_arrow_dtype_supported(),
    reason="pandas ArrowDtype is not supported in current environment",
)

_BASE_OPS = {"LiteFrameFromLocal", "LiteFrameProjection", "LiteFrameConcat"}

_PLAIN = LF({"a": ["x", "y"], "b": [1, 3], "c": [2, 4]})


def _var_literal_values(lf):
    """Per-part var-column literal values, in concat (variable-major) order."""
    concat_op = lf.data.op
    if not isinstance(concat_op, LiteFrameConcat):
        return []
    vals = []
    for inp in concat_op.inputs:
        for proj in inp.op.projections:
            expr = proj.expr if isinstance(proj, NamedExpr) else proj
            if isinstance(expr, LiteFrameLiteral):
                vals.append(expr.value)
    return vals


@pytest.mark.parametrize(
    "kwargs,expected_cols,var_vals",
    [
        pytest.param(
            {"id_vars": ["a"], "value_vars": ["b", "c"]},
            ["a", "variable", "value"],
            ["b", "c"],
            id="id_and_values",
        ),
        pytest.param(
            {"value_vars": ["b", "c"]},
            ["variable", "value"],
            ["b", "c"],
            id="no_id_vars",
        ),
        pytest.param(
            {"id_vars": ["a"]},
            ["a", "variable", "value"],
            ["b", "c"],
            id="default_value_vars",
        ),
        pytest.param(
            {"id_vars": ["A"], "value_vars": ["B", "C"]},
            ["a", "variable", "value"],
            ["b", "c"],
            id="lowercase_normalization",
        ),
        pytest.param(
            {"id_vars": "a", "value_vars": "b"},
            ["a", "variable", "value"],
            ["b"],
            id="string_args_wrapped",
        ),
        pytest.param(
            {"id_vars": ["a"], "value_vars": []},
            ["a", "variable", "value"],
            [],
            id="empty_value_vars",
        ),
    ],
)
def test_schema_and_var_values(kwargs, expected_cols, var_vals):
    r = _PLAIN.melt(**kwargs)
    assert list(r.columns) == expected_cols
    assert r.shape[1] == len(expected_cols)
    if var_vals:
        assert _var_literal_values(r) == var_vals


def test_output_shape_unmaterialized():
    r = _PLAIN.melt(id_vars=["a"], value_vars=["b", "c"])
    assert math.isnan(r.shape[0]) if isinstance(r.shape[0], float) else r.shape[0] > 0


def test_graph_uses_existing_operators():
    r = _PLAIN.melt(id_vars=["a"], value_vars=["b", "c"])
    assert _op_types(r) == _BASE_OPS


def test_no_cast_when_uniform():
    r = _PLAIN.melt(id_vars=["a"], value_vars=["b", "c"])
    assert _count_expr_type(r, LiteFrameAstype) == 0


def test_cast_emitted_for_mixed_numeric():
    lf = LF({"a": ["x", "y"], "b": [1, 3], "c": [2.5, 4.5]})
    r = lf.melt(id_vars=["a"], value_vars=["b", "c"])
    # Explicit expectation (not derived from production code): int + double
    # promotes to double under pandas' own promotion rules.
    assert r.dtypes["value"] == wrap_arrow_dtype(pa.float64())
    assert _count_expr_type(r, LiteFrameAstype) > 0


@_requires_arrow_dtype
@pytest.mark.parametrize(
    "left_pa,right_pa",
    [
        # pandas promotes both pairs to object -> must be rejected (LF#1897).
        (pa.duration("ns"), pa.int64()),
        (pa.bool_(), pa.int64()),
    ],
)
def test_incompatible_arrow_promotions_rejected(left_pa, right_pa):
    with pytest.raises(ValueError, match="incompatible dtypes"):
        _promoted_value_dtype([pd.ArrowDtype(left_pa), pd.ArrowDtype(right_pa)])


@_requires_arrow_dtype
def test_promoted_numeric_arrow_dtypes():
    assert _promoted_value_dtype(
        [pd.ArrowDtype(pa.int64()), pd.ArrowDtype(pa.float64())]
    ) == wrap_arrow_dtype(pa.float64())


@_requires_arrow_dtype
def test_promoted_uniform_dtypes_none():
    assert (
        _promoted_value_dtype([pd.ArrowDtype(pa.int64()), pd.ArrowDtype(pa.int64())])
        is None
    )


@_requires_arrow_dtype
def test_incompatible_arrow_dtypes_rejected():
    # duration + int must raise (was silently cast to timedelta(0), LF#1897).
    lf = LF(
        {
            "id": ["a", "b"],
            "d": pd.array([1, 2], dtype=pd.ArrowDtype(pa.duration("ns"))),
            "n": pd.array([1, 2], dtype=pd.ArrowDtype(pa.int64())),
        }
    )
    with pytest.raises(ValueError, match="incompatible dtypes"):
        lf.melt(id_vars=["id"], value_vars=["d", "n"])


def test_empty_value_vars_without_id_melts_all():
    # pandas: value_vars=[] with no id_vars melts every column (LF#1897).
    lf = LF({"a": [1, 2], "b": [3.5, 4.5]})
    r = lf.melt(value_vars=[])
    assert list(r.columns) == ["variable", "value"]
    # an incompatible pair (string + int) still promotes to object and is
    # rejected under the design-doc rule, not silently cast
    with pytest.raises(ValueError, match="incompatible dtypes"):
        LF({"a": ["x", "y"], "b": [1, 3]}).melt(value_vars=[])


def test_empty_value_vars_with_id_is_true_empty():
    # pandas: value_vars=[] with id_vars -> 0-row frame with id cols + var/value.
    r = _PLAIN.melt(id_vars=["a"], value_vars=[])
    assert list(r.columns) == ["a", "variable", "value"]
    assert r.shape[0] == 0


def test_var_literal_dtype_matches_pandas():
    r = _PLAIN.melt(id_vars=["a"], value_vars=["b"])
    assert r.dtypes["variable"] == pd.Series(["b"]).dtype


def test_hidden_columns_dropped():
    raw = pd.DataFrame({"a": ["x", "y"], "b": [1, 3], "_hc": [10.0, 20.0]})
    lf = from_local_df(raw)
    lf._data.frame_metadata = FrameMetadata(hidden_columns=["_hc"])
    lf._data.__dict__.pop("dtypes", None)
    lf._data.__dict__.pop("columns", None)
    r = lf.melt(id_vars=["a"], value_vars=["b"])
    assert "_hc" not in r.columns
    assert list(r.columns) == ["a", "variable", "value"]


@pytest.mark.parametrize(
    "id_vars,value_vars,var_name,value_name,exc",
    [
        (None, 3, "variable", "value", ValueError),
        (None, [1], "variable", "value", ValueError),
        (["a", "a"], ["b"], "variable", "value", ValueError),
        (["a"], ["b", "b"], "variable", "value", ValueError),
        (["a"], ["z"], "variable", "value", KeyError),
        (["z"], ["b"], "variable", "value", KeyError),
        (["a"], ["a"], "variable", "value", ValueError),
        (["a"], ["b"], 3, "value", ValueError),
        (["a"], ["b"], "variable", 3, ValueError),
        (["a"], ["b"], "variable", "a", ValueError),
        (["a"], ["b"], "variable", "b", ValueError),
        (["a"], ["b"], "a", "value", ValueError),
        (["a"], ["b"], "same", "same", ValueError),
    ],
)
def test_errors(id_vars, value_vars, var_name, value_name, exc):
    with pytest.raises(exc):
        _PLAIN.melt(
            id_vars=id_vars,
            value_vars=value_vars,
            var_name=var_name,
            value_name=value_name,
        )


def test_incompatible_value_dtypes_rejected():
    lf = LF({"a": ["x", "y"], "b": [1, 3], "s": ["p", "q"]})
    with pytest.raises(ValueError, match="incompatible dtypes"):
        lf.melt(id_vars=["a"], value_vars=["b", "s"])
