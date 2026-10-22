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

from maxframe.codegen.spe.liteframe.expressions import LiteFrameExprEvaluator
from maxframe.liteframe import LiteFrame
from maxframe.liteframe.accessors.string_.core import (
    LiteFrameStrExpr,
    LiteFrameStrMultiExpr,
    str_multi_method_to_handlers,
)
from maxframe.liteframe.expressions import (
    LiteFrameColumn,
    LiteFrameMultiColumnSubExpr,
    NamedExpr,
)
from maxframe.liteframe.operators.project import LiteFrameProjection
from maxframe.utils import wrap_arrow_dtype


def test_str_accessor_single_column_validation():
    """Test accessor only works on single-column LiteFrame."""
    df = LiteFrame({"a": ["hello"], "b": ["world"]})
    with pytest.raises(ValueError, match="multi-column"):
        df.str.upper()


@pytest.mark.parametrize(
    "method,args,expected_dtype",
    [
        # One representative per dtype class; per-method dtype correctness
        # is covered by STR_METHOD_OUTPUT_DTYPE + execution tests.
        ("upper", [], wrap_arrow_dtype(pa.string())),
        ("contains", ["hello"], wrap_arrow_dtype(pa.bool_())),
        ("len", [], wrap_arrow_dtype(pa.int64())),
        ("find", ["o"], wrap_arrow_dtype(pa.int64())),
        ("repeat", [2], wrap_arrow_dtype(pa.string())),
    ],
)
def test_str_methods_dtype_inference(method, args, expected_dtype):
    """Test str methods: dtype inference without execution."""
    df = LiteFrame({"col": ["hello", "world"]})
    result = getattr(df.str, method)(*args)

    # Verify result is LiteFrame
    assert isinstance(result, LiteFrame)

    # Verify operator structure
    op = result.data.op
    assert isinstance(op, LiteFrameProjection)
    assert len(op.projections) == 1

    # Verify projection structure
    projection = op.projections[0]
    assert isinstance(projection, NamedExpr)
    assert projection.name == "col"
    assert isinstance(projection.expr, LiteFrameStrExpr)
    assert projection.expr.method == method
    assert projection.expr.args == tuple(args)

    # Verify dtype inference
    assert result.dtypes.iloc[0] == expected_dtype
    assert projection.dtype == expected_dtype


def test_string_accessor_after_projection_fuses():
    """String accessor on a projection-produced column should fuse."""
    lf = LiteFrame({"s": ["abc", "de"]})
    lf2 = lf.select("s")
    result = lf2.str.upper()

    op = result.data.op
    assert isinstance(op, LiteFrameProjection)
    # The operand should be a LiteFrameColumn, not another projection layer
    expr = op.projections[0].expr
    assert hasattr(expr, "operand")
    assert isinstance(expr.operand, LiteFrameColumn)
    # Input should be the root source
    assert result.data.inputs[0] is lf._data


@pytest.mark.parametrize(
    ("method", "args", "kwargs", "expected_cols", "expected_dtype"),
    [
        ("partition", ("_",), {}, ["0", "1", "2"], wrap_arrow_dtype(pa.string())),
        ("rpartition", ("_",), {}, ["0", "1", "2"], wrap_arrow_dtype(pa.string())),
        ("extract", (r"(\d)(\w)",), {}, ["0", "1"], wrap_arrow_dtype(pa.string())),
        (
            "split",
            ("_",),
            {"expand": True, "n": 1},
            ["0", "1"],
            wrap_arrow_dtype(pa.string()),
        ),
        (
            "rsplit",
            ("_",),
            {"expand": True, "n": 2},
            ["0", "1", "2"],
            wrap_arrow_dtype(pa.string()),
        ),
    ],
)
def test_str_multi_column_structure(
    method, args, kwargs, expected_cols, expected_dtype
):
    """Multi-column str methods produce a projection with N NamedExpr sub-exprs."""
    df = LiteFrame({"col": ["a1b2", "c3d4"]})
    result = getattr(df.str, method)(*args, **kwargs)

    assert isinstance(result, LiteFrame)
    assert list(result.columns) == expected_cols

    op = result.data.op
    assert isinstance(op, LiteFrameProjection)
    assert len(op.projections) == len(expected_cols)

    for i, name in enumerate(expected_cols):
        proj = op.projections[i]
        assert isinstance(proj, NamedExpr)
        assert proj.name == name
        assert isinstance(proj.expr, LiteFrameMultiColumnSubExpr)
        assert proj.expr.output_name == name
        assert proj.expr.dtype == expected_dtype


@pytest.mark.parametrize(
    ("method", "args", "kwargs"),
    [
        ("split", ("_",), {}),
        ("split", ("_",), {"expand": False}),
        ("split", ("_",), {"expand": False, "n": 2}),
        ("rsplit", ("_",), {}),
        ("rsplit", ("_",), {"expand": False}),
    ],
)
def test_str_split_expand_false_produces_list_column(method, args, kwargs):
    """split/rsplit with expand=False returns a single list-of-string column
    (pandas semantics) rather than being rejected."""
    from maxframe.lib.dtypes_extension import is_list_dtype

    df = LiteFrame({"col": ["a_b_c_d", "1_2_3_4"]})
    result = getattr(df.str, method)(*args, **kwargs)

    op = result.data.op
    assert isinstance(op, LiteFrameProjection)
    assert len(op.projections) == 1
    assert isinstance(op.projections[0].expr, LiteFrameStrExpr)
    assert is_list_dtype(result.dtypes.iloc[0])


@pytest.mark.parametrize("n", [-1, 0], ids=["n=-1", "n=0"])
def test_str_split_expand_true_unknown_n_rejected(n):
    """split with expand=True and n<=0 has a runtime-dependent column count
    (pandas treats n<=0 as unlimited), so it is rejected at graph-build
    time (no single-column fallback)."""
    df = LiteFrame({"col": ["a1b2", "c3d4"]})
    with pytest.raises(ValueError, match="expand=True"):
        df.str.split("_", expand=True, n=n)


def test_str_extract_no_groups_raises():
    """extract with a pattern containing no capture groups raises ValueError."""
    df = LiteFrame({"col": ["a1b2", "c3d4"]})
    with pytest.raises(ValueError, match="no capture groups"):
        df.str.extract(r"\d\w")


@pytest.mark.parametrize(
    ("method", "args", "kwargs", "data"),
    [
        ("partition", ("_",), {}, ["a_b", "c_d_e"]),
        ("rpartition", ("_",), {}, ["a_b", "c_d_e"]),
        ("extract", (r"(\d)(\w)",), {}, ["a1b2", "c3d4"]),
        ("split", ("_",), {"expand": True, "n": 1}, ["a_b", "c_d_e"]),
        ("rsplit", ("_",), {"expand": True, "n": 1}, ["a_b", "c_d_e"]),
    ],
)
def test_str_multi_column_codegen(method, args, kwargs, data):
    """Multi-column str expression evaluates correctly via LiteFrameExprEvaluator."""
    df = pd.DataFrame({"s": data})
    col = LiteFrameColumn(name="s", dtype=wrap_arrow_dtype(pa.string()))

    # Infer output names via the handler
    handler = str_multi_method_to_handlers[method]
    dtypes_map = handler.infer_dtypes(
        method, wrap_arrow_dtype(pa.string()), *args, **kwargs
    )
    output_names = [str(k) for k in dtypes_map.keys()]

    multi = LiteFrameStrMultiExpr(
        operands=[col],
        method=method,
        args=args,
        kwargs=kwargs,
        output_names=output_names,
        dtypes={str(k): str(v) for k, v in dtypes_map.items()},
    )

    evaluator = LiteFrameExprEvaluator(df)
    expected = getattr(df["s"].str, method)(*args, **kwargs)
    expected.columns = output_names

    for name in output_names:
        sub = LiteFrameMultiColumnSubExpr(
            multi_expr=multi,
            output_name=name,
            dtype=dtypes_map[int(name)],
        )
        result = evaluator.eval(sub)
        pd.testing.assert_series_equal(result, expected[name])
