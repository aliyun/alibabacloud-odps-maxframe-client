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
import pandas as pd
import polars as pl
import pyarrow as pa
import pytest

from maxframe.codegen.spe.liteframe.expressions import LiteFrameExprEvaluator
from maxframe.liteframe import LiteFrame
from maxframe.liteframe.accessors.string_.accessor import _build_split_dtypes
from maxframe.liteframe.accessors.string_.core import (
    LiteFrameStrExpr,
    LiteFrameStrMultiExpr,
    str_multi_method_to_handlers,
)
from maxframe.liteframe.core import LiteFrameResult
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
        # partition/rpartition and split/rsplit are symmetric at the
        # structure level; one of each pair suffices.
        ("partition", ("_",), {}, ["0", "1", "2"], wrap_arrow_dtype(pa.string())),
        ("extract", (r"(\d)(\w)",), {}, ["0", "1"], wrap_arrow_dtype(pa.string())),
        (
            "split",
            ("_",),
            {"expand": True, "n": 1},
            ["0", "1"],
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
        # expand=False is the default; explicit vs implicit is the same
        # path. One split + one rsplit covers both symmetric variants.
        ("split", ("_",), {"expand": False, "n": 2}),
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


def test_str_split_expand_true_dynamic_schema():
    """split with expand=True and n<=0 produces a schema-dynamic projection
    whose column count is resolved at execution time."""
    df = LiteFrame({"col": ["a1b2", "c3d4"]})
    result = df.str.split("_", expand=True, n=-1)

    assert result.is_schema_dynamic
    assert np.isnan(result.data.shape[1])
    assert list(result.data.columns) == []
    # The projection has 2 inputs: data LiteFrame + dtypes Object tileable
    assert len(result.data.inputs) == 2
    op = result.data.op
    assert isinstance(op, LiteFrameProjection)
    assert len(op.projections) == 1
    proj = op.projections[0]
    assert isinstance(proj, LiteFrameStrMultiExpr)
    assert proj.is_dynamic
    assert proj.method == "split"


@pytest.mark.parametrize(
    ("method", "sep", "n"),
    [
        ("split", "_", -1),
        ("split", ".", -1),  # literal separator (not regex)
        ("split", None, -1),  # whitespace split
        ("split", "_", 0),  # n=0 also dynamic
        ("rsplit", "_", -1),  # rsplit dynamic
        ("rsplit", None, -1),  # rsplit whitespace
    ],
    ids=[
        "split_under",
        "split_dot",
        "split_ws",
        "split_n0",
        "rsplit_under",
        "rsplit_ws",
    ],
)
def test_str_split_rsplit_dynamic_schema_metadata(method, sep, n):
    """split/rsplit with expand=True and n<=0 produces a schema-dynamic
    projection regardless of separator (literal, whitespace) or method."""
    df = LiteFrame({"col": ["a.b_c", "d_e.f"]})
    args = (sep,) if sep is not None else ()
    kwargs = {"expand": True, "n": n}
    if sep is None:
        kwargs.pop("n", None)  # whitespace split doesn't take n in some paths
        kwargs["n"] = n
    result = getattr(df.str, method)(*args, **kwargs)

    assert result.is_schema_dynamic
    assert np.isnan(result.data.shape[1])
    assert list(result.data.columns) == []
    assert len(result.data.inputs) == 2
    op = result.data.op
    assert isinstance(op, LiteFrameProjection)
    proj = op.projections[0]
    assert isinstance(proj, LiteFrameStrMultiExpr)
    assert proj.is_dynamic
    assert proj.method == method


@pytest.mark.parametrize(
    ("max_val", "n_rows", "expected_n_cols", "expected_labels"),
    [
        # Normal: max field count = 3 → 3 columns
        (3, 2, 3, ["0", "1", "2"]),
        # Single field
        (1, 1, 1, ["0"]),
        # Zero fields (e.g. whitespace-only "   " → [])
        (0, 1, 0, []),
        # Empty source: 0 columns
        (np.nan, 0, 0, []),
        # Non-empty all-null: 1 null column
        (np.nan, 2, 1, ["0"]),
        # DataFrame input (max() result shape)
        (pd.DataFrame({0: [2]}), 2, 2, ["0", "1"]),
        # LiteFrameResult inputs (resolve_tileable_input=True path):
        # max() and size().max() arrive as LiteFrameResult wrappers.
        # max() is float64 because _field_count returns float (for consistent
        # schema across null/non-null chunks).
        (
            LiteFrameResult(_frame=pl.DataFrame({"max": [3.0]})),
            LiteFrameResult(_frame=pl.DataFrame({"size": [2]})),
            3,
            ["0", "1", "2"],
        ),
        # LiteFrameResult all-null: max is NaN → 1 column
        (
            LiteFrameResult(
                _frame=pl.DataFrame({"max": [None]}, schema={"max": pl.Float64})
            ),
            LiteFrameResult(_frame=pl.DataFrame({"size": [2]})),
            1,
            ["0"],
        ),
    ],
    ids=[
        "normal",
        "single_field",
        "zero_fields",
        "empty",
        "all_null",
        "dataframe_input",
        "liteframe_result",
        "liteframe_result_all_null",
    ],
)
def test_build_split_dtypes_edge_cases(
    max_val, n_rows, expected_n_cols, expected_labels
):
    """_build_split_dtypes produces the correct column count and labels for
    empty, all-null, and normal inputs — matching pandas str.split(expand=True)
    semantics."""
    result = _build_split_dtypes(max_val, n_rows)
    assert len(result) == expected_n_cols
    assert list(result.index) == expected_labels


def test_build_split_dtypes_dtype_is_string():
    """All expanded columns carry the string ArrowDtype."""
    result = _build_split_dtypes(2, 2)
    assert all(
        result.iloc[i] == wrap_arrow_dtype(pa.string()) for i in range(len(result))
    )


def test_str_extract_no_groups_raises():
    """extract with a pattern containing no capture groups raises ValueError."""
    df = LiteFrame({"col": ["a1b2", "c3d4"]})
    with pytest.raises(ValueError, match="no capture groups"):
        df.str.extract(r"\d\w")


@pytest.mark.parametrize(
    ("method", "args", "kwargs", "data"),
    [
        # partition/rpartition and split/rsplit are symmetric at the
        # codegen level; one of each pair suffices.
        ("partition", ("_",), {}, ["a_b", "c_d_e"]),
        ("extract", (r"(\d)(\w)",), {}, ["a1b2", "c3d4"]),
        ("split", ("_",), {"expand": True, "n": 1}, ["a_b", "c_d_e"]),
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
