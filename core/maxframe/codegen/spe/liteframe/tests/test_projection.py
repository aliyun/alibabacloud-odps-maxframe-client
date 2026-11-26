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

try:
    import polars as pl
except ImportError:  # pragma: no cover
    pl = None

from maxframe.codegen.spe.core import SPECodeContext, SPECodeGenerator
from maxframe.codegen.spe.liteframe.projection import (
    LiteFrameProjectionAdapter,
    _extract_input_range_columns,
    _resolve_range_state,
    _run_udf,
)
from maxframe.liteframe import LiteFrame
from maxframe.liteframe.datasource.from_local import from_local_df
from maxframe.liteframe.expressions import (
    LiteFrameAdd,
    LiteFrameColumn,
    LiteFrameLiteral,
    LiteFrameUDFExpr,
    NamedExpr,
)
from maxframe.liteframe.functional.apply import apply
from maxframe.liteframe.functional.map import map_
from maxframe.liteframe.functional.transform import transform
from maxframe.liteframe.functional.transform_chunk import transform_chunk
from maxframe.liteframe.operators.project import LiteFrameProjection
from maxframe.protocol import DefaultIndexType
from maxframe.utils import wrap_arrow_dtype

RANGE_COL_NAME = "__mf_range_index__"


def test_projection_adapter_column_select(make_liteframe):
    """LiteFrameProjectionAdapter should generate _run_projection call."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    data = make_liteframe(df)
    proj = LiteFrameProjection(
        projections=[
            LiteFrameColumn(name="a", dtype=wrap_arrow_dtype(pa.int64())),
        ],
    )
    out = proj(data)
    op = out.op
    adapter = LiteFrameProjectionAdapter()
    context = SPECodeContext()
    result = adapter.generate_code(op, context)
    assert len(result) == 1
    assert "LiteFrameProjectionAdapter._run_projection" in result[0]


def test_projection_run_projection_basic():
    """Test _run_projection static method with column selection."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    projections = [
        LiteFrameColumn(name="a", dtype=wrap_arrow_dtype(pa.int64())),
    ]
    result = LiteFrameProjectionAdapter._run_projection(
        df, projections, set(), False, None
    )
    assert list(result.columns) == ["a"]
    pd.testing.assert_series_equal(result["a"], df["a"])


def test_projection_run_projection_with_expression():
    """Test _run_projection with an arithmetic expression."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    col_a = LiteFrameColumn(name="a", dtype=wrap_arrow_dtype(pa.int64()))
    col_b = LiteFrameColumn(name="b", dtype=wrap_arrow_dtype(pa.int64()))
    add_expr = LiteFrameAdd(left=col_a, right=col_b, dtype=wrap_arrow_dtype(pa.int64()))
    projections = [
        NamedExpr(name="c", expr=add_expr, dtype=wrap_arrow_dtype(pa.int64())),
    ]
    result = LiteFrameProjectionAdapter._run_projection(
        df, projections, set(), False, None
    )
    assert list(result.columns) == ["c"]
    expected = df["a"] + df["b"]
    expected.name = "c"
    pd.testing.assert_series_equal(result["c"], expected)


def test_projection_run_projection_empty_projections_preserves_rows():
    """A projection set yielding no columns still carries the input height.

    Regression: a bare ``pd.DataFrame({})`` collapses to (0, 0), so a
    zero-column projection silently dropped every input row.
    """
    df = pd.DataFrame({"a": [1, 2, 3]})

    result = LiteFrameProjectionAdapter._run_projection(df, [], set(), False, None)

    assert result.shape == (3, 0)


def test_projection_drops_passthrough_range_column(make_liteframe):
    """Selecting only a physical column drops the virtual range column."""
    df = pd.DataFrame({"a": [10, 20, 30]})
    lf = from_local_df(df, default_index_type=DefaultIndexType.range)
    assert RANGE_COL_NAME in lf.columns

    out = lf[["a"]]
    op = out.op
    adapter = LiteFrameProjectionAdapter()
    context = SPECodeContext()
    code = adapter.generate_code(op, context)
    assert "LiteFrameProjectionAdapter._run_projection" in code[0]

    result = LiteFrameProjectionAdapter._run_projection(
        df, op.projections, set(), False, {RANGE_COL_NAME: (0, 3, 1)}
    )
    assert list(result.columns) == ["a"]
    assert result["a"].tolist() == [10, 20, 30]


def test_projection_affine_range_stays_virtual():
    """An affine transform of a range column stays virtual (skipped)."""
    df = pd.DataFrame({"a": [10, 20, 30]})
    i64 = wrap_arrow_dtype(pa.int64())
    rc = LiteFrameColumn(name=RANGE_COL_NAME, dtype=i64)
    add_expr = LiteFrameAdd(
        left=rc,
        right=LiteFrameLiteral(value=1, dtype=i64),
        dtype=i64,
    )
    proj = LiteFrameProjection(
        projections=[
            NamedExpr(name="idx_plus_1", expr=add_expr, dtype=i64),
            LiteFrameColumn(name="a", dtype=i64),
        ],
    )
    lf = from_local_df(df, default_index_type=DefaultIndexType.range)
    out = proj(lf)

    # The affine range column stays virtual in the output metadata.
    assert out.frame_metadata is not None
    assert "idx_plus_1" in out.frame_metadata.range_columns

    op = out.op
    adapter = LiteFrameProjectionAdapter()
    context = SPECodeContext()
    code = adapter.generate_code(op, context)
    assert "True" in code[0] or "False" in code[0]

    # Resolve the virtual set the same way the adapter does.
    in_range_columns = _extract_input_range_columns(op.inputs[0])
    virtual_range_names, need_materialize = _resolve_range_state(op, in_range_columns)
    assert need_materialize is False
    assert "idx_plus_1" in virtual_range_names

    result = LiteFrameProjectionAdapter._run_projection(
        df, op.projections, virtual_range_names, need_materialize, in_range_columns
    )
    # Virtual range column is not materialized into the physical frame.
    assert list(result.columns) == ["a"]
    assert result["a"].tolist() == [10, 20, 30]


# ---------------------------------------------------------------------------
# UDF projections (map / transform_chunk / transform_row)
# ---------------------------------------------------------------------------


def test_projection_map_udf():
    """map() UDF applies element-wise via the projection adapter."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30]})
    lf = from_local_df(df)
    res = map_(lf, lambda x: x * 2)
    dag = res.build_graph()
    g = SPECodeGenerator("s1").generate(dag)
    lv = g.constants.copy()
    lv["running"] = True
    exec(g.code, {}, lv)
    out_key = list(g.output_key_to_variables.values())[0]
    result = lv[out_key]
    assert result["a"].tolist() == [2, 4, 6]
    assert result["b"].tolist() == [20, 40, 60]


@pytest.mark.parametrize("kind", ["callable-instance", "unresolvable-input-hint"])
def test_projection_map_scalar_annotation_executes(kind):
    """map() honours a resolvable return annotation and executes correctly."""
    calls = []

    if kind == "callable-instance":

        class Func:
            def __call__(self, value) -> str:
                calls.append(value)
                return "v" + str(value)

        func = Func()
    else:

        def unresolvable(value: "MissingInput") -> str:  # noqa: F821
            calls.append(value)
            return "v" + str(value)

        func = unresolvable

    lf = from_local_df(pd.DataFrame({"a": [1, 2]}))
    res = map_(lf, func)
    assert calls == []
    assert res.dtypes["a"] == wrap_arrow_dtype(pa.string())

    dag = res.build_graph()
    g = SPECodeGenerator("s1").generate(dag)
    lv = g.constants.copy()
    lv["running"] = True
    exec(g.code, {}, lv)
    out_key = list(g.output_key_to_variables.values())[0]
    assert lv[out_key]["a"].tolist() == ["v1", "v2"]


def test_projection_transform_chunk_udf():
    """transform_chunk UDF adds a column, preserving row count."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30]})
    lf = from_local_df(df)
    res = transform_chunk(lf, lambda d: d.assign(c=d["a"] + d["b"]))
    dag = res.build_graph()
    g = SPECodeGenerator("s1").generate(dag)
    lv = g.constants.copy()
    lv["running"] = True
    exec(g.code, {}, lv)
    out_key = list(g.output_key_to_variables.values())[0]
    result = lv[out_key]
    assert list(result.columns) == ["a", "b", "c"]
    assert result["c"].tolist() == [11, 22, 33]


def test_projection_transform_row_udf():
    """transform(axis=1) UDF produces per-row dict results."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30]})
    lf = from_local_df(df)
    res = transform(
        lf,
        lambda row: {"c": row["a"] + row["b"]},
        axis=1,
        dtypes=pd.Series({"c": wrap_arrow_dtype(pa.int64())}),
    )
    dag = res.build_graph()
    g = SPECodeGenerator("s1").generate(dag)
    lv = g.constants.copy()
    lv["running"] = True
    exec(g.code, {}, lv)
    out_key = list(g.output_key_to_variables.values())[0]
    result = lv[out_key]
    assert result["c"].tolist() == [11, 22, 33]


def _run_spe_apply_row(lf, **apply_kwargs):
    """Build, generate, and exec an apply(axis=1) projection; return the result."""
    res = apply(lf, axis=1, **apply_kwargs)
    assert isinstance(res.op, LiteFrameProjection)
    dag = res.build_graph()
    g = SPECodeGenerator("s1").generate(dag)
    lv = g.constants.copy()
    lv["running"] = True
    exec(g.code, {}, lv)
    out_key = list(g.output_key_to_variables.values())[0]
    return lv[out_key]


@pytest.mark.parametrize(
    "func, dtypes, result_type, expected_cols, check",
    [
        pytest.param(
            lambda row: row["a"] + row["b"],
            pd.Series({"s": wrap_arrow_dtype(pa.int64())}),
            None,
            ["s"],
            lambda r: r["s"].tolist() == [11, 22, 33],
            id="scalar",
        ),
        pytest.param(
            lambda row: {"v": row["a"] * 2},
            pd.Series({"out": object}),
            None,
            ["out"],
            lambda r: [d["v"] for d in r["out"].tolist()] == [2, 4, 6],
            id="dict_default",
        ),
        pytest.param(
            lambda row: {"x": row["a"] * 2, "y": row["b"] + 1},
            pd.Series(
                {"x": wrap_arrow_dtype(pa.int64()), "y": wrap_arrow_dtype(pa.int64())}
            ),
            "expand",
            ["x", "y"],
            lambda r: r["x"].tolist() == [2, 4, 6] and r["y"].tolist() == [11, 21, 31],
            id="expand",
        ),
        pytest.param(
            lambda row: {"v": row["a"]},
            pd.Series({"out": object}),
            "reduce",
            ["out"],
            lambda r: [d["v"] for d in r["out"].tolist()] == [1, 2, 3],
            id="reduce",
        ),
    ],
)
def test_projection_apply_row_result_type(
    func, dtypes, result_type, expected_cols, check
):
    """apply(axis=1) single-output projection result_type matrix."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30]})
    result = _run_spe_apply_row(
        from_local_df(df), func=func, dtypes=dtypes, result_type=result_type
    )
    assert result.columns.tolist() == expected_cols
    assert check(result)


def test_projection_apply_row_broadcast_preserves_float_scalars():
    """apply(axis=1, result_type='broadcast') preserves float scalars on an
    integer input chunk (pandas' native broadcast would truncate 1.5 -> 1)."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30]})
    result = _run_spe_apply_row(
        from_local_df(df),
        func=lambda row: row["a"] * 1.5,
        dtypes=pd.Series(
            {"a": wrap_arrow_dtype(pa.float64()), "b": wrap_arrow_dtype(pa.float64())}
        ),
        result_type="broadcast",
    )
    assert result.columns.tolist() == ["a", "b"]
    assert result["a"].tolist() == [1.5, 3.0, 4.5]
    assert result["b"].tolist() == [1.5, 3.0, 4.5]


def test_projection_apply_row_key_mismatch_nan_filled():
    """apply(axis=1) is non-strict: a row missing a declared dict key is
    NaN-filled rather than rejected (unlike transform_row)."""

    def udf(row):
        if row["a"] < 3:
            return {"x": row["a"], "y": row["b"]}
        return {"x": row["a"]}

    df = pd.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30]})
    result = _run_spe_apply_row(
        from_local_df(df),
        func=udf,
        dtypes=pd.Series(
            {"x": wrap_arrow_dtype(pa.float64()), "y": wrap_arrow_dtype(pa.float64())}
        ),
        result_type="expand",
    )
    assert result.columns.tolist() == ["x", "y"]
    assert result["x"].tolist() == [1.0, 2.0, 3.0]
    assert result["y"].tolist()[:2] == [10.0, 20.0]
    assert pd.isna(result["y"].iloc[2])


@pytest.mark.parametrize(
    "func, result_type, input_frame_type, expected_a, expected_b",
    [
        pytest.param(
            lambda row: [row["a"] * 2, row["b"] * 2],
            "broadcast",
            None,
            [2, 4, 6],
            [20, 40, 60],
            id="list_broadcast",
        ),
        pytest.param(
            lambda row: pd.Series([row["a"] * 2, row["b"] * 2]),
            "broadcast",
            None,
            [2, 4, 6],
            [20, 40, 60],
            id="positional_series_broadcast",
        ),
        pytest.param(
            lambda row: pd.Series({"a": row["a"] * 2, "b": row["b"] * 2}),
            "broadcast",
            None,
            [2, 4, 6],
            [20, 40, 60],
            id="named_series_broadcast",
        ),
        pytest.param(
            lambda row: pl.Series(row.row(0)),
            "broadcast",
            "polars",
            [1, 2, 3],
            [10, 20, 30],
            id="polars_series_coerced",
            marks=pytest.mark.skipif(pl is None, reason="polars not installed"),
        ),
        pytest.param(
            lambda row: row,
            "expand",
            "polars",
            [1, 2, 3],
            [10, 20, 30],
            id="polars_dataframe_coerced_to_dict",
            marks=pytest.mark.skipif(pl is None, reason="polars not installed"),
        ),
    ],
)
def test_projection_apply_row_broadcast_and_polars_coercion(
    func, result_type, input_frame_type, expected_a, expected_b
):
    """result_type='broadcast' result shapes + polars pl.Series/pl.DataFrame
    per-row coercions (mirroring DPE's apply_polars_rowwise).

    Covers: list/tuple positional split, positional/named pd.Series broadcast
    (pandas auto-combines to DataFrame), polars pl.Series -> pd.Series coercion,
    and polars pl.DataFrame -> dict coercion under expand.
    """
    df = pd.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30]})
    kwargs = dict(
        func=func,
        dtypes=pd.Series(
            {"a": wrap_arrow_dtype(pa.int64()), "b": wrap_arrow_dtype(pa.int64())}
        ),
        result_type=result_type,
    )
    if input_frame_type:
        kwargs["input_frame_type"] = input_frame_type
    result = _run_spe_apply_row(from_local_df(df), **kwargs)
    assert result.columns.tolist() == ["a", "b"]
    assert result["a"].tolist() == expected_a
    assert result["b"].tolist() == expected_b


def test_projection_apply_row_broadcast_list_length_mismatch_raises():
    """result_type='broadcast' raises on list length != input column count."""
    df = pd.DataFrame({"a": [1, 2], "b": [10, 20]})
    with pytest.raises(ValueError, match="requires results with 2 elements"):
        _run_spe_apply_row(
            from_local_df(df),
            func=lambda row: [row["a"] * 2],  # 1 element, 2 input columns
            dtypes=pd.Series(
                {"a": wrap_arrow_dtype(pa.int64()), "b": wrap_arrow_dtype(pa.int64())}
            ),
            result_type="broadcast",
        )


def test_projection_transform_chunk_row_count_mismatch():
    """transform_chunk raises when output row count differs from input."""
    df = pd.DataFrame({"a": [1, 2, 3]})
    lf = from_local_df(df)
    res = transform_chunk(lf, lambda d: d.head(1))
    dag = res.build_graph()
    g = SPECodeGenerator("s1").generate(dag)
    lv = g.constants.copy()
    lv["running"] = True
    with pytest.raises(ValueError, match="row count"):
        exec(g.code, {}, lv)


def test_projection_mixed_udf_and_expression(make_liteframe):
    """A single projection op mixing a UDF column and a pure expression."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30]})
    data = make_liteframe(df)
    i64 = wrap_arrow_dtype(pa.int64())
    # One UDF projection (map on 'a') + one pure expression ('b' + 1).
    udf_expr = LiteFrameUDFExpr(
        func=lambda x: x * 10,
        mode="map",
        operands=[LiteFrameColumn(name="a", dtype=i64)],
        output_names=["a"],
        dtypes={"a": i64},
    )
    projections = [
        NamedExpr(name="a", expr=udf_expr, dtype=i64),
        NamedExpr(
            name="b2",
            expr=LiteFrameAdd(
                left=LiteFrameColumn(name="b", dtype=i64),
                right=LiteFrameLiteral(value=1, dtype=i64),
                dtype=i64,
            ),
            dtype=i64,
        ),
    ]
    op = LiteFrameProjection(projections=projections)
    out = op(data)
    result = LiteFrameProjectionAdapter._run_projection(
        df, out.op.projections, set(), False
    )
    assert result["a"].tolist() == [10, 20, 30]
    assert result["b2"].tolist() == [11, 21, 31]


# ---------------------------------------------------------------------------
# Regressions for AI-review comments C1 (output validation) and C7 (map
# args/kwargs forwarding).
# ---------------------------------------------------------------------------


def test_transform_chunk_rejects_wrong_columns():
    """C1: a transform_chunk UDF returning wrong column names raises under
    the default 'raises' policy instead of being silently reindexed to null."""
    df = pd.DataFrame({"a": [1, 2, 3]})
    i64 = wrap_arrow_dtype(pa.int64())
    udf_expr = LiteFrameUDFExpr(
        func=lambda d: d.rename(columns={"a": "wrong"}),
        operands=[LiteFrameColumn(name="a", dtype=i64)],
        output_names=["a"],
        dtypes={"a": i64},
        mode="transform_chunk",
    )
    with pytest.raises(Exception):  # OutputColumnMismatchError
        _run_udf(udf_expr, df)


def test_transform_row_rejects_wrong_dict_key():
    """Second-pass ⑨: a transform_row UDF returning {"wrong": v} with declared
    output "expected" raises under the default 'raises' policy instead of being
    silently reindexed to an all-null "expected" column.  Dict results keep
    their natural labels until validate_and_align_output."""
    df = pd.DataFrame({"a": [1, 2, 3]})
    i64 = wrap_arrow_dtype(pa.int64())
    udf_expr = LiteFrameUDFExpr(
        func=lambda row: {"wrong": row["a"]},
        operands=[LiteFrameColumn(name="a", dtype=i64)],
        output_names=["expected"],
        dtypes={"expected": i64},
        mode="transform_row",
    )
    with pytest.raises(Exception):  # OutputColumnMismatchError
        _run_udf(udf_expr, df)


def test_transform_row_dict_arbitrary_schema():
    """transform(axis=1) may declare an arbitrary output schema via dtypes."""
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    i64 = wrap_arrow_dtype(pa.int64())
    udf_expr = LiteFrameUDFExpr(
        func=lambda row: {"ab": row["a"] + row["b"]},
        operands=[
            LiteFrameColumn(name="a", dtype=i64),
            LiteFrameColumn(name="b", dtype=i64),
        ],
        output_names=["ab"],
        dtypes={"ab": i64},
        mode="transform_row",
    )
    result = _run_udf(udf_expr, df)
    assert list(result.columns) == ["ab"]
    assert result["ab"].tolist() == [4, 6]


def test_transform_axis1_public_arbitrary_schema():
    """transform(axis=1) with two inputs and one output column."""
    lf = LiteFrame({"a": [1, 2], "b": [3, 4]})
    op = lf.transform(
        lambda row: {"ab": row["a"] + row["b"]}, axis=1, dtypes={"ab": int}
    )._data.op
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    out = LiteFrameProjectionAdapter._run_projection(df, op.projections, set(), False)
    assert list(out.columns) == ["ab"]
    assert out["ab"].tolist() == [4, 6]


def test_map_forwards_args_and_kwargs():
    """C7: map forwards args/kwargs to the element callable (SPE execution)."""
    df = pd.DataFrame({"a": [1, 2, 3]})
    i64 = wrap_arrow_dtype(pa.int64())
    udf_expr = LiteFrameUDFExpr(
        func=lambda x, n, *, scale: (x + n) * scale,
        operands=[LiteFrameColumn(name="a", dtype=i64)],
        output_names=["a"],
        dtypes={"a": i64},
        mode="map",
        args=(1,),
        kwargs={"scale": 2},
    )
    result = _run_udf(udf_expr, df)
    assert result["a"].tolist() == [4, 6, 8]


# ---------------------------------------------------------------------------
# Callable-class protocol: close() is called once at chunk end
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mode,output_names,expected_col,expected_vals,expected_calls",
    [
        ("transform_row", ["x"], "x", [2, 4, 6], 3),
        ("map", ["a"], "a", [2, 4, 6], 3),
        ("transform_chunk", ["a", "a2"], "a2", [2, 4, 6], 1),
    ],
)
def test_callable_class_close_called_once(
    mode, output_names, expected_col, expected_vals, expected_calls
):
    """Expression UDF callable class: instantiate once, close() once at chunk end.

    Regression: SPE's _run_udf passed the class directly to pandas, which
    instantiated it per element/row.  Now make_callable_wrapper instantiates
    once, __call__ is invoked per element/row/chunk, and close() runs once.
    """
    i64 = wrap_arrow_dtype(pa.int64())
    state = {"instances": 0, "calls": 0, "closed": 0}

    if mode == "transform_row":

        class _UDF:
            def __init__(self):
                state["instances"] += 1

            def __call__(self, row):
                state["calls"] += 1
                return {"x": row["a"] * 2}

            def close(self):
                state["closed"] += 1

    elif mode == "map":

        class _UDF:
            def __init__(self):
                state["instances"] += 1

            def __call__(self, x):
                state["calls"] += 1
                return x * 2

            def close(self):
                state["closed"] += 1

    else:  # transform_chunk

        class _UDF:
            def __init__(self):
                state["instances"] += 1

            def __call__(self, df, end=False):
                state["calls"] += 1
                return df.assign(a2=df["a"] * 2)[["a", "a2"]]

            def close(self):
                state["closed"] += 1

    udf_expr = LiteFrameUDFExpr(
        func=_UDF,
        mode=mode,
        operands=[LiteFrameColumn(name="a", dtype=i64)],
        output_names=output_names,
        dtypes={n: i64 for n in output_names},
        input_frame_type="pandas",
        check_output_dtypes="raises",
    )
    df = pd.DataFrame({"a": [1, 2, 3]})
    result = _run_udf(udf_expr, df)
    assert result[expected_col].tolist() == expected_vals
    assert state["instances"] == 1
    assert state["calls"] == expected_calls
    assert state["closed"] == 1


# ---------------------------------------------------------------------------
# Accessor projections (str / datetime / timedelta)
# ---------------------------------------------------------------------------


def _run_accessor_projection(lf_result, df):
    """Run _run_projection for an accessor result and return the DataFrame."""
    op = lf_result._data.op
    return LiteFrameProjectionAdapter._run_projection(
        df, op.projections, set(), False, {}
    )


@pytest.mark.parametrize(
    "method,args,kwargs",
    [
        ("upper", (), {}),
        ("lower", (), {}),
        ("strip", (), {}),
        ("replace", ("l", "L"), {}),
        ("zfill", (8,), {}),
        ("count", ("l",), {}),
        ("isalpha", (), {}),
    ],
)
def test_projection_str_single_column(method, args, kwargs):
    """Single-column str accessor methods project correctly via SPE."""
    df = pd.DataFrame({"s": ["hello", "world", "foo bar"]})
    lf = from_local_df(df)
    result = _run_accessor_projection(getattr(lf.str, method)(*args, **kwargs), df)
    expected = getattr(df["s"].str, method)(*args, **kwargs)
    pd.testing.assert_series_equal(result[result.columns[0]], expected)


@pytest.mark.parametrize(
    "method,args,kwargs,n_cols",
    [
        ("partition", ("_",), {}, 3),
        ("rpartition", ("_",), {}, 3),
        ("extract", (r"([a-z])",), {}, 1),
        ("split", ("_",), {"expand": True, "n": 1}, 2),
    ],
)
def test_projection_str_multi_column(method, args, kwargs, n_cols):
    """Multi-column str accessor methods project correctly via SPE."""
    df = pd.DataFrame({"s": ["a_b_c", "d_e", "f"]})
    lf = from_local_df(df)
    result = _run_accessor_projection(getattr(lf.str, method)(*args, **kwargs), df)
    expected = getattr(df["s"].str, method)(*args, **kwargs)
    expected.columns = [str(i) for i in range(n_cols)]
    pd.testing.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "attr,is_method,args",
    [
        ("year", False, ()),
        ("is_leap_year", False, ()),
        ("strftime", True, ("%Y-%m-%d",)),
        ("normalize", True, ()),
        ("round", True, ("1h",)),
    ],
)
def test_projection_datetime_accessor(attr, is_method, args):
    """Datetime accessor methods project correctly via SPE."""
    df = pd.DataFrame({"dt": pd.date_range("2020-01-01", periods=3, freq="D")})
    lf = from_local_df(df)
    if is_method:
        lf_result = getattr(lf.dt, attr)(*args)
        expected = getattr(df["dt"].dt, attr)(*args)
    else:
        lf_result = getattr(lf.dt, attr)
        expected = getattr(df["dt"].dt, attr)
    result = _run_accessor_projection(lf_result, df)
    pd.testing.assert_series_equal(result[result.columns[0]], expected)


@pytest.mark.parametrize(
    "attr,is_method",
    [
        ("days", False),
        ("seconds", False),
        ("total_seconds", True),
    ],
)
def test_projection_timedelta_accessor(attr, is_method):
    """Timedelta accessor methods project correctly via SPE."""
    df = pd.DataFrame(
        {"td": pd.to_timedelta([93784005006000, -129600000000000, 7], unit="ns")}
    )
    lf = from_local_df(df)
    if is_method:
        lf_result = getattr(lf.dt, attr)()
        expected = getattr(df["td"].dt, attr)()
    else:
        lf_result = getattr(lf.dt, attr)
        expected = getattr(df["td"].dt, attr)
    result = _run_accessor_projection(lf_result, df)
    pd.testing.assert_series_equal(result[result.columns[0]], expected)


@pytest.mark.parametrize(
    "method,expected_cols",
    [
        ("isocalendar", ["year", "week", "day"]),
        (
            "components",
            [
                "days",
                "hours",
                "minutes",
                "seconds",
                "milliseconds",
                "microseconds",
                "nanoseconds",
            ],
        ),
    ],
)
def test_projection_datetime_multi_column(method, expected_cols):
    """Multi-column datetime accessor methods project correctly via SPE."""
    if method == "isocalendar":
        df = pd.DataFrame({"dt": pd.date_range("2020-01-01", periods=3, freq="D")})
    else:
        df = pd.DataFrame(
            {"td": pd.to_timedelta([93784005006000, -129600000000000, 7], unit="ns")}
        )
    lf = from_local_df(df)
    # isocalendar is a method; components is a property (pandas semantics).
    attr = getattr(lf.dt, method)
    lf_result = attr() if callable(attr) else attr
    raw = getattr(df[df.columns[0]].dt, method)
    expected = raw() if callable(raw) else raw
    result = _run_accessor_projection(lf_result, df)
    assert list(result.columns) == expected_cols
    pd.testing.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    ("data", "expected_cols"),
    [
        pytest.param(["a_b_c", "d_e", "f"], ["0", "1", "2"], id="local_matches_global"),
        pytest.param(["a_b", "c_d"], ["0", "1", "2"], id="pads_to_global_max"),
        pytest.param(["a_b_c_d", "e_f_g"], ["0", "1", "2"], id="truncates_excess"),
    ],
)
def test_projection_str_dynamic_split(data, expected_cols):
    """Dynamic split(expand=True, n<=0) produces N string columns discovered
    at execution time.  Per-chunk results are padded (shorter) or truncated
    (longer) to the global column count from the dtypes Series, so every
    chunk produces the same schema.

    The global max is simulated as 3 columns via the extra dtypes input;
    ``pads_to_global_max`` feeds 2-field rows (→ null third col), and
    ``truncates_excess`` feeds 4-field rows (→ fourth field dropped).
    """
    df = pd.DataFrame({"s": data})
    lf = from_local_df(df)
    lf_result = lf.str.split("_", expand=True, n=-1)
    op = lf_result._data.op
    assert lf_result.is_schema_dynamic
    assert len(op.inputs) == 2

    # The extra input resolves to a pd.Series of dtypes; simulate a 3-col
    # global schema.
    dtypes_series = pd.Series({str(i): "string" for i in range(len(expected_cols))})

    result = LiteFrameProjectionAdapter._run_projection(
        df, op.projections, set(), False, {}, [dtypes_series]
    )
    assert list(result.columns) == expected_cols
    # Build expected: pandas split, then pad shorter chunks / truncate excess
    # to the global column count (the simulated dtypes Series width).
    expected = df["s"].str.split("_", expand=True, n=-1)
    for i in range(len(expected.columns), len(expected_cols)):
        expected[i] = None
    expected = expected.iloc[:, : len(expected_cols)]
    expected.columns = expected_cols
    pd.testing.assert_frame_equal(result, expected)
