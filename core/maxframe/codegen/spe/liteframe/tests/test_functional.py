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

from maxframe.codegen.spe.core import SPECodeContext
from maxframe.codegen.spe.liteframe.functional import LiteFrameFunctionalAdapter
from maxframe.liteframe.functional.flat_apply import flat_apply
from maxframe.utils import is_arrow_dtype_supported, wrap_arrow_dtype


def _multi_output_dtypes():
    """Two-output dtype spec: columns x and y, both int64 arrow."""
    return [
        pd.Series({"x": wrap_arrow_dtype(pa.int64())}),
        pd.Series({"y": wrap_arrow_dtype(pa.int64())}),
    ]


def test_functional_adapter_code_generation(make_liteframe):
    """LiteFrameFunctionalAdapter should generate _run_functional call."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    data = make_liteframe(df)
    # Use the LiteFrame API to construct the operator with proper inference
    out = data.apply(
        lambda col: col * 2,
        axis=0,
        dtypes=pd.Series(
            {"a": wrap_arrow_dtype(pa.int64()), "b": wrap_arrow_dtype(pa.int64())}
        ),
    )
    op = out.op
    adapter = LiteFrameFunctionalAdapter()
    context = SPECodeContext()
    result = adapter.generate_code(op, context)
    assert len(result) == 1
    assert "LiteFrameFunctionalAdapter._run_functional" in result[0]


# ---------------------------------------------------------------------------
# _run_functional — apply / transform / apply_chunk (plain, no groupby)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "df_data, func, method, axis, args, check",
    [
        pytest.param(
            {"a": [1, 2, 3], "b": [4, 5, 6]},
            lambda col: col * 2,
            "apply",
            0,
            (),
            lambda r: (list(r["a"]) == [2, 4, 6] and list(r["b"]) == [8, 10, 12]),
            id="apply-axis0",
        ),
        pytest.param(
            {"a": [1, 2, 3], "b": [10, 20, 30]},
            lambda row: row["a"] + row["b"],
            "apply",
            1,
            (),
            lambda r: list(r.iloc[:, 0]) == [11, 22, 33],
            id="apply-axis1",
        ),
        pytest.param(
            {"a": [1, 2, 3], "b": [4, 5, 6]},
            lambda d: d * 10,
            "apply_chunk",
            0,
            (),
            lambda r: (list(r["a"]) == [10, 20, 30] and list(r["b"]) == [40, 50, 60]),
            id="apply-chunk",
        ),
        pytest.param(
            {"a": [1, 2, 3], "b": [4, 5, 6]},
            lambda col: col + 1,
            "transform",
            0,
            (),
            lambda r: (list(r["a"]) == [2, 3, 4] and list(r["b"]) == [5, 6, 7]),
            id="transform",
        ),
        pytest.param(
            {"a": [1, 2, 3]},
            lambda col, factor: col * factor,
            "apply",
            0,
            (10,),
            lambda r: list(r["a"]) == [10, 20, 30],
            id="apply-with-args",
        ),
    ],
)
def test_functional_run_apply(df_data, func, method, axis, args, check):
    """_run_functional drives apply / apply_chunk / transform with optional args."""
    df = pd.DataFrame(df_data)
    result = LiteFrameFunctionalAdapter._run_functional(
        df, func, method, axis, args, None
    )
    assert check(result)


# ---------------------------------------------------------------------------
# flat_apply — row-expanding UDF (added with the flat_apply operator)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "func, expected",
    [
        pytest.param(
            lambda row: [row["a"], row["a"] * 2],
            [1, 2, 2, 4, 3, 6],
            id="list",
        ),
        pytest.param(
            lambda row: (row["a"] * i for i in range(2)),
            [0, 1, 0, 2],
            id="generator",
        ),
    ],
)
def test_functional_run_flat_apply_expand(func, expected):
    """flat_apply expands each row to multiple rows via a list/generator."""
    df = pd.DataFrame({"a": [1, 2, 3]} if len(expected) == 6 else {"a": [1, 2]})
    result = LiteFrameFunctionalAdapter._run_functional(
        df, func, "flat_apply", 0, (), None
    )
    assert result.iloc[:, 0].tolist() == expected


# ---------------------------------------------------------------------------
# groupby_params — groupby apply / transform (added with groupby UDF APIs)
# ---------------------------------------------------------------------------


_GROUPBY_PARAMS = {"by": "k", "dropna": True}


@pytest.mark.parametrize(
    "df_data, func, method, check",
    [
        pytest.param(
            {"k": ["a", "a", "b"], "v": [1, 2, 3]},
            lambda g: g.sum(numeric_only=True),
            "apply",
            lambda r: r["v"].tolist() == [3, 3],
            id="groupby-apply",
        ),
        pytest.param(
            {"k": ["a", "a", "b"], "v": [1, 2, 3]},
            lambda g: g.assign(v=g["v"] * 10),
            "apply_chunk",
            lambda r: r["v"].tolist() == [10, 20, 30],
            id="groupby-apply-chunk",
        ),
        pytest.param(
            {"k": ["a", "a", "b", "b"], "v": [1, 2, 3, 4]},
            lambda x: x * 10,
            "transform",
            lambda r: (
                len(r) == 4
                and list(r.columns) == ["v"]
                and r["v"].tolist() == [10, 20, 30, 40]
            ),
            id="groupby-transform",
        ),
        pytest.param(
            {"k": ["a", "b"], "v": [1, 2]},
            lambda row: [row["v"], row["v"] * 10],
            "flat_apply",
            lambda r: (
                len(r) == 4
                and list(r.columns) == ["k", "0"]
                and r.iloc[:, 1].tolist() == [1, 10, 2, 20]
            ),
            id="groupby-flat-apply",
        ),
    ],
)
def test_functional_run_groupby(df_data, func, method, check):
    """groupby apply / apply_chunk / transform / flat_apply via groupby_params."""
    df = pd.DataFrame(df_data)
    result = LiteFrameFunctionalAdapter._run_functional(
        df, func, method, 0, (), None, groupby_params=_GROUPBY_PARAMS
    )
    assert check(result)


# ---------------------------------------------------------------------------
# flat_apply — DPE parity: multi-output, dtype alignment, polars, error contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "df_data, func, groupby_params, check_x, check_y",
    [
        pytest.param(
            {"a": [1, 2, 3]},
            lambda row: ([{"x": row["a"]}], [{"y": row["a"] * 10}]),
            None,
            [1, 2, 3],
            [10, 20, 30],
            id="flat-apply-multi-output",
        ),
        pytest.param(
            {"k": ["a", "b"], "v": [1, 2]},
            lambda row: ([{"x": row["v"]}], [{"y": row["v"] * 10}]),
            _GROUPBY_PARAMS,
            [1, 2],
            [10, 20],
            id="groupby-flat-apply-multi-output",
        ),
    ],
)
def test_functional_run_flat_apply_multiple_outputs(
    df_data, func, groupby_params, check_x, check_y
):
    """flat_apply multi-output returns a tuple of frames, one per declared output."""
    df = pd.DataFrame(df_data)
    dtypes = _multi_output_dtypes()
    if groupby_params is not None:
        # groupby flat_apply prepends the grouping keys to every output
        # (group_keys=True default), so the declared per-output schemas must
        # carry them too -- mirroring the compile-time schema.
        k_dtype = wrap_arrow_dtype(pa.string())
        dtypes = [pd.concat([pd.Series({"k": k_dtype}), d]) for d in dtypes]
    result = LiteFrameFunctionalAdapter._run_functional(
        df,
        func,
        "flat_apply",
        0,
        (),
        None,
        groupby_params=groupby_params,
        output_dtypes=dtypes,
        n_output=2,
    )
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert result[0]["x"].tolist() == check_x
    assert result[1]["y"].tolist() == check_y


@pytest.mark.parametrize(
    "func,output_dtypes,x_vals,b_vals,x_dtype",
    [
        pytest.param(
            lambda row: [{"x": float(row["a"])}, {"x": float(row["a"] * 2)}],
            [
                pd.Series(
                    {
                        "x": wrap_arrow_dtype(pa.int64()),
                        "b": wrap_arrow_dtype(pa.int64()),
                    }
                )
            ],
            [1, 2, 2, 4],
            [10, 10, 20, 20],
            "int64[pyarrow]",
            id="typed_cast",
        ),
        pytest.param(
            lambda row: [{"x": row["a"]}],
            [None],
            [1, 2],
            [10, 20],
            None,
            id="skip_infer",
        ),
    ],
)
def test_functional_run_flat_apply_subset_column_order(
    func, output_dtypes, x_vals, b_vals, x_dtype
):
    """func-out cols inserted at first subset position; broadcast col b after.

    Covers typed path (dtype casting) and skip_infer (func_out_cols is None).
    Regression: skip_infer used to append func-out after broadcast ([b,x]).
    """
    df = pd.DataFrame({"a": [1, 2], "b": [10, 20]})
    result = LiteFrameFunctionalAdapter._run_functional(
        df,
        func,
        "flat_apply",
        0,
        (),
        None,
        subset=["a"],
        output_dtypes=output_dtypes,
        n_output=1,
    )
    assert list(result.columns) == ["x", "b"]
    assert result["x"].tolist() == x_vals
    assert result["b"].tolist() == b_vals
    if x_dtype:
        assert str(result["x"].dtype) == x_dtype


@pytest.mark.parametrize(
    "groupby_params, subset, expected",
    [
        # group_keys contract on subset=[] (anchorless): keys prepended / omitted.
        pytest.param(
            {"by": "k", "group_keys": True},
            [],
            ["k", "a", "b", "x"],
            id="subset_empty_gk_true",
        ),
        pytest.param(
            {"by": "k", "group_keys": False},
            [],
            ["a", "b", "x"],
            id="subset_empty_gk_false",
        ),
        # selection including the key: key still governed by group_keys, not broadcast.
        pytest.param(
            {"by": "k", "group_keys": True, "selection": ["a", "k"]},
            ["a"],
            ["k", "x"],
            id="selection_includes_key_gk_true",
        ),
        pytest.param(
            {"by": "k", "group_keys": False, "selection": ["a", "k"]},
            ["a"],
            ["x"],
            id="selection_includes_key_gk_false",
        ),
        # key as first subset anchor: func-out at key's position; key omitted (gk=False) / prepended (gk=True).
        pytest.param(
            {"by": "k", "group_keys": False, "selection": ["k", "a", "b"]},
            ["k"],
            ["x", "a", "b"],
            id="key_is_subset_anchor_gk_false",
        ),
        pytest.param(
            {"by": "k", "group_keys": True, "selection": ["k", "a", "b"]},
            ["k"],
            ["k", "x", "a", "b"],
            id="key_is_subset_anchor_gk_true",
        ),
    ],
)
def test_functional_run_flat_apply_groupby_column_order(
    groupby_params, subset, expected
):
    """groupby flat_apply column order: group keys follow the ``group_keys``
    contract (prepended when True, omitted when False), never the broadcast
    path -- even when an explicit selection includes the key.

    Regressions: subset=[] left key as broadcast; selection including key
    treated it as broadcast.
    """
    df = pd.DataFrame({"k": ["g"], "a": [1], "b": [2]})
    result = LiteFrameFunctionalAdapter._run_functional(
        df,
        lambda row: [{"x": 1}],
        "flat_apply",
        0,
        (),
        None,
        groupby_params=groupby_params,
        subset=subset,
        output_dtypes=None,
        n_output=1,
    )
    assert list(result.columns) == expected


@pytest.mark.parametrize(
    "df, udf, groupby_params, subset, output_dtypes, expected_cols, check_dtypes",
    [
        pytest.param(
            pd.DataFrame(
                {
                    "k": pd.Series([], dtype=object),
                    "a": pd.Series([], dtype=int),
                    "b": pd.Series([], dtype=int),
                }
            ),
            lambda r: [{"x": 1}],
            {"by": "k", "group_keys": False},
            [],
            None,
            ["a", "b"],
            False,
            id="skip_infer_retains_broadcast",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "k": pd.Series([], dtype="string"),
                    "a": pd.Series([], dtype="int64"),
                    "b": pd.Series([], dtype="int64"),
                    "z": pd.Series([], dtype="int64"),
                }
            ),
            lambda r: [{"x": 1}],
            {"by": "k", "group_keys": True, "selection": ["k", "a", "b"]},
            ["k"],
            [
                pd.Series(
                    {
                        "k": wrap_arrow_dtype(pa.string()),
                        "x": wrap_arrow_dtype(pa.int64()),
                        "a": wrap_arrow_dtype(pa.int64()),
                        "b": wrap_arrow_dtype(pa.int64()),
                    }
                )
            ],
            ["k", "x", "a", "b"],
            True,
            id="typed_preserves_schema",
        ),
        pytest.param(
            pd.DataFrame({"k": pd.Series([], dtype="string")}),
            lambda r: [{"k": 1}],
            {"by": "k", "group_keys": False, "selection": ["k"]},
            ["k"],
            [pd.Series({"k": int})],
            ["k"],
            False,
            id="func_out_matches_key",
        ),
    ],
)
def test_functional_run_flat_apply_groupby_empty(
    df, udf, groupby_params, subset, output_dtypes, expected_cols, check_dtypes
):
    """Empty groupby flat_apply partitions preserve column order/dtypes."""
    result = LiteFrameFunctionalAdapter._run_functional(
        df,
        udf,
        "flat_apply",
        0,
        (),
        None,
        groupby_params=groupby_params,
        subset=subset,
        output_dtypes=output_dtypes,
        n_output=1,
        check_output_dtypes="ignore",
    )
    assert list(result.columns) == expected_cols
    assert len(result) == 0
    if check_dtypes and output_dtypes is not None:
        od = output_dtypes[0]
        for c in expected_cols:
            if c in od.index:
                assert str(result[c].dtype) == str(od[c])


def test_functional_run_flat_apply_polars_input():
    """flat_apply passes a single-row polars DataFrame in polars mode."""
    if not is_arrow_dtype_supported():
        pytest.skip("polars UDF result conversion needs ArrowDtype (pyarrow>=8)")
    df = pd.DataFrame({"a": [1, 2]})
    out_dtypes = pd.Series({"x": wrap_arrow_dtype(pa.int64())})
    result = LiteFrameFunctionalAdapter._run_functional(
        df,
        lambda row: [{"x": int(row["a"][0])}],
        "flat_apply",
        0,
        (),
        None,
        output_dtypes=[out_dtypes],
        input_frame_type="polars",
        n_output=1,
    )
    assert result["x"].tolist() == [1, 2]


@pytest.mark.parametrize(
    "func, match_msg, out_dtypes, n_output",
    [
        pytest.param(
            lambda row: (row["a"], row["a"] * 2),
            "top-level tuple",
            [pd.Series({"x": wrap_arrow_dtype(pa.int64())})],
            1,
            id="rejects-top-level-tuple",
        ),
        pytest.param(
            lambda row: ([{"x": row["a"]}],),  # only 1 of 2 declared
            "2 output",
            _multi_output_dtypes(),
            2,
            id="multi-output-arity-error",
        ),
    ],
)
def test_functional_run_flat_apply_rejects(func, match_msg, out_dtypes, n_output):
    """flat_apply rejects malformed UDF returns (top-level tuple / wrong arity)."""
    df = pd.DataFrame({"a": [1]})
    with pytest.raises(ValueError, match=match_msg):
        LiteFrameFunctionalAdapter._run_functional(
            df,
            func,
            "flat_apply",
            0,
            (),
            None,
            output_dtypes=out_dtypes,
            n_output=n_output,
        )


def test_flat_apply_multi_output_codegen(make_liteframe):
    """generate_code emits tuple-unpacking for multi-output flat_apply."""
    lf = make_liteframe(pd.DataFrame({"a": [1, 2, 3]}))
    results = flat_apply(
        lf,
        lambda row: ([{"x": row["a"]}], [{"y": row["a"] * 10}]),
        dtypes=[{"x": int}, {"y": int}],
    )
    op = results[0].op
    adapter = LiteFrameFunctionalAdapter()
    context = SPECodeContext()
    code = adapter.generate_code(op, context)
    assert len(code) == 1
    # Two output variables unpacked from a single _run_functional call.
    assert " = " in code[0]
    assert ", " in code[0].split(" = ")[0]
    assert "LiteFrameFunctionalAdapter._run_functional" in code[0]
    assert "n_output=2" in code[0]


# ---------------------------------------------------------------------------
# Regressions for AI-review comments C2/C8/C9/C10/C11/C16.
# ---------------------------------------------------------------------------


def test_apply_chunk_rejects_wrong_columns():
    """C2: apply_chunk with a wrong-column UDF raises under the default
    'raises' policy instead of silently accepting the mismatched schema."""
    df = pd.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(Exception):  # OutputColumnMismatchError
        LiteFrameFunctionalAdapter._run_functional(
            df,
            lambda d: d.rename(columns={"a": "wrong"}),
            "apply_chunk",
            0,
            (),
            None,
            output_dtypes=[pd.Series({"a": "int64"})],
            check_output_dtypes="raises",
        )


def test_groupby_apply_selection_columns():
    """C8: groupby('g')[['v']].apply passes only column v to the UDF; an unset
    selection excludes the key; an explicit [] passes zero columns."""
    df = pd.DataFrame({"g": ["a", "a", "b"], "v": [1, 2, 3], "w": [4, 5, 6]})

    # selection=['v'] -> UDF sees only v
    seen = []
    LiteFrameFunctionalAdapter._run_functional(
        df,
        lambda d: seen.append(list(d.columns)) or d,
        "apply",
        0,
        (),
        None,
        groupby_params={"by": "g", "dropna": True, "selection": ["v"]},
    )
    assert all(cols == ["v"] for cols in seen)

    # unset selection -> UDF sees non-key columns (v, w)
    seen = []
    LiteFrameFunctionalAdapter._run_functional(
        df,
        lambda d: seen.append(list(d.columns)) or d,
        "apply",
        0,
        (),
        None,
        groupby_params={"by": "g", "dropna": True},
    )
    assert all(cols == ["v", "w"] for cols in seen)

    # explicit [] -> UDF sees zero columns
    seen = []
    LiteFrameFunctionalAdapter._run_functional(
        df,
        lambda d: seen.append(list(d.columns)) or d,
        "apply",
        0,
        (),
        None,
        groupby_params={"by": "g", "dropna": True, "selection": []},
    )
    assert all(cols == [] for cols in seen)


def test_groupby_transform_selection_columns():
    """groupby('g')[['v']].transform transforms only v, not all non-key cols.

    Mirrors the apply branch's use of ``_resolve_groupby_selection`` and the
    DPE ``GroupByTransformRunner``: an explicit selection restricts the
    transformed columns; an unset selection transforms all non-key columns.
    """
    df = pd.DataFrame({"g": ["a", "a", "b"], "v": [1, 2, 3], "w": [4, 5, 6]})

    # selection=['v'] -> only v is transformed; w is absent from the result
    result = LiteFrameFunctionalAdapter._run_functional(
        df,
        lambda x: x * 10,
        "transform",
        0,
        (),
        None,
        groupby_params={"by": "g", "dropna": True, "selection": ["v"]},
    )
    assert list(result.columns) == ["v"]
    assert result["v"].tolist() == [10, 20, 30]

    # unset selection -> all non-key columns transformed
    result = LiteFrameFunctionalAdapter._run_functional(
        df,
        lambda x: x * 10,
        "transform",
        0,
        (),
        None,
        groupby_params={"by": "g", "dropna": True},
    )
    assert list(result.columns) == ["v", "w"]
    assert result["v"].tolist() == [10, 20, 30]
    assert result["w"].tolist() == [40, 50, 60]


def test_groupby_transform_series_per_column_contract(make_liteframe):
    """groupby.transform passes a per-column Series, not a whole-group DataFrame."""
    from maxframe.liteframe import LiteFrame

    lf = LiteFrame({"key": ["a", "a", "b", "b"], "val": [1, 2, 3, 4]})
    op = (
        lf.groupby("key")
        .transform(lambda g: g - g.mean(), dtypes={"val": float})
        ._data.op
    )
    df = pd.DataFrame({"key": ["a", "a", "b", "b"], "val": [1, 2, 3, 4]})
    result = LiteFrameFunctionalAdapter._run_functional(
        df,
        op.func,
        op.mode,
        op.axis,
        op.args,
        op.kwargs,
        groupby_params=op.groupby_params,
    )
    assert list(result.columns) == ["val"]
    assert result["val"].tolist() == [-0.5, 0.5, -0.5, 0.5]


def test_groupby_apply_sort_false_preserves_encounter_order():
    """C9: groupby('g', sort=False).apply preserves group-encounter order
    instead of inheriting pandas' sort=True default."""
    df = pd.DataFrame({"g": ["b", "a", "b", "a"], "v": [1, 2, 3, 4]})
    result = LiteFrameFunctionalAdapter._run_functional(
        df,
        lambda d: int(d["v"].sum()),
        "apply",
        0,
        (),
        None,
        groupby_params={"by": "g", "dropna": True, "sort": False},
    )
    # Group order b,a preserved (sort=False) -> b's sum (1+3=4) comes first,
    # then a's sum (2+4=6).  Under sort=True the order would be a,b -> [6,4].
    # The grouping key is prepended (groupby.apply reset_index semantics).
    assert result["g"].tolist() == ["b", "a"]
    assert result.iloc[:, 1].tolist() == [4, 6]


@pytest.mark.parametrize(
    "func, method, group_keys, expected_cols",
    [
        pytest.param(
            lambda g: g[["v"]] * 2,
            "apply",
            True,
            ["k", "v"],
            id="apply-keys-on",
        ),
        pytest.param(
            lambda g: g[["v"]] * 2,
            "apply",
            False,
            ["v"],
            id="apply-keys-off",
        ),
        pytest.param(
            lambda g: g.assign(v=g["v"] * 10),
            "apply_chunk",
            True,
            ["k", "v"],
            id="apply-chunk-keys-on",
        ),
        pytest.param(
            lambda g: g.assign(v=g["v"] * 10),
            "apply_chunk",
            False,
            ["v"],
            id="apply-chunk-keys-off",
        ),
        pytest.param(
            lambda row: [{"v": row["v"]}, {"v": row["v"] * 10}],
            "flat_apply",
            True,
            ["k", "v"],
            id="flat-apply-keys-on",
        ),
        pytest.param(
            lambda row: [{"v": row["v"]}, {"v": row["v"] * 10}],
            "flat_apply",
            False,
            ["v"],
            id="flat-apply-keys-off",
        ),
    ],
)
def test_groupby_apply_group_keys_gates_prepend(
    func, method, group_keys, expected_cols
):
    """group_keys gates whether the runtime prepends the grouping keys.

    Applies to the whole apply family (apply / apply_chunk / flat_apply):
    the default (True) prepends them (pandas groupby.apply reset_index
    shape); False omits them entirely, matching the declared schema and
    pandas ``group_keys=False``.
    """
    df = pd.DataFrame({"k": ["a", "a", "b"], "v": [1, 2, 3]})
    result = LiteFrameFunctionalAdapter._run_functional(
        df,
        func,
        method,
        0,
        (),
        None,
        groupby_params={
            "by": "k",
            "dropna": True,
            "sort": True,
            "group_keys": group_keys,
        },
    )
    assert list(result.columns) == expected_cols
    if method == "apply" and not group_keys:
        # group_keys=False output equals pandas group_keys=False apply
        p = df.groupby("k", group_keys=False).apply(func)
        assert result.reset_index(drop=True).equals(p.reset_index(drop=True))


def test_groupby_apply_scalar_result():
    """C10: a scalar-returning groupby apply UDF is normalized to a frame so
    pd.concat does not raise TypeError."""
    df = pd.DataFrame({"g": ["a", "a", "b"], "v": [1, 2, 3]})
    result = LiteFrameFunctionalAdapter._run_functional(
        df,
        lambda group: int(group["v"].sum()),
        "apply",
        0,
        (),
        None,
        groupby_params={"by": "g", "dropna": True},
    )
    # One row per group: the grouping key is prepended (groupby.apply
    # reset_index semantics), with the scalar in the following column.
    assert len(result) == 2
    assert result["g"].tolist() == ["a", "b"]
    assert result.iloc[:, 1].tolist() == [3, 3]


def test_groupby_apply_dict_result():
    """Second-pass ⑦: a dict-returning groupby apply UDF is normalized to a
    frame with declared columns.  Covers the dict path of
    _normalize_group_result independently of the scalar case."""
    df = pd.DataFrame({"g": ["a", "a", "b"], "v": [1, 2, 3]})
    result = LiteFrameFunctionalAdapter._run_functional(
        df,
        lambda group: {"total": int(group["v"].sum()), "n": len(group)},
        "apply",
        0,
        (),
        None,
        groupby_params={"by": "g", "dropna": True},
    )
    # The grouping key is prepended (groupby.apply reset_index semantics).
    assert list(result.columns) == ["g", "total", "n"]
    assert len(result) == 2
    # group a: total=3, n=2; group b: total=3, n=1
    assert result["total"].tolist() == [3, 3]
    assert result["n"].tolist() == [2, 1]


def test_groupby_apply_empty_returns_declared_schema():
    """An empty groupby apply returns a zero-row frame with the declared
    output columns, not the input columns.

    Regression: an empty frame grouped by 'k' with apply declared as
    ['k', 'sum'] previously returned df.iloc[:0] (columns ['k', 'v']),
    causing validation to report missing 'sum' and unexpected 'v'.
    """
    df = pd.DataFrame({"k": [], "v": []})
    out_dtypes = pd.Series(
        {"k": wrap_arrow_dtype(pa.string()), "sum": wrap_arrow_dtype(pa.int64())}
    )
    result = LiteFrameFunctionalAdapter._run_functional(
        df,
        lambda g: g[["v"]].assign(sum=g["v"].sum()),
        "apply",
        0,
        (),
        None,
        groupby_params={"by": "k", "dropna": True, "selection": ["v"]},
        output_dtypes=[out_dtypes],
        n_output=1,
    )
    assert len(result) == 0
    assert list(result.columns) == ["k", "sum"]


@pytest.mark.parametrize(
    "udf, n_output, check_output_dtypes, expect_error",
    [
        pytest.param(
            lambda g: (g[["v"]], g[["v"]] * 2),
            2,
            "ignore",
            None,
            id="multi_output",
        ),
        pytest.param(
            lambda g: g[["v"]],
            2,
            "ignore",
            "DataFrame with 0 .* but 2",
            id="arity_non_tuple",
        ),
        pytest.param(
            lambda g: (g[["v"]],),
            2,
            "ignore",
            "1 .* but 2",
            id="arity_wrong_length",
        ),
    ],
)
def test_groupby_apply_multi_output(udf, n_output, check_output_dtypes, expect_error):
    """Groupby apply multi-output: tuple results materialized; arity validated.

    - multi_output: both outputs collected, concatenated, returned as tuple.
    - arity_non_tuple: non-tuple for n_output=2 raises ValueError.
    - arity_wrong_length: 1-tuple for n_output=2 raises ValueError.
    """
    df = pd.DataFrame({"k": ["a", "a", "b"], "v": [1, 2, 3]})
    dtypes = _multi_output_dtypes()
    if expect_error:
        with pytest.raises(ValueError, match=expect_error):
            LiteFrameFunctionalAdapter._run_functional(
                df,
                udf,
                "apply",
                0,
                (),
                None,
                groupby_params={"by": "k", "dropna": True, "selection": ["v"]},
                output_dtypes=dtypes,
                n_output=n_output,
                check_output_dtypes=check_output_dtypes,
            )
        return
    result = LiteFrameFunctionalAdapter._run_functional(
        df,
        udf,
        "apply",
        0,
        (),
        None,
        groupby_params={"by": "k", "dropna": True, "selection": ["v"]},
        output_dtypes=dtypes,
        n_output=n_output,
        check_output_dtypes=check_output_dtypes,
    )
    assert isinstance(result, tuple) and len(result) == 2
    assert result[0]["v"].tolist() == [1, 2, 3]
    assert result[1]["v"].tolist() == [2, 4, 6]


def test_apply_axis0_all_scalar_result():
    """C16: apply(lambda c: c.sum(), axis=0) returns a one-row frame when each
    column produces a scalar, instead of raising 'all scalar values'."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    result = LiteFrameFunctionalAdapter._run_functional(
        df,
        lambda column: column.sum(),
        "apply",
        0,
        (),
        None,
    )
    assert len(result) == 1
    assert result["a"].iloc[0] == 6
    assert result["b"].iloc[0] == 15


def test_apply_chunk_polars_input():
    """C11: apply_chunk with input_frame_type='polars' passes a polars frame
    to the UDF and normalizes the result back to pandas."""
    pytest.importorskip("polars")
    if not is_arrow_dtype_supported():
        pytest.skip("polars UDF result conversion needs ArrowDtype (pyarrow>=8)")
    import polars as pl

    df = pd.DataFrame({"a": [1, 2, 3]})
    result = LiteFrameFunctionalAdapter._run_functional(
        df,
        lambda frame: frame.select(pl.col("a") * 2),
        "apply_chunk",
        0,
        (),
        None,
        input_frame_type="polars",
    )
    assert result["a"].tolist() == [2, 4, 6]


@pytest.mark.parametrize("check_output_dtypes", ["raises", "ignore"])
@pytest.mark.parametrize(
    "df, func, axis, groupby_params, expected_vals, output_dtypes",
    [
        pytest.param(
            pd.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30]}),
            lambda row: row["a"] + row["b"],
            1,
            None,
            [11, 22, 33],
            [pd.Series({"0": "int64"})],
            id="apply_axis1",
        ),
        pytest.param(
            pd.DataFrame({"g": ["a", "a", "b"], "v": [1, 2, 3]}),
            lambda group: int(group["v"].sum()),
            0,
            {"by": "g", "dropna": True, "sort": True, "group_keys": False},
            [3, 3],
            [pd.Series({"0": "int64"})],
            id="groupby_apply",
        ),
        # skip_infer (no declared schema): the frame is still returned to the
        # user, so its integer label (0) must canonicalize to "0" here too.
        pytest.param(
            pd.DataFrame({"a": [1, 2, 3]}),
            lambda row: row["a"] * 2,
            1,
            None,
            [2, 4, 6],
            None,
            id="apply_axis1_skip_infer",
        ),
    ],
)
def test_scalar_result_canonicalizes_label_to_string(
    df,
    func,
    axis,
    groupby_params,
    expected_vals,
    output_dtypes,
    check_output_dtypes,
):
    """SPE scalar UDF returns a column labeled "0" (string), not integer 0.

    Covers the declared-schema path (``output_dtypes=[...]``) and the
    skip_infer path (``output_dtypes=None``); both must canonicalize the
    integer label (0) to "0" in every dtype policy.  Without canonicalization,
    "raises" reported a spurious mismatch and "ignore" returned the integer
    label unchanged.
    """
    result = LiteFrameFunctionalAdapter._run_functional(
        df,
        func,
        "apply",
        axis,
        (),
        None,
        groupby_params=groupby_params,
        output_dtypes=output_dtypes,
        check_output_dtypes=check_output_dtypes,
    )
    assert list(result.columns) == ["0"]
    assert result["0"].tolist() == expected_vals


# ---------------------------------------------------------------------------
# Callable-class lifecycle in SPE _run_functional (comment #1)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mode,axis,udf_fn,output_dtypes,expected_vals",
    [
        # apply / apply_chunk: per-column / whole-frame, returns data * 2
        (
            "apply",
            0,
            lambda d: d * 2,
            [pd.Series({"a": "int64", "b": "int64"})],
            [2, 4, 6, 8],
        ),
        (
            "apply_chunk",
            0,
            lambda d: d * 2,
            [pd.Series({"a": "int64", "b": "int64"})],
            [2, 4, 6, 8],
        ),
        # flat_apply: per-row, returns [{"x": ...}]
        (
            "flat_apply",
            0,
            lambda row: [{"x": int(row["a"])}],
            [pd.Series({"x": wrap_arrow_dtype(pa.int64())})],
            [1, 2, 3],
        ),
    ],
)
def test_spe_callable_class_lifecycle(mode, axis, udf_fn, output_dtypes, expected_vals):
    """SPE _run_functional wraps callable classes (one instance, end=True,
    one close()).

    Regression (comment #1): SPE's apply_chunk / axis=0 / axis=1 called
    func() directly; only flat_apply used the wrapper.  Now all non-groupby
    modes use make_callable_wrapper / call_user_func.
    """
    df = (
        pd.DataFrame({"a": [1, 2, 3]})
        if mode == "flat_apply"
        else pd.DataFrame({"a": [1, 2, 3, 4], "b": [10, 20, 30, 40]})
    )
    state = {"instances": 0, "calls": 0, "closed": 0, "end_values": []}

    class _UDF:
        def __init__(self):
            state["instances"] += 1

        def __call__(self, data, end=False):
            state["calls"] += 1
            state["end_values"].append(end)
            return udf_fn(data)

        def close(self):
            state["closed"] += 1

    result = LiteFrameFunctionalAdapter._run_functional(
        df,
        _UDF,
        mode,
        axis,
        (),
        None,
        output_dtypes=output_dtypes,
        n_output=1 if mode == "flat_apply" else None,
    )
    assert state["instances"] == 1
    assert state["closed"] == 1
    assert all(state["end_values"]), f"got {state['end_values']}"
    col = "x" if mode == "flat_apply" else "a"
    assert result[col].tolist() == expected_vals


def test_spe_groupby_transform_callable_class_per_group_lifecycle():
    """SPE groupby transform: fresh instance + close() per group, original
    row order preserved (comments 241042634 + 241060284).

    Regression: the per-group concat followed group-iteration order, so
    interleaved groups (k=['a','b','a','b']) returned reordered rows.
    Now reindex to df.index after concat.  Also asserts one instance +
    one close per group and end=True on every invocation.
    """
    df = pd.DataFrame({"k": ["a", "b", "a", "b"], "v": [1, 2, 3, 4]})
    state = {"instances": 0, "closed": 0, "end_values": []}

    class _UDF:
        def __init__(self):
            state["instances"] += 1

        def __call__(self, col, end=False):
            state["end_values"].append(end)
            return col * 10

        def close(self):
            state["closed"] += 1

    result = LiteFrameFunctionalAdapter._run_functional(
        df,
        _UDF,
        "transform",
        0,
        (),
        None,
        groupby_params={"by": "k", "dropna": True},
        output_dtypes=[pd.Series({"v": "int64"})],
    )
    assert state["instances"] == 2  # one per group
    assert state["closed"] == 2
    assert all(state["end_values"]), f"got {state['end_values']}"
    assert result["v"].tolist() == [10, 20, 30, 40]
