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

import sys
from typing import Tuple

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pytest

from maxframe.lib.dtypes_extension.blob import (
    AbstractExternalBlob,
    ExternalBlobDtype,
    SolidBlob,
    is_polars_blob_type,
)
from maxframe.lib.version import parse as parse_version
from maxframe.liteframe import LiteFrame
from maxframe.liteframe.core import FrameMetadata
from maxframe.liteframe.expressions import LiteFrameColumn, LiteFrameUDFExpr, NamedExpr
from maxframe.liteframe.functional.apply import apply
from maxframe.liteframe.functional.apply_chunk import apply_chunk
from maxframe.liteframe.functional.flat_apply import flat_apply
from maxframe.liteframe.functional.map import map_
from maxframe.liteframe.functional.transform import transform
from maxframe.liteframe.functional.transform_chunk import transform_chunk
from maxframe.liteframe.functional.type_infer import (
    _extract_dtypes_from_df,
    _normalize_to_arrow_dtype,
    _parse_explicit_dtypes,
    build_mock_dataframe,
    infer_output_dtypes,
    parse_return_annotation,
)
from maxframe.liteframe.functional.utils import (
    build_flat_apply_output_order,
    wrap_inference_func,
)
from maxframe.liteframe.operators.datasource import LiteFrameDataSource
from maxframe.liteframe.operators.functional import LiteFrameFunctional
from maxframe.liteframe.operators.project import LiteFrameProjection
from maxframe.udf import BuiltinFunction
from maxframe.utils import is_arrow_dtype_supported, wrap_arrow_dtype

skip_no_arrow_dtype = pytest.mark.skipif(
    not is_arrow_dtype_supported(), reason="Arrow dtype not supported"
)


# ---------------------------------------------------------------------------
# Type inference
# ---------------------------------------------------------------------------


def test_liteframe_type_from_getitem():
    """LiteFrameType parses slice-based type hints."""
    lft = LiteFrame["a":int, "b":float]
    assert len(lft.data_fields) == 2
    assert lft.data_fields[0].name == "a"
    assert lft.data_fields[1].name == "b"


_params = [
    pytest.param(
        LiteFrame["x":int, "y":float],
        1,
        id="single",
    ),
    pytest.param(
        (LiteFrame["a":int], LiteFrame["b":str]),
        2,
        id="bare_tuple",
    ),
]
if sys.version_info >= (3, 9):
    _params.append(
        pytest.param(
            Tuple[LiteFrame["a":int], LiteFrame["b":float]],  # noqa: F821
            2,
            id="typing_tuple",
        )
    )


@pytest.mark.parametrize("annotation, expected_count", _params)
def test_parse_return_annotation(annotation, expected_count):
    """parse_return_annotation handles single and Tuple of LiteFrameTypes."""
    results = parse_return_annotation(annotation)
    assert len(results) == expected_count


def test_build_mock_dataframe():
    """build_mock_dataframe creates a 2-row pandas DataFrame with Arrow dtypes."""
    dtypes = pd.Series(
        {
            "a": wrap_arrow_dtype(pa.int64()),
            "b": wrap_arrow_dtype(pa.float64()),
            "c": wrap_arrow_dtype(pa.string()),
        }
    )
    mock_df = build_mock_dataframe(dtypes)
    assert mock_df.shape == (2, 3)
    assert list(mock_df.columns) == ["a", "b", "c"]


@pytest.mark.parametrize(
    "dtypes_arg, skip_infer, func, expected_len, expected_cols",
    [
        pytest.param(
            {"x": int, "y": float},
            False,
            lambda df: df,
            1,
            ["x", "y"],
            id="explicit_dtypes",
        ),
        pytest.param(
            None,
            True,
            lambda df: df,
            None,
            None,
            id="skip_infer",
        ),
    ],
)
def test_infer_output_dtypes(dtypes_arg, skip_infer, func, expected_len, expected_cols):
    """infer_output_dtypes with explicit dtypes and skip_infer."""
    input_dtypes = pd.Series(
        {
            "a": wrap_arrow_dtype(pa.int64()),
            "b": wrap_arrow_dtype(pa.float64()),
        }
    )
    result = infer_output_dtypes(
        func=func,
        input_dtypes=input_dtypes,
        dtypes=dtypes_arg,
        skip_infer=skip_infer,
    )
    if expected_len is None:
        assert result is None
    else:
        assert len(result) == expected_len
        if expected_cols is not None:
            assert list(result[0].index) == expected_cols


@skip_no_arrow_dtype
def test_infer_output_dtypes_from_mock():
    """infer_output_dtypes falls back to mock inference."""
    input_dtypes = pd.Series(
        {
            "a": wrap_arrow_dtype(pa.int64()),
            "b": wrap_arrow_dtype(pa.float64()),
        }
    )

    def my_func(df):
        return df.assign(c=df["a"] + df["b"])

    result = infer_output_dtypes(
        func=my_func,
        input_dtypes=input_dtypes,
        dtypes=None,
        skip_infer=False,
    )
    assert len(result) == 1
    assert "c" in result[0].index


_annotation_params = [
    pytest.param(LiteFrame["x":int], [["x"]], id="single"),  # noqa: F821
    pytest.param(
        (LiteFrame["a":int], LiteFrame["b":float]),  # noqa: F821
        [["a"], ["b"]],
        id="bare_tuple",
    ),
]
if sys.version_info >= (3, 9):
    _annotation_params.append(
        pytest.param(
            Tuple[LiteFrame["a":int], LiteFrame["b":float]],  # noqa: F821
            [["a"], ["b"]],
            id="typing_tuple",
        )
    )


@skip_no_arrow_dtype
@pytest.mark.parametrize("annotation, expected_cols", _annotation_params)
def test_infer_output_dtypes_from_annotation(annotation, expected_cols):
    """infer_output_dtypes uses return annotation -> LiteFrame[...]."""
    input_dtypes = pd.Series({"a": wrap_arrow_dtype(pa.int64())})

    def my_func(df) -> annotation:  # noqa: F821
        return df

    result = infer_output_dtypes(
        func=my_func,
        input_dtypes=input_dtypes,
        dtypes=None,
        skip_infer=False,
    )
    assert len(result) == len(expected_cols)
    for i, cols in enumerate(expected_cols):
        assert list(result[i].index) == cols


def _identity_func(x):
    return x


@pytest.mark.parametrize(
    "func, expected_custom_code",
    [
        pytest.param(lambda df: df, True, id="lambda"),
        pytest.param(BuiltinFunction(_identity_func), False, id="builtin"),
    ],
)
def test_liteframe_apply_operator_fields(func, expected_custom_code):
    """LiteFrameFunctional stores mode, axis, groupby_params; has_custom_code detects UDF type."""
    op = LiteFrameFunctional(
        func=func,
        mode="apply_chunk",
        axis=0,
        input_frame_type="pandas",
    )
    assert op.mode == "apply_chunk"
    assert op.axis == 0
    assert op.has_custom_code() is expected_custom_code
    assert op.groupby_params is None


def test_liteframe_projection_udf_has_custom_code():
    """LiteFrameUDFExpr / LiteFrameProjection detect UDF type via
    has_custom_code.  A plain function has custom code; a BuiltinFunction
    does not.
    """
    lf = _make_liteframe({"a": wrap_arrow_dtype(pa.int64())})

    # Plain lambda -> has_custom_code True
    projections = [
        NamedExpr(
            name="a",
            expr=LiteFrameUDFExpr(
                func=lambda x: x,
                mode="map",
                operands=[LiteFrameColumn(name="a", dtype=lf.dtypes["a"])],
                output_names=["a"],
                dtypes={"a": lf.dtypes["a"]},
                input_frame_type=None,
                check_output_dtypes="raises",
            ),
            dtype=lf.dtypes["a"],
        )
    ]
    op = LiteFrameProjection(projections=projections)
    assert op.has_custom_code() is True

    # BuiltinFunction -> has_custom_code False
    projections[0] = NamedExpr(
        name="a",
        expr=LiteFrameUDFExpr(
            func=BuiltinFunction(_identity_func),
            mode="map",
            operands=[LiteFrameColumn(name="a", dtype=lf.dtypes["a"])],
            output_names=["a"],
            dtypes={"a": lf.dtypes["a"]},
            input_frame_type=None,
            check_output_dtypes="raises",
        ),
        dtype=lf.dtypes["a"],
    )
    op = LiteFrameProjection(projections=projections)
    assert op.has_custom_code() is False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_liteframe(dtypes_dict):
    """Helper: create a LiteFrame tileable from a dtypes dict."""
    dtypes = pd.Series(
        {
            k: wrap_arrow_dtype(pa.from_numpy_dtype(v)) if isinstance(v, type) else v
            for k, v in dtypes_dict.items()
        }
    )
    op = LiteFrameDataSource()
    return op.new_liteframe(
        [],
        shape=(100, len(dtypes)),
        physical_dtypes=dtypes,
        frame_metadata=FrameMetadata(),
    )


def _make_liteframe_with_hidden(dtypes_dict, hidden_names):
    """Helper: create a LiteFrame whose physical_dtypes includes hidden columns."""
    dtypes = pd.Series(
        {
            k: wrap_arrow_dtype(pa.from_numpy_dtype(v)) if isinstance(v, type) else v
            for k, v in dtypes_dict.items()
        }
    )
    op = LiteFrameDataSource()
    lf = op.new_liteframe(
        [],
        shape=(100, len(dtypes)),
        physical_dtypes=dtypes,
        frame_metadata=FrameMetadata(),
    )
    lf._data.frame_metadata = FrameMetadata(hidden_columns=list(hidden_names))
    lf._data.__dict__.pop("dtypes", None)
    lf._data.__dict__.pop("columns", None)
    return lf


# ---------------------------------------------------------------------------
# Expression-based API lowering (map / transform_chunk / transform axis=1)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dtypes",
    [
        pytest.param(
            {"a": wrap_arrow_dtype(pa.int64()), "b": wrap_arrow_dtype(pa.float64())},
            id="two_cols",
        ),
        pytest.param(
            {"a": wrap_arrow_dtype(pa.int64())},
            id="single_col",
        ),
    ],
)
def test_map_creates_projection(dtypes):
    """map() creates a LiteFrameProjection and preserves shape."""
    lf = _make_liteframe(dtypes)
    result = map_(lf, lambda x: x * 2, dtype=wrap_arrow_dtype(pa.int64()))
    assert isinstance(result.op, LiteFrameProjection)
    assert list(result.dtypes.index) == list(dtypes.keys())
    assert result.shape == lf.shape


@skip_no_arrow_dtype
def test_transform_chunk_creates_projection():
    """transform_chunk() creates a LiteFrameProjection with UDFExpr."""
    lf = _make_liteframe(
        {"a": wrap_arrow_dtype(pa.int64()), "b": wrap_arrow_dtype(pa.float64())}
    )
    result = transform_chunk(lf, lambda df: df * 2)
    assert isinstance(result.op, LiteFrameProjection)
    assert result.shape[0] == lf.shape[0]


@skip_no_arrow_dtype
@pytest.mark.parametrize(
    "axis, expected_op_type",
    [
        pytest.param(1, LiteFrameProjection, id="axis1_projection"),
        pytest.param(0, LiteFrameFunctional, id="axis0_apply"),
    ],
)
def test_transform_axis(axis, expected_op_type):
    """transform() dispatches to projection (axis=1) or operator (axis=0)."""
    lf = _make_liteframe(
        {"a": wrap_arrow_dtype(pa.int64()), "b": wrap_arrow_dtype(pa.float64())}
    )
    result = transform(lf, lambda x: x, axis=axis)
    assert isinstance(result.op, expected_op_type)
    if axis == 0:
        assert result.op.mode == "transform"


@pytest.mark.parametrize(
    "api_func, mode",
    [
        pytest.param(apply, "apply", id="apply"),
        pytest.param(apply_chunk, "apply_chunk", id="apply_chunk"),
    ],
)
def test_apply_creates_op(api_func, mode):
    """apply/apply_chunk creates LiteFrameFunctional with correct mode."""
    lf = _make_liteframe({"a": wrap_arrow_dtype(pa.int64())})
    result = api_func(lf, lambda df: df, dtypes={"a": int})
    assert isinstance(result.op, LiteFrameFunctional)
    assert result.op.mode == mode


def test_apply_chunk_multiple_outputs():
    """apply_chunk() supports multiple output LiteFrames."""
    lf = _make_liteframe({"a": wrap_arrow_dtype(pa.int64())})
    results = apply_chunk(
        lf,
        lambda df: (df, df),
        dtypes=[{"a": int}, {"b": float}],
    )
    assert len(results) == 2
    # Each output should expose its declared dtypes
    assert list(results[0].dtypes.index) == ["a"]
    assert list(results[1].dtypes.index) == ["b"]


@pytest.mark.parametrize(
    "method_name",
    ["map", "apply", "apply_chunk", "flat_apply", "transform", "transform_chunk"],
)
def test_method_installed(method_name):
    """Implemented LiteFrame methods are callable after _install()."""
    lf = _make_liteframe({"a": wrap_arrow_dtype(pa.int64())})
    assert hasattr(lf, method_name)
    assert callable(getattr(lf, method_name))


@pytest.mark.parametrize(
    "api_name, multi_dtypes",
    [
        pytest.param(
            "transform", [{"a": int, "b": int}, {"a": int, "b": int}], id="transform"
        ),
        pytest.param(
            "transform_chunk",
            [{"a": int, "b": int}, {"a": int, "b": int}],
            id="transform_chunk",
        ),
        pytest.param(
            "groupby_transform_chunk",
            [{"val": int}, {"val": int}],
            id="groupby_transform_chunk",
        ),
    ],
)
def test_transform_apis_reject_multi_output_dtypes(api_name, multi_dtypes):
    """transform-style APIs should reject multi-output dtypes schemas."""
    if api_name.startswith("groupby_"):
        lf = _make_liteframe(
            {
                "key": wrap_arrow_dtype(pa.string()),
                "val": wrap_arrow_dtype(pa.int64()),
            }
        )
        gb = lf.groupby("key")
        method = getattr(gb, api_name.replace("groupby_", ""))
        with pytest.raises(ValueError, match="single-output API"):
            method(lambda g: g, dtypes=multi_dtypes)
    else:
        lf = _make_liteframe(
            {
                "a": wrap_arrow_dtype(pa.int64()),
                "b": wrap_arrow_dtype(pa.int64()),
            }
        )
        method = getattr(lf, api_name)
        # transform defaults to axis=0 (operator-based) and also rejects
        # multi-output via validate_single_output; force axis=1 to exercise the
        # expression-based path's validation.
        kwargs = {"axis": 1} if api_name == "transform" else {}
        with pytest.raises(ValueError, match="single-output API"):
            method(lambda pdf: pdf, dtypes=multi_dtypes, **kwargs)


@skip_no_arrow_dtype
@pytest.mark.parametrize(
    "dtypes_dict, hidden, call, expected_visible",
    [
        # Expression-based UDFs preserve hidden columns via the projection
        # operator's pass-through (_append_hidden_projections).
        pytest.param(
            {
                "a": wrap_arrow_dtype(pa.int64()),
                "b": wrap_arrow_dtype(pa.int64()),
                "_hc": wrap_arrow_dtype(pa.int64()),
            },
            ["_hc"],
            lambda lf: map_(lf, lambda x: x, dtype=wrap_arrow_dtype(pa.int64())),
            ["a", "b"],
            id="map_preserves_hidden",
        ),
        pytest.param(
            {
                "a": wrap_arrow_dtype(pa.int64()),
                "b": wrap_arrow_dtype(pa.int64()),
                "_hc": wrap_arrow_dtype(pa.int64()),
            },
            ["_hc"],
            lambda lf: transform_chunk(lf, lambda df: df * 2),
            ["a", "b"],
            id="transform_chunk_preserves_hidden",
        ),
        pytest.param(
            {
                "a": wrap_arrow_dtype(pa.int64()),
                "b": wrap_arrow_dtype(pa.int64()),
                "_hc": wrap_arrow_dtype(pa.int64()),
            },
            ["_hc"],
            lambda lf: transform(
                lf,
                lambda row: {"a": row["a"], "b": row["b"]},
                axis=1,
                dtypes={"a": int, "b": int},
            ),
            ["a", "b"],
            id="transform_axis1_preserves_hidden",
        ),
        # Operator-based transform(axis=0) preserves hidden columns via
        # _propagate_hidden_metadata + _merge_hidden_into_physical.
        pytest.param(
            {
                "a": wrap_arrow_dtype(pa.int64()),
                "b": wrap_arrow_dtype(pa.int64()),
                "_hc": wrap_arrow_dtype(pa.int64()),
            },
            ["_hc"],
            lambda lf: transform(lf, lambda x: x, axis=0, dtypes={"a": int, "b": int}),
            ["a", "b"],
            id="transform_axis0_preserves_hidden",
        ),
    ],
)
def test_preserves_hidden_columns(dtypes_dict, hidden, call, expected_visible):
    """UDF APIs preserve hidden columns into physical_dtypes + metadata.

    Every preserve-policy UDF (expression-based ``map`` / ``transform_chunk``
    / ``transform(axis=1)`` via the projection pass-through, and operator-based
    ``transform(axis=0)`` via ``_merge_hidden_into_physical``) must keep the
    hidden column physical-but-hidden so the visible schema stays consistent
    and ``dtypes.drop(hidden)`` does not raise.
    """
    lf = _make_liteframe_with_hidden(dtypes_dict, hidden_names=hidden)
    assert "_hc" in lf._physical_dtypes.index

    result = call(lf)
    # visible schema excludes the hidden column and must not raise
    assert list(result.dtypes.index) == expected_visible
    # hidden column is carried into physical_dtypes + metadata
    assert "_hc" in result._physical_dtypes.index
    assert "_hc" in result._hidden_columns
    assert len(result._physical_dtypes) == len(expected_visible) + len(hidden)


@pytest.mark.parametrize(
    "method, kwargs",
    [
        pytest.param("apply", {}, id="apply_axis0"),
        pytest.param("apply_chunk", {}, id="apply_chunk"),
    ],
)
def test_op_apply_family_drops_hidden_columns(method, kwargs):
    """apply(axis=0) / apply_chunk drop hidden columns.

    Apply-family APIs that may change the row count (apply(axis=0) reduction,
    apply_chunk) drop hidden columns from the UDF input at the API
    (``self.drop(hidden)``) and do not carry them into the output, because
    re-appending hidden columns from the original input would misalign rows
    (or leave hidden metadata inconsistent with the physical result).  A hidden
    column only survives if the UDF explicitly returns it.

    Note: apply(axis=1) preserves row count (one result per row), so it
    preserves hidden columns — see test_op_apply_axis1_preserves_hidden.

    The UDF body is irrelevant here: explicit ``dtypes`` short-circuits type
    inference, so the UDF is never invoked during op construction at this
    (logical) level.
    """
    lf = _make_liteframe_with_hidden(
        {
            "a": wrap_arrow_dtype(pa.int64()),
            "_hc": wrap_arrow_dtype(pa.int64()),
        },
        hidden_names=["_hc"],
    )
    result = getattr(lf, method)(lambda df: df[["a"]], dtypes={"a": int}, **kwargs)
    assert "_hc" not in result._physical_dtypes.index
    assert not result._hidden_columns


@skip_no_arrow_dtype
def test_op_apply_axis1_preserves_hidden():
    """apply(axis=1) preserves hidden columns (row count is unchanged).

    Unlike apply(axis=0) / apply_chunk, apply(axis=1) always produces one
    result per input row, so the row count is preserved and hidden columns
    can be safely re-appended after execution.  The hidden columns are
    stripped from the UDF input and carried into the output's
    physical_dtypes + frame_metadata, matching the transform-family preserve
    policy.
    """
    lf = _make_liteframe_with_hidden(
        {
            "a": wrap_arrow_dtype(pa.int64()),
            "b": wrap_arrow_dtype(pa.int64()),
            "_hc": wrap_arrow_dtype(pa.int64()),
        },
        hidden_names=["_hc"],
    )
    result = lf.apply(
        lambda row: {"x": row["a"], "y": row["b"]},
        axis=1,
        dtypes={"x": int, "y": int},
    )
    # Hidden column is carried into physical_dtypes and frame_metadata
    assert "_hc" in result._physical_dtypes.index
    assert "_hc" in result._hidden_columns
    # Visible schema excludes the hidden column
    assert list(result.dtypes.index) == ["x", "y"]
    assert len(result._physical_dtypes) == 3  # x, y, _hc


# ---------------------------------------------------------------------------
# Explicit dtypes parsing (single vs multiple outputs)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dtypes_arg, expected_n_outputs, expected_cols",
    [
        # list of (name, dtype) tuples -> SINGLE output, named schema
        pytest.param([("x", int), ("y", float)], 1, ["x", "y"], id="tuples_single"),
        # single (name, dtype) tuple in a list -> single output, one column
        pytest.param([("x", int)], 1, ["x"], id="single_tuple"),
        # dict -> single output
        pytest.param({"x": int, "y": float}, 1, ["x", "y"], id="dict_single"),
        # list of dicts -> MULTIPLE outputs (one schema per output)
        pytest.param([{"a": int}, {"b": float}], 2, ["a"], id="dicts_multi"),
        # single dict in a list -> one output (multi-output of length 1)
        pytest.param([{"x": int}], 1, ["x"], id="single_dict_list"),
        # numeric dict key canonicalizes to "0" (boundary contract)
        pytest.param({0: int}, 1, ["0"], id="numeric_key"),
        # positional list of bare dtypes -> single output, string index
        # (LiteFrame column names are always strings; integer positional
        # labels canonicalize to "0", "1" at the schema boundary)
        pytest.param(["int64", "float64"], 1, ["0", "1"], id="positional_bare"),
    ],
)
def test_parse_explicit_dtypes(dtypes_arg, expected_n_outputs, expected_cols):
    """_parse_explicit_dtypes distinguishes single-output named schemas from
    multiple outputs.

    Regression: a list of (name, dtype) tuples was
    parsed as a positional list -> {0: object, 1: object} instead of a single
    named schema {x: int64, y: float64}.  It must also be distinguished from
    the multiple-output form (a list of dicts/pd.Series).
    """
    result = _parse_explicit_dtypes(dtypes_arg)
    assert len(result) == expected_n_outputs
    assert list(result[0].index) == expected_cols
    # the tuple form yields real dtypes, not object
    if dtypes_arg == [("x", int), ("y", float)]:
        assert str(result[0]["x"]).startswith("int")
        assert "float" in str(result[0]["y"])


def test_parse_explicit_dtypes_numeric_string_collision_rejected():
    """A declared ``dtypes={0: int, "0": float}`` raises a collision.

    Regression: both labels canonicalize to "0" after stringification, so
    the schema becomes ambiguous (duplicate column label).  The boundary
    contract rejects this at declaration time rather than silently dropping
    one declaration.
    """
    with pytest.raises(ValueError, match="Duplicate column labels"):
        _parse_explicit_dtypes({0: int, "0": float})


# ---------------------------------------------------------------------------
# apply() rejects unsupported axis values
# ---------------------------------------------------------------------------


@skip_no_arrow_dtype
@pytest.mark.parametrize(
    "axis",
    [
        pytest.param(2, id="axis_2"),
        pytest.param(3, id="axis_3"),
        pytest.param(-3, id="axis_neg3"),
    ],
)
def test_apply_rejects_invalid_axis(axis):
    """apply() raises ValueError for unsupported axis values at construction.

    Regression: an unsupported axis (e.g. axis=2) was
    passed through and silently fell into the executor's whole-frame fallback,
    creating an undocumented third apply mode.  Now it is validated up front
    via validate_axis (imported from tensor.utils), matching pandas-style APIs.
    """
    lf = _make_liteframe(
        {"a": wrap_arrow_dtype(pa.int64()), "b": wrap_arrow_dtype(pa.float64())}
    )
    with pytest.raises(ValueError, match="axis"):
        apply(lf, lambda df: df, axis=axis, dtypes={"a": int, "b": float})


# ---------------------------------------------------------------------------
# GroupBy API tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["apply", "apply_chunk"])
def test_groupby_apply_via_method(method):
    """LiteFrameGroupBy.apply/apply_chunk work after _install()."""
    lf = _make_liteframe(
        {
            "key": wrap_arrow_dtype(pa.string()),
            "val": wrap_arrow_dtype(pa.int64()),
        }
    )
    gb = lf.groupby("key")
    result = getattr(gb, method)(lambda g: g, dtypes={"key": str, "val": int})
    assert result is not None
    assert isinstance(result.op, LiteFrameFunctional)
    assert result.op.mode == method
    assert result.op.groupby_params is not None


@skip_no_arrow_dtype
@pytest.mark.parametrize("method", ["apply", "apply_chunk"])
def test_groupby_apply_drops_hidden_columns(method):
    """GroupBy apply/apply_chunk drop hidden columns from the operator input.

    Regression: groupby_apply_op computed inference dtypes from visible
    columns (effective_groupby_input_dtypes excludes hidden) but passed the
    original physical LiteFrame (whose _physical_dtypes includes hidden) into
    the operator, so hidden columns leaked into the UDF input at execution.
    Now hidden columns are dropped from the operator input at the API
    (``self._liteframe.drop(hidden)``), matching the non-groupby apply /
    apply_chunk drop policy.  A UDF asking ``'_hc' in group.columns`` must see
    False (the executor-level regression), and here at the logical level the
    functional operator's input must not carry ``_hc`` in physical_dtypes.
    """
    lf = _make_liteframe_with_hidden(
        {
            "key": wrap_arrow_dtype(pa.string()),
            "val": wrap_arrow_dtype(pa.int64()),
            "_hc": wrap_arrow_dtype(pa.int64()),
        },
        hidden_names=["_hc"],
    )
    assert "_hc" in lf._physical_dtypes.index
    gb = lf.groupby("key")
    result = getattr(gb, method)(lambda g: g, dtypes={"key": str, "val": int})
    assert isinstance(result.op, LiteFrameFunctional)
    # The operator's input is the (hidden-stripped) LiteFrame; its physical
    # schema must not include the dropped hidden column.
    op_input = result.op.inputs[0]
    assert "_hc" not in op_input._physical_dtypes.index
    # And the output carries no hidden metadata (apply-family drop policy).
    assert not result._hidden_columns


@skip_no_arrow_dtype
@pytest.mark.parametrize(
    "udf, expected_cols",
    [
        # Series-returning transform: one output column (NOT a transposed
        # multi-column frame from the Series index).
        pytest.param(lambda g: g["val"] * 2, ["val"], id="series_result"),
        # DataFrame-returning transform: columns from the result frame.
        pytest.param(lambda g: g.assign(val=g["val"] * 2), ["val"], id="df_result"),
        # Scalar-returning transform: broadcast to all input columns
        # (only "val" here since "key" is excluded), so the inferred column
        # is "val" (matching pandas groupby.transform broadcast semantics).
        pytest.param(lambda g: int(g["val"].sum()), ["val"], id="scalar_result"),
    ],
)
def test_groupby_transform_inference_series_one_column(udf, expected_cols):
    """A Series-returning groupby transform infers ONE output column.

    Regression: the groupby inference wrapper used the apply normalization
    (``_normalize_result_to_frame``) which turns a Series into a one-row,
    multi-column frame (Series index -> columns).  But the transform runtime
    contract treats a Series as a single output column, so inference inferred
    the wrong column count (e.g. a 2-element Series -> columns {0,1}) and
    execution then raised "a Series can only be used as a single output
    column".  The wrapper now uses the transform normalization for transform
    modes so a Series infers exactly one column.
    """
    lf = _make_liteframe(
        {
            "key": wrap_arrow_dtype(pa.string()),
            "val": wrap_arrow_dtype(pa.int64()),
        }
    )
    result = lf.groupby("key").transform(udf)
    assert list(result.dtypes.index) == expected_cols
    assert len(result.dtypes) == 1


@skip_no_arrow_dtype
@pytest.mark.parametrize(
    "mode, input_frame_type, expected_cols",
    [
        pytest.param("groupby", "pandas", ["key", "0"], id="groupby_pandas"),
        pytest.param("groupby", "polars", ["key", "0"], id="groupby_polars"),
        pytest.param("apply_row", "pandas", ["0"], id="apply_row_pandas"),
        pytest.param("apply_row", "polars", ["0"], id="apply_row_polars"),
        # whole-frame inference (transform_chunk / apply_chunk): a UDF
        # returning ``pd.DataFrame({0: ...})`` infers "0", not integer 0.
        pytest.param("apply_chunk", "pandas", ["0"], id="apply_chunk"),
        pytest.param("transform_chunk", "pandas", ["0"], id="transform_chunk"),
    ],
)
def test_scalar_inference_label_stringified(mode, input_frame_type, expected_cols):
    """A scalar/positional UDF result infers a string column label "0".

    LiteFrame column names are always strings; a scalar result normalizes to a
    pandas frame with an integer label (0), and a whole-frame ``pd.DataFrame``
    built from an integer key ({0: ...}) carries that integer label -- both
    must be stringified to "0" at the inference boundary so the schema matches
    the string-labeled runtime output and survives the ODPS round trip.
    """
    if mode == "groupby":
        lf = _make_liteframe(
            {"key": wrap_arrow_dtype(pa.string()), "val": wrap_arrow_dtype(pa.int64())}
        )
        result = lf.groupby("key").apply(
            lambda g: len(g), input_frame_type=input_frame_type
        )
    elif mode == "apply_row":
        lf = _make_liteframe(
            {"a": wrap_arrow_dtype(pa.int64()), "b": wrap_arrow_dtype(pa.int64())}
        )
        result = lf.apply(
            lambda row: row["a"] + row["b"],
            axis=1,
            input_frame_type=input_frame_type,
        )
    else:
        # whole-frame: transform_chunk calls infer_output_dtypes directly;
        # apply_chunk goes through the wrap_inference_func "chunk" wrapper.
        lf = _make_liteframe({"a": wrap_arrow_dtype(pa.int64())})
        udf = lambda df: pd.DataFrame({0: df["a"] * 2})  # noqa: E731
        result = (
            lf.apply_chunk(udf) if mode == "apply_chunk" else lf.transform_chunk(udf)
        )
    assert list(result.dtypes.index) == expected_cols


@skip_no_arrow_dtype
def test_groupby_transform_skip_infer_does_not_advertise_input_schema():
    """skip_infer=True keeps the output schema unknown (not the input schema).

    Regression: when skip_infer=True the transform output fell back to the
    full input schema (key,val), so a downstream select on a column the UDF
    does not produce (e.g. result[['key']] when the UDF returns only 'val')
    was accepted at graph construction and then failed with
    ColumnNotFoundError at execution.  Now the output is untyped
    (physical_dtypes=None) so the schema stays unknown until the UDF runs --
    mirroring build_functional_output's skip_infer path.
    """
    lf = _make_liteframe(
        {
            "key": wrap_arrow_dtype(pa.string()),
            "val": wrap_arrow_dtype(pa.int64()),
        }
    )
    result = lf.groupby("key").transform(lambda g: g[["val"]], skip_infer=True)
    # Output schema is unknown: no physical_dtypes, key not predeclared.
    assert result._physical_dtypes is None
    assert result._hidden_columns == set()
    # The key column must not be advertised as a known output column.
    # dtypes is None/empty when the schema is unknown.
    assert result.dtypes is None or len(result.dtypes) == 0


def test_groupby_getitem_selection():
    """LiteFrameGroupBy.__getitem__ stores column selection."""
    lf = _make_liteframe(
        {
            "key": wrap_arrow_dtype(pa.string()),
            "v1": wrap_arrow_dtype(pa.int64()),
            "v2": wrap_arrow_dtype(pa.int64()),
        }
    )
    gb = lf.groupby("key")
    # __getitem__ returns a new LiteFrameGroupBy with selection
    gb_sel = gb["v1"]
    assert gb_sel._selection == ["v1"]
    # Original is not modified
    assert gb._selection is None

    # selection appears in groupby_params
    params = gb_sel.groupby_params
    assert params["selection"] == ["v1"]

    # Multiple columns
    gb_multi = gb[["v1", "v2"]]
    assert gb_multi._selection == ["v1", "v2"]

    # Invalid column raises KeyError
    with pytest.raises(KeyError, match="not_found"):
        gb["not_found"]


@skip_no_arrow_dtype
@pytest.mark.parametrize(
    "dtypes_dict, hidden, call, expected_visible",
    [
        pytest.param(
            {
                "a": wrap_arrow_dtype(pa.int64()),
                "b": wrap_arrow_dtype(pa.int64()),
                "_hc": wrap_arrow_dtype(pa.int64()),
            },
            ["_hc"],
            lambda lf: transform(lf, lambda x: x, axis=0, dtypes={"a": int, "b": int}),
            ["a", "b"],
            id="transform_axis0",
        ),
        pytest.param(
            {
                "key": wrap_arrow_dtype(pa.string()),
                "val": wrap_arrow_dtype(pa.int64()),
                "_hc": wrap_arrow_dtype(pa.int64()),
            },
            ["_hc"],
            lambda lf: lf.groupby("key").transform(lambda g: g, dtypes={"val": int}),
            ["val"],
            id="groupby_transform",
        ),
        pytest.param(
            {
                "key": wrap_arrow_dtype(pa.string()),
                "val": wrap_arrow_dtype(pa.int64()),
                "_hc": wrap_arrow_dtype(pa.int64()),
            },
            ["_hc"],
            lambda lf: lf.groupby("key").transform_chunk(
                lambda g: g, dtypes={"val": int}
            ),
            ["val"],
            id="groupby_transform_chunk",
        ),
    ],
)
def test_transform_carries_hidden_columns(dtypes_dict, hidden, call, expected_visible):
    """transform-style APIs carry hidden columns into physical_dtypes.

    Regression: hidden columns were preserved in frame_metadata but absent
    from physical_dtypes, so LiteFrameData.dtypes raised KeyError when
    dropping hidden names.  Hidden columns must be present in physical_dtypes
    and tracked in metadata so the visible schema stays consistent.
    """
    lf = _make_liteframe_with_hidden(dtypes_dict, hidden_names=hidden)
    assert "_hc" in lf._physical_dtypes.index

    result = call(lf)
    # visible schema excludes the hidden column and must not raise
    assert list(result.dtypes.index) == expected_visible
    # hidden column is carried into physical_dtypes + metadata
    assert "_hc" in result._physical_dtypes.index
    assert "_hc" in result._hidden_columns
    assert len(result._physical_dtypes) == len(expected_visible) + len(hidden)


@pytest.mark.parametrize("method", ["apply", "apply_chunk"])
def test_groupby_apply_does_not_propagate_hidden_metadata(method):
    """groupby.apply/apply_chunk must not carry hidden metadata into output.

    apply may change the row count / row order, so re-appending hidden columns
    from the original input would misalign rows (or, for row-count-changing
    apply, leave hidden metadata inconsistent with the physical result and
    raise KeyError in LiteFrameData.dtypes).  Hidden columns only survive when
    the UDF explicitly returns them.
    """
    lf = _make_liteframe_with_hidden(
        {
            "key": wrap_arrow_dtype(pa.string()),
            "val": wrap_arrow_dtype(pa.int64()),
            "_hc": wrap_arrow_dtype(pa.int64()),
        },
        hidden_names=["_hc"],
    )
    result = getattr(lf.groupby("key"), method)(
        lambda g: g[["key", "val"]], dtypes={"key": str, "val": int}
    )
    assert "_hc" not in result._physical_dtypes.index
    assert not result._hidden_columns


def test_groupby_apply_multi_output_via_method():
    """LiteFrameGroupBy.apply() with multiple outputs creates n_output outputs."""
    lf = _make_liteframe(
        {
            "key": wrap_arrow_dtype(pa.string()),
            "val": wrap_arrow_dtype(pa.int64()),
        }
    )
    gb = lf.groupby("key")
    result = gb.apply(
        lambda g: (g, g * 2),
        dtypes=[{"key": str, "val": int}, {"key": str, "val": int}],
    )
    assert result is not None
    assert len(result) == 2
    for r in result:
        assert isinstance(r.op, LiteFrameFunctional)
        assert r.op.mode == "apply"
        assert r.op.groupby_params is not None


# ---------------------------------------------------------------------------
# GroupBy inference: the "groupby" wrapper normalizes non-DataFrame results
# so mock inference produces a non-empty schema (mirrors _normalize_group_result).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "udf,expected_n_outputs",
    [
        pytest.param(lambda g: g.assign(val2=g["val"] * 2), 1, id="dataframe"),
        pytest.param(lambda g: g["val"] * 2, 1, id="series"),
        pytest.param(lambda g: g["val"].sum(), 1, id="scalar"),
        pytest.param(lambda g: {"s": g["val"].sum()}, 1, id="dict"),
        pytest.param(lambda g: (g, g * 10), 2, id="tuple_of_df"),
        pytest.param(lambda g: (g["val"] * 2, g["val"].sum()), 2, id="tuple_mixed"),
    ],
)
def test_groupby_inference_wrapper_normalizes_results(udf, expected_n_outputs):
    """The groupby inference wrapper normalizes non-DataFrame UDF results.

    GroupBy UDFs may return Series/scalar/dict (not just DataFrame).  The
    "groupby" wrapper in wrap_inference_func normalizes these into frames so
    infer_output_dtypes can extract a non-empty schema.  Without this wrapper
    (the old "chunk" mode that returned raw results), Series/scalar/dict
    returns produced empty schemas and downstream execution failed.
    """
    mock_df = pd.DataFrame({"key": ["a", "b", "a"], "val": [1, 2, 3]})
    wrapper = wrap_inference_func(udf, "groupby", "pandas", (), {})
    result = wrapper(mock_df)

    if expected_n_outputs > 1:
        assert isinstance(result, tuple)
        assert len(result) == expected_n_outputs
        for element in result:
            assert isinstance(element, pd.DataFrame)
            # Each normalized element must have extractable (non-empty) dtypes.
            dtypes = _extract_dtypes_from_df(element)
            assert len(dtypes) > 0, "inference produced an empty schema"
    else:
        assert isinstance(result, pd.DataFrame)
        dtypes = _extract_dtypes_from_df(result)
        assert len(dtypes) > 0, "inference produced an empty schema"


@pytest.mark.parametrize("input_frame_type", ["pandas", "polars"])
def test_groupby_transform_per_row_series_not_misclassified_as_per_column(
    input_frame_type,
):
    """A per-row Series whose length equals the input column count stays per-row.

    Regression (review 28568533 #8): the two-row mock group made a
    2-element positional Series indistinguishable from a per-column Series
    (g.mean()), so mock inference treated it as one value per column and
    execution broadcast across the wrong columns
    (v1=[11,11,110,110], v2=[22,22,220,220]).  A positional / RangeIndex
    Series must be a single per-row output column even when its length
    happens to equal the input column count; only a Series whose labels
    match the input columns is per-column.
    """
    if input_frame_type == "pandas":
        udf = lambda g: (g["v1"] + g["v2"]).reset_index(drop=True)
    else:
        udf = lambda g: pl.Series(
            [int(g["v1"][0]) + int(g["v2"][0]), int(g["v1"][1]) + int(g["v2"][1])]
        )
    # The group key column is excluded from the UDF input (pandas 3.0
    # semantics), so the mock group the UDF sees carries only v1/v2.
    mock_df = pd.DataFrame({"v1": [1, 2], "v2": [10, 20]})
    wrapper = wrap_inference_func(udf, "groupby_transform", input_frame_type, (), {})
    result = wrapper(mock_df)
    assert isinstance(result, pd.DataFrame)
    dtypes = _extract_dtypes_from_df(result)
    # A per-row Series is a single output column, NOT one value per column.
    assert len(dtypes) == 1, (
        f"per-row Series was misclassified as per-column: "
        f"inferred {len(dtypes)} columns {list(dtypes.index)}"
    )
    # g.mean()-style (labels == input columns) is still per-column (2 cols).
    per_col = wrap_inference_func(
        lambda g: g[["v1", "v2"]].mean(), "groupby_transform", "pandas", (), {}
    )(mock_df)
    assert len(_extract_dtypes_from_df(per_col)) == 2


# ---------------------------------------------------------------------------
# flat_apply (operator-based, row-expanding)
# ---------------------------------------------------------------------------


def test_flat_apply_creates_op():
    """flat_apply() creates a LiteFrameFunctional with mode='flat_apply'."""
    lf = _make_liteframe({"a": wrap_arrow_dtype(pa.int64())})
    result = flat_apply(lf, lambda row: [row["a"]], dtypes={"x": int})
    assert isinstance(result.op, LiteFrameFunctional)
    assert result.op.mode == "flat_apply"
    assert np.isnan(result.shape[0])  # row count unknown (1:N expansion)


@pytest.mark.parametrize(
    "subset, func, dtypes, expected_op_subset, expected_cols",
    [
        pytest.param(
            None,
            lambda row: [row["a"]],
            {"x": int},
            ["a", "b", "c"],
            ["x"],
            id="none_pure_flatmap",
        ),
        pytest.param(
            "a",
            lambda row: [row["a"]],
            {"x": int},
            ["a"],
            ["x", "b", "c"],
            id="scalar_string",
        ),
        pytest.param(
            ["b"],
            lambda row: [row["b"]],
            {"out": str},
            ["b"],
            ["a", "out", "c"],
            id="broadcast_keeps_order",
        ),
    ],
)
def test_flat_apply_subset(subset, func, dtypes, expected_op_subset, expected_cols):
    """subset restricts UDF input; non-subset cols broadcast in original order."""
    lf = _make_liteframe(
        {
            "a": wrap_arrow_dtype(pa.int64()),
            "b": wrap_arrow_dtype(pa.string()),
            "c": wrap_arrow_dtype(pa.float64()),
        }
    )
    result = flat_apply(lf, func, subset=subset, dtypes=dtypes)
    assert result.op.subset == expected_op_subset
    assert list(result.dtypes.index) == expected_cols
    # Broadcast cols inherit their input dtypes.
    for c in expected_cols:
        if c in ("a", "b", "c"):
            assert result.dtypes[c] == lf.dtypes[c]


@pytest.mark.parametrize(
    "subset, dtypes, exc, match",
    [
        pytest.param(
            ["missing"], {"x": int}, KeyError, "not found", id="invalid_subset_col"
        ),
        pytest.param(["b"], {"a": int}, ValueError, "collide", id="output_collision"),
    ],
)
def test_flat_apply_subset_errors(subset, dtypes, exc, match):
    """Invalid subset columns and broadcast-name collisions raise."""
    lf = _make_liteframe(
        {"a": wrap_arrow_dtype(pa.int64()), "b": wrap_arrow_dtype(pa.int64())}
    )
    with pytest.raises(exc, match=match):
        flat_apply(lf, lambda row: [row["b"]], subset=subset, dtypes=dtypes)


def test_flat_apply_drops_hidden_columns():
    """flat_apply drops hidden columns from the UDF input and output."""
    lf = _make_liteframe_with_hidden(
        {"a": wrap_arrow_dtype(pa.int64()), "_hc": wrap_arrow_dtype(pa.int64())},
        hidden_names=["_hc"],
    )
    result = flat_apply(lf, lambda row: [row["a"]], dtypes={"x": int})
    assert "_hc" not in result._physical_dtypes.index
    assert not result._hidden_columns


def test_flat_apply_skip_infer_untyped():
    """skip_infer produces an untyped output with an unknown shape."""
    lf = _make_liteframe({"a": wrap_arrow_dtype(pa.int64())})
    result = flat_apply(lf, lambda row: [row["a"]], skip_infer=True)
    assert result.op.output_dtypes is None
    assert np.isnan(result.shape[0])
    assert np.isnan(result.shape[1])


@pytest.mark.parametrize(
    "subset, dtypes, expected_cols_list",
    [
        pytest.param(
            ["a", "b"], [{"x": int}, {"y": int}], [["x"], ["y"]], id="no_broadcast"
        ),
        pytest.param(
            ["b"],
            [{"x": int}, {"y": int}],
            [["a", "x"], ["a", "y"]],
            id="with_broadcast",
        ),
    ],
)
def test_flat_apply_multiple_outputs(subset, dtypes, expected_cols_list):
    """flat_apply() supports multiple outputs; broadcast cols appear in each."""
    lf = _make_liteframe(
        {"a": wrap_arrow_dtype(pa.int64()), "b": wrap_arrow_dtype(pa.int64())}
    )
    results = flat_apply(
        lf,
        lambda row: ([row[subset[-1]]], [row[subset[-1]] * 2]),
        subset=subset,
        dtypes=dtypes,
    )
    assert len(results) == 2
    assert results[0].op.n_output == 2
    for i, expected in enumerate(expected_cols_list):
        assert list(results[i].dtypes.index) == expected


def test_flat_apply_inference_from_mock():
    """flat_apply infers func-output dtypes from a mock call when dtypes=None."""
    lf = _make_liteframe({"a": wrap_arrow_dtype(pa.int64())})

    def func(row):
        return [{"x": row["a"], "y": float(row["a"])}]

    result = flat_apply(lf, func)
    assert list(result.dtypes.index) == ["x", "y"]


def test_flat_apply_scalar_result_canonicalizes_label_to_string():
    """flat_apply with a scalar pandas UDF result advertises "0" not 0.

    Regression: the inference wrapper previously stringified column labels
    only when ``input_frame_type == "polars"``; the pandas path left the
    integer label ``0`` on the inferred schema, so the declared dtypes index
    was ``[0]`` instead of ``["0"]`` — contradicting the LiteFrame boundary
    contract that column names are always strings.
    """
    lf = _make_liteframe({"a": wrap_arrow_dtype(pa.int64())})
    result = flat_apply(lf, lambda row: row["a"], input_frame_type="pandas")
    assert list(result.dtypes.index) == ["0"]


def test_flat_apply_range_column_is_broadcastable():
    """A visible range column is eligible for broadcast (kept in original order)."""
    op = LiteFrameDataSource()
    lf = op.new_liteframe(
        [],
        shape=(100, 1),
        physical_dtypes=pd.Series({"a": wrap_arrow_dtype(pa.int64())}),
        frame_metadata=FrameMetadata(range_columns={"id": (0, 100, 1)}),
    )
    result = flat_apply(lf, lambda row: [row["a"]], subset=["a"], dtypes={"x": int})
    assert list(result.dtypes.index) == ["id", "x"]
    assert result.dtypes["id"] == wrap_arrow_dtype(pa.int64())
    # Range-column metadata is dropped from the output (row count changes).
    assert not result.frame_metadata.range_columns


# ---------------------------------------------------------------------------
# groupby flat_apply (operator-based, row-expanding per group)
# ---------------------------------------------------------------------------


def test_groupby_flat_apply_creates_op():
    """groupby_flat_apply creates a LiteFrameFunctional with groupby_params."""
    lf = _make_liteframe(
        {
            "key": wrap_arrow_dtype(pa.string()),
            "val": wrap_arrow_dtype(pa.int64()),
        }
    )
    result = lf.groupby("key")[["key", "val"]].flat_apply(
        lambda row: [{"x": row["val"]}], subset=["val"], dtypes={"x": int}
    )
    assert isinstance(result.op, LiteFrameFunctional)
    assert result.op.mode == "flat_apply"
    assert result.op.groupby_params is not None
    assert result.op.subset == ["val"]
    # subset=['val'] -> broadcast=['key'], func-out=['x'] -> {key, x}
    assert list(result.dtypes.index) == ["key", "x"]
    assert np.isnan(result.shape[0])  # row count unknown (1:N expansion)


@pytest.mark.parametrize(
    "subset, expected_cols",
    [
        pytest.param(None, ["key", "x"], id="none_pure_flatmap"),
        pytest.param(["val"], ["key", "x"], id="broadcast_key"),
        pytest.param(["key", "val"], ["key", "x"], id="no_broadcast"),
    ],
)
def test_groupby_flat_apply_subset(subset, expected_cols):
    """subset restricts UDF input; non-subset effective cols broadcast.

    The grouping keys are prepended to the output schema by default
    (group_keys=True); when the keys already survive via broadcast (explicit
    selection of the key columns) they are not duplicated.
    """
    lf = _make_liteframe(
        {
            "key": wrap_arrow_dtype(pa.string()),
            "val": wrap_arrow_dtype(pa.int64()),
        }
    )
    result = lf.groupby("key")[["key", "val"]].flat_apply(
        lambda row: [{"x": row["val"]}], subset=subset, dtypes={"x": int}
    )
    assert list(result.dtypes.index) == expected_cols
    assert result.op.subset == (["key", "val"] if subset is None else list(subset))


def test_groupby_flat_apply_drops_hidden_columns():
    """groupby_flat_apply drops hidden columns from input and output."""
    lf = _make_liteframe_with_hidden(
        {
            "key": wrap_arrow_dtype(pa.string()),
            "val": wrap_arrow_dtype(pa.int64()),
            "_hc": wrap_arrow_dtype(pa.int64()),
        },
        hidden_names=["_hc"],
    )
    result = lf.groupby("key")[["key", "val"]].flat_apply(
        lambda row: [{"x": row["val"]}], subset=["val"], dtypes={"x": int}
    )
    assert "_hc" not in result._physical_dtypes.index
    assert not result._hidden_columns
    # The operator input must not carry the hidden column.
    assert "_hc" not in result.op.inputs[0]._physical_dtypes.index


def test_groupby_flat_apply_multi_output():
    """groupby_flat_apply supports multiple outputs; broadcast in each."""
    lf = _make_liteframe(
        {
            "key": wrap_arrow_dtype(pa.string()),
            "val": wrap_arrow_dtype(pa.int64()),
        }
    )
    results = lf.groupby("key")[["key", "val"]].flat_apply(
        lambda row: ([{"x": row["val"]}], [{"y": row["val"] * 2}]),
        subset=["val"],
        dtypes=[{"x": int}, {"y": int}],
    )
    assert len(results) == 2
    assert results[0].op.n_output == 2
    assert list(results[0].dtypes.index) == ["key", "x"]
    assert list(results[1].dtypes.index) == ["key", "y"]


def test_groupby_flat_apply_key_exclusion_default():
    """Without a selection, groupby key columns are excluded from the UDF input.

    The keys still land in the output schema (prepended, group_keys=True
    default) -- only the UDF input excludes them.
    """
    lf = _make_liteframe(
        {
            "key": wrap_arrow_dtype(pa.string()),
            "val": wrap_arrow_dtype(pa.int64()),
        }
    )
    # No selection: effective cols = [val] (key excluded); subset=None -> [val].
    result = lf.groupby("key").flat_apply(
        lambda row: [{"x": row["val"]}], dtypes={"x": int}
    )
    assert result.op.subset == ["val"]
    assert list(result.dtypes.index) == ["key", "x"]


@pytest.mark.parametrize(
    "group_keys, subset, expected",
    [
        # selection includes key; key governed by group_keys, not broadcast.
        pytest.param(True, ["val"], ["key", "x"], id="sel_includes_key_gk_true"),
        pytest.param(False, ["val"], ["x"], id="sel_includes_key_gk_false"),
        # key as first subset anchor: func-out at key's position.
        pytest.param(True, ["key"], ["key", "x", "val"], id="anchor_key_gk_true"),
        pytest.param(False, ["key"], ["x", "val"], id="anchor_key_gk_false"),
    ],
)
def test_groupby_flat_apply_schema_group_keys_and_anchor(group_keys, subset, expected):
    """groupby flat_apply output schema: key follows ``group_keys`` contract;
    func-out inserts at the first subset column's position.
    """
    lf = _make_liteframe(
        {"key": wrap_arrow_dtype(pa.string()), "val": wrap_arrow_dtype(pa.int64())}
    )
    result = lf.groupby("key", group_keys=group_keys)[["key", "val"]].flat_apply(
        lambda row: [{"x": 1}], subset=subset, dtypes={"x": int}
    )
    assert list(result.dtypes.index) == expected


@pytest.mark.parametrize(
    "pl_dtype, expected_pa",
    [
        pytest.param(pl.Int64, pa.int64(), id="int64"),
        pytest.param(pl.Float64, pa.float64(), id="float64"),
        pytest.param(pl.Utf8, pa.large_string(), id="string"),
        # polars' Decimal precision/scale round-trip through to_arrow() is
        # version-dependent: 0.18.x ignores the declared precision/scale and
        # defaults to decimal128(38, 0), while 1.x preserves them.  On legacy
        # polars (< 1.0, allowed for Python < 3.9) the declaration is
        # deliberately rejected instead of drifting; on modern polars the
        # exact declared parameters are asserted.
        pytest.param(
            pl.Decimal(precision=10, scale=2),
            pa.decimal128(10, 2),
            id="decimal",
        ),
        pytest.param(pl.List(pl.Int64), pa.large_list(pa.int64()), id="list"),
    ],
)
def test_normalize_to_arrow_dtype_polars(pl_dtype, expected_pa):
    """Regression: polars dtypes normalize to arrow via
    a 0-row Series round-trip instead of crashing."""
    if isinstance(expected_pa, pa.Decimal128Type):
        # Exact precision/scale is only guaranteed on polars >= 1.x; legacy
        # 0.18.x (allowed for Python < 3.9) cannot preserve it, so the
        # declaration must be rejected with a clear error rather than
        # publish decimal128(38, 0).
        if parse_version(pl.__version__) < parse_version("1.0"):
            with pytest.raises(ValueError, match="cannot preserve"):
                _normalize_to_arrow_dtype(pl_dtype)
            return
    assert _normalize_to_arrow_dtype(pl_dtype) == wrap_arrow_dtype(expected_pa)


@skip_no_arrow_dtype
def test_normalize_to_arrow_dtype_accepts_advertised_arrow_decimal():
    """Regression: the legacy-polars Decimal rejection message advertises
    ``pd.ArrowDtype(pa.decimal128(...))`` as the remediation; the advertised
    form must be accepted by ``_normalize_to_arrow_dtype`` and return the
    same Arrow dtype (raw ``pa.DataType`` values remain a separate API
    decision).
    """
    advertised = pd.ArrowDtype(pa.decimal128(10, 2))
    normalized = _normalize_to_arrow_dtype(advertised)
    assert normalized == wrap_arrow_dtype(pa.decimal128(10, 2))
    assert normalized == advertised


@skip_no_arrow_dtype
def test_map_with_polars_dtype_compiles():
    """Regression: ``map(dtype=pl.Int64)`` normalizes
    the polars dtype to arrow instead of crashing."""
    lf = _make_liteframe({"a": wrap_arrow_dtype(pa.int64())})
    result = map_(lf, lambda x: x * 2, dtype=pl.Int64)
    assert isinstance(result.op, LiteFrameProjection)
    assert result.dtypes["a"] == wrap_arrow_dtype(pa.int64())


def test_build_mock_dataframe_with_blob():
    """build_mock_dataframe with ExternalBlobDtype column produces SolidBlob values."""
    dtypes = pd.Series(
        {
            "a": wrap_arrow_dtype(pa.int64()),
            "blob_col": ExternalBlobDtype(),
        }
    )
    mock_df = build_mock_dataframe(dtypes)
    assert mock_df.shape == (2, 2)
    assert isinstance(mock_df["blob_col"].dtype, ExternalBlobDtype)
    for val in mock_df["blob_col"]:
        assert isinstance(val, SolidBlob)


def test_infer_output_dtypes_with_blob_columns():
    """infer_output_dtypes does not crash when input includes blob columns."""
    input_dtypes = pd.Series(
        {
            "a": wrap_arrow_dtype(pa.int64()),
            "blob_col": ExternalBlobDtype(),
        }
    )

    def select_non_blob(df):
        return df[["a"]]

    result = infer_output_dtypes(
        func=select_non_blob,
        input_dtypes=input_dtypes,
        dtypes=None,
        skip_infer=False,
    )
    assert len(result) == 1
    assert list(result[0].index) == ["a"]


def test_infer_output_dtypes_blob_only_input():
    """infer_output_dtypes works when all input columns are blob type."""
    input_dtypes = pd.Series(
        {
            "blob_col": ExternalBlobDtype(),
            "val": wrap_arrow_dtype(pa.int64()),
        }
    )

    def drop_blob(df):
        return df[["val"]]

    result = infer_output_dtypes(
        func=drop_blob,
        input_dtypes=input_dtypes,
        dtypes=None,
        skip_infer=False,
    )
    assert len(result) == 1
    assert list(result[0].index) == ["val"]


@skip_no_arrow_dtype
def test_infer_output_dtypes_polars_mock_blob_input():
    """A polars UDF that opens a blob input must infer successfully.

    Regression: build_mock_dataframe built the polars mock through
    pl.from_pandas, which flattened blob columns to plain Binary, so a UDF
    calling ``df["b"][0].open("rb")`` raised TypeError and inference
    silently degraded.  The mock must use the blob-aware converter so blob
    columns carry AbstractExternalBlob, matching execution time.
    """
    input_dtypes = pd.Series(
        {
            "a": wrap_arrow_dtype(pa.int64()),
            "b": ExternalBlobDtype(),
        }
    )

    mock_df = build_mock_dataframe(input_dtypes, input_frame_type="polars")
    assert is_polars_blob_type(mock_df["b"].dtype)
    assert isinstance(mock_df["b"][0], AbstractExternalBlob)

    def open_blob_and_select_a(df):
        df["b"][0].open("rb")
        return df[["a"]]

    result = infer_output_dtypes(
        func=open_blob_and_select_a,
        input_dtypes=input_dtypes,
        dtypes=None,
        skip_infer=False,
        input_frame_type="polars",
    )
    assert len(result) == 1
    assert list(result[0].index) == ["a"]


def test_prepend_group_key_columns_per_output_missing():
    """Keys are prepended per output frame, not derived from frames[0] only.

    A multi-output flat_apply where output 0 carries a key-named column but
    output 1 does not: the prepend must still add the key to output 1.
    Deriving missing from frames[0] alone would skip the prepend for all
    outputs and let output 1 diverge from its declared schema.
    """
    from maxframe.liteframe.functional.utils import prepend_group_key_columns

    group_df = pd.DataFrame({"k": ["x", "x"], "v": [1, 2]})
    # output 0 already has k; output 1 does not.
    frames = [
        pd.DataFrame({"k": ["x", "x"], "a": [10, 20]}),
        pd.DataFrame({"b": [100, 200]}),
    ]
    declared = pd.Series(
        {"k": wrap_arrow_dtype(pa.string()), "a": wrap_arrow_dtype(pa.int64())}
    )
    out = prepend_group_key_columns(frames, group_df, declared, ["k"])
    assert "k" in out[0].columns and "a" in out[0].columns
    assert "k" in out[1].columns and "b" in out[1].columns
    assert out[1]["k"].tolist() == ["x", "x"]


# --- Callable-class inference: end=True lifecycle (comments #2/#4/#5) ---


@pytest.mark.parametrize(
    "mode,udf_cls,cols,expected_dtype",
    [
        # #2: map() inference called the constructor, not __call__.
        (
            "map",
            type("_UDF", (), {"__call__": lambda self, v, end=False: str(v)}),
            ["a"],
            pa.string(),
        ),
        # #4: transform_chunk inference didn't pass end=True.
        (
            "transform_chunk",
            type(
                "_UDF",
                (),
                {
                    "__call__": lambda self, df, end=False: df.astype(
                        "float64[pyarrow]"
                    )
                    if end
                    else None
                },
            ),
            ["a"],
            pa.float64(),
        ),
        # #5: apply(axis=0) per-column inference passed end=is_last.
        (
            "apply_axis0",
            type(
                "_UDF",
                (),
                {
                    "__call__": lambda self, col, end=False: col.astype(float)
                    if end
                    else col
                },
            ),
            ["a", "b"],
            pa.float64(),
        ),
    ],
)
def test_callable_class_inference_uses_end_true(mode, udf_cls, cols, expected_dtype):
    """Callable-class inference mirrors the runtime end=True lifecycle.

    Regression: map() called the constructor (not __call__), transform_chunk
    omitted end=True, and apply(axis=0) passed end=is_last — so inference
    diverged from runtime.  Now all three paths pass end=True.
    """
    lf = _make_liteframe({c: wrap_arrow_dtype(pa.int64()) for c in cols})
    result = (
        map_(lf, udf_cls)
        if mode == "map"
        else transform_chunk(lf, udf_cls)
        if mode == "transform_chunk"
        else apply(lf, udf_cls, axis=0)
    )
    expected = wrap_arrow_dtype(expected_dtype)
    for c in cols:
        assert (
            result.dtypes[c] == expected
        ), f"col {c}: {result.dtypes[c]} != {expected}"


def test_map_inference_preserves_datetime_for_plain_function():
    """map() with a plain-function datetime result infers a timestamp dtype.

    Regression (comment 241042637): replacing pd.Series([result_val]).dtype
    with type(result_val) regressed datetime inference — type(datetime)
    yields object via np.dtype.  Restoring pd.Series as the primary path
    with a string fallback preserves both datetime and string inference.
    """
    import datetime

    lf = _make_liteframe({"a": wrap_arrow_dtype(pa.int64())})
    result = map_(lf, lambda _: datetime.datetime(2026, 1, 2, 3, 4))
    # pandas maps datetime.datetime to datetime64[ns]; the arrow-equivalent
    # timestamp unit (ns/us) depends on the pyarrow version, so assert the
    # dtype is an arrow timestamp, not a specific unit.
    dt = result.dtypes["a"]
    pa_dt = getattr(dt, "pyarrow_dtype", None)
    assert pa_dt is not None and pa.types.is_timestamp(
        pa_dt
    ), f"expected arrow timestamp, got {dt}"


@pytest.mark.parametrize(
    "visible_cols,subset_cols,func_out_cols,expected",
    [
        pytest.param(
            ["a", "b", "c", "d"],
            ["c", "a"],
            ["x"],
            ["b", "x", "d"],
            id="reordered_subset",
        ),
        pytest.param(["a", "b"], ["a"], ["x"], ["x", "b"], id="contiguous_front"),
        pytest.param(["a", "b", "c"], ["a", "c"], None, ["b"], id="none_func_out"),
        pytest.param(
            ["a", "b"],
            [],
            ["x"],
            ["a", "b", "x"],
            id="empty_subset_appends_func_out",
        ),
    ],
)
def test_build_flat_apply_output_order(
    visible_cols, subset_cols, func_out_cols, expected
):
    """func-out cols inserted at first subset col's position; broadcast cols
    keep relative order.  subset=[] (anchorless) appends func-out after the
    broadcast cols (preserving the pre-MR caller-visible order).  func_out_cols
    =None returns broadcast cols only."""
    result = build_flat_apply_output_order(visible_cols, subset_cols, func_out_cols)
    assert result == expected
