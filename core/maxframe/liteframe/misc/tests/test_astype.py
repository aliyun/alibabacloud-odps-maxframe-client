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

from maxframe.lib.dtypes_extension.blob import ExternalBlobDtype
from maxframe.liteframe.core import RANGE_COL_NAME, LiteFrame
from maxframe.liteframe.expressions import (
    LiteFrameAdd,
    LiteFrameAstype,
    LiteFrameColumn,
    LiteFrameLiteral,
    NamedExpr,
)
from maxframe.liteframe.initializer import LiteFrame as LF
from maxframe.liteframe.operators.datasource import LiteFrameReadODPSTable
from maxframe.liteframe.operators.project import LiteFrameProjection
from maxframe.protocol import DefaultIndexType
from maxframe.utils import wrap_arrow_dtype


@pytest.mark.parametrize(
    "target_dtype_str, target_arrow_dtype",
    [
        ("float64", pa.float64()),
        ("float32", pa.float32()),
    ],
    ids=["float64", "float32"],
)
def test_astype_single_dtype(target_dtype_str, target_arrow_dtype):
    """Test astype with a single dtype applied to all columns."""
    lf = LF({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
    result = lf.astype(target_dtype_str)

    assert isinstance(result, LiteFrame)
    op = result.data.op
    assert isinstance(op, LiteFrameProjection)
    assert len(op.projections) == 2

    for proj in op.projections:
        assert isinstance(proj, NamedExpr)
        assert isinstance(proj.expr, LiteFrameAstype)
        assert proj.expr.target_dtype == wrap_arrow_dtype(target_arrow_dtype)
        assert isinstance(proj.expr.operand, LiteFrameColumn)

    assert result.dtypes["a"] == wrap_arrow_dtype(target_arrow_dtype)
    assert result.dtypes["b"] == wrap_arrow_dtype(target_arrow_dtype)


@pytest.mark.parametrize(
    "dtype_arg, expected_a_dtype, expected_b_unchanged",
    [
        ("float64", pa.float64(), False),
        ({"a": "float64"}, pa.float64(), True),
        ({"a": "int32"}, pa.int32(), True),
    ],
    ids=["single_float64", "dict_float64", "dict_int32"],
)
def test_astype_dtype_selection(dtype_arg, expected_a_dtype, expected_b_unchanged):
    """Test astype with single dtype and dict dtype, verifying column selection behavior."""
    lf = LF({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
    result = lf.astype(dtype_arg)

    assert isinstance(result, LiteFrame)
    op = result.data.op
    assert isinstance(op, LiteFrameProjection)

    a_proj = [p for p in op.projections if p.name == "a"][0]
    assert isinstance(a_proj.expr, LiteFrameAstype)
    assert a_proj.expr.target_dtype == wrap_arrow_dtype(expected_a_dtype)

    b_proj = [p for p in op.projections if p.name == "b"][0]
    if expected_b_unchanged:
        assert isinstance(b_proj, LiteFrameColumn)
    else:
        assert isinstance(b_proj.expr, LiteFrameAstype)

    assert result.dtypes["a"] == wrap_arrow_dtype(expected_a_dtype)
    if expected_b_unchanged:
        assert result.dtypes["b"] == lf.dtypes["b"]


# ---------------------------------------------------------------------------
# Fusion tests: astype on projection outputs should fuse to root source
# ---------------------------------------------------------------------------


def test_astype_after_arithmetic_fuses():
    """(lf + 5).astype("float64") should fuse to a single projection on root source."""
    lf = LF({"a": [1, 2, 3]})
    result = (lf + 5).astype("float64")

    op = result.data.op
    assert isinstance(op, LiteFrameProjection)
    expr = op.projections[0].expr
    assert isinstance(expr, LiteFrameAstype)
    assert isinstance(expr.operand, LiteFrameAdd)
    assert isinstance(expr.operand.left, LiteFrameColumn)
    assert expr.operand.left.name == "a"
    assert isinstance(expr.operand.right, LiteFrameLiteral)
    assert expr.operand.right.value == 5
    assert result.data.inputs[0] is lf._data


def test_astype_range_column_sets_non_affine_range_analysis():
    """astype on a range column should produce range_analysis with
    is_affine=False, references_range=True (since astype is not affine)."""
    dtypes = pd.Series(
        {
            "a": wrap_arrow_dtype(pa.int64()),
        }
    )
    op_src = LiteFrameReadODPSTable(
        table_name="t",
        dtypes=dtypes,
        default_index_type=DefaultIndexType.range,
    )
    lf = op_src._new_liteframe_from_source(shape=(10, 1))

    # Cast the range column — astype is not affine, so range_analysis
    # should mark it as non-affine with references_range=True
    casted = lf.astype({RANGE_COL_NAME: "int32"})
    op = casted.data.op
    assert isinstance(op, LiteFrameProjection)
    assert op.range_analysis is not None
    idx_result = op.range_analysis[RANGE_COL_NAME]
    assert idx_result.is_affine is False
    assert idx_result.references_range is True

    # A subsequent arithmetic op should fuse to the root source
    result = casted + 5
    op = result.data.op
    assert isinstance(op, LiteFrameProjection)
    assert result.data.inputs[0] is lf._data
    # range_analysis should still mark the range column as non-affine
    # after fusion with the arithmetic op
    assert op.range_analysis is not None
    fused_result = op.range_analysis[RANGE_COL_NAME]
    assert fused_result.is_affine is False
    assert fused_result.references_range is True


# ---------------------------------------------------------------------------
# Blob dtype tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dtype_arg,expected_dtype,use_isinstance",
    [
        pytest.param("blob", ExternalBlobDtype, True, id="blob_string"),
        pytest.param(
            "binary", wrap_arrow_dtype(pa.binary()), False, id="binary_string"
        ),
        pytest.param(bytes, wrap_arrow_dtype(pa.binary()), False, id="bytes_type"),
    ],
)
def test_astype_blob_or_binary(dtype_arg, expected_dtype, use_isinstance):
    """astype with 'blob'/'binary'/bytes produces the expected dtype per column."""
    lf = LF({"a": [1, 2, 3], "b": [4, 5, 6]})
    result = lf.astype(dtype_arg)

    assert isinstance(result, LiteFrame)
    op = result.data.op
    assert isinstance(op, LiteFrameProjection)

    for proj in op.projections:
        assert isinstance(proj, NamedExpr)
        assert isinstance(proj.expr, LiteFrameAstype)
        assert isinstance(proj.expr.operand, LiteFrameColumn)
        if use_isinstance:
            assert isinstance(proj.expr.target_dtype, expected_dtype)
        else:
            assert proj.expr.target_dtype == expected_dtype

    for col in ("a", "b"):
        if use_isinstance:
            assert isinstance(result.dtypes[col], expected_dtype)
        else:
            assert result.dtypes[col] == expected_dtype


def test_astype_dict_with_blob_and_binary():
    """Test astype with dict containing 'blob' and 'binary' for different columns."""
    lf = LF({"a": [1, 2, 3], "b": [4, 5, 6]})
    result = lf.astype({"a": "blob", "b": "binary"})

    assert isinstance(result, LiteFrame)
    op = result.data.op

    a_proj = [p for p in op.projections if p.name == "a"][0]
    assert isinstance(a_proj.expr, LiteFrameAstype)
    assert isinstance(a_proj.expr.target_dtype, ExternalBlobDtype)

    b_proj = [p for p in op.projections if p.name == "b"][0]
    assert isinstance(b_proj.expr, LiteFrameAstype)
    assert b_proj.expr.target_dtype == wrap_arrow_dtype(pa.binary())

    assert isinstance(result.dtypes["a"], ExternalBlobDtype)
    assert result.dtypes["b"] == wrap_arrow_dtype(pa.binary())


def test_astype_external_blob_dtype_passthrough():
    """Test that an ExternalBlobDtype instance passes through astype unchanged."""
    lf = LF({"a": [1, 2, 3]})
    blob_dtype = ExternalBlobDtype()
    result = lf.astype({"a": blob_dtype})

    op = result.data.op
    a_proj = [p for p in op.projections if p.name == "a"][0]
    assert isinstance(a_proj.expr, LiteFrameAstype)
    assert isinstance(a_proj.expr.target_dtype, ExternalBlobDtype)

    assert isinstance(result.dtypes["a"], ExternalBlobDtype)
