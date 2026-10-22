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
import pyarrow as pa
import pytest

from maxframe.liteframe.arithmetic.range_affine import (
    RangeAnalysisResult,
    analyze_column,
    analyze_literal,
    analyze_range_affine,
    combine_add,
    combine_mul,
    combine_neg,
    combine_sub,
    transform_range_info,
)
from maxframe.liteframe.core import FrameMetadata, RangeInfo
from maxframe.liteframe.expressions import (
    LiteFrameAdd,
    LiteFrameColumn,
    LiteFrameDiv,
    LiteFrameFloorDiv,
    LiteFrameLiteral,
    LiteFrameMul,
    LiteFrameMultiColumnSubExpr,
    LiteFrameNeg,
    LiteFrameSub,
    LiteFrameUDFExpr,
    col,
)
from maxframe.liteframe.operators.datasource import LiteFrameReadODPSTable
from maxframe.liteframe.operators.project import LiteFrameProjection
from maxframe.utils import wrap_arrow_dtype

int64 = wrap_arrow_dtype(pa.int64())


def _rc(*names):
    return {n: RangeInfo(start=0, stop=10, step=1) for n in names}


def _rng_col(name="rng", dtype=None):
    return LiteFrameColumn(name=name, dtype=dtype or int64)


def _int_lit(val):
    return LiteFrameLiteral(value=val, dtype=int64)


# ---------------------------------------------------------------------------
# Integration fixture
# ---------------------------------------------------------------------------

RANGE_COL = "__row_id__"


def _make_range_lf(nrows=10):
    """Create a LiteFrame with a virtual range column named RANGE_COL.

    Builds the FrameMetadata directly so the test is not coupled to the
    source op's hardcoded range-column name.
    """
    dtypes = pd.Series({"a": int64})
    op_src = LiteFrameReadODPSTable(table_name="t", dtypes=dtypes)
    frame_metadata = FrameMetadata(
        range_columns={RANGE_COL: RangeInfo(start=0, stop=nrows, step=1)}
    )
    return op_src.new_liteframe(
        [], shape=(nrows, 2), physical_dtypes=dtypes, frame_metadata=frame_metadata
    )


# ---------------------------------------------------------------------------
# analyze_column
# ---------------------------------------------------------------------------


def test_analyze_column_range():
    result = analyze_column("rng", _rc("rng"))
    assert result.is_affine is True
    assert result.references_range is True
    assert result.scale == 1
    assert result.offset == 0
    assert result.col_name == "rng"


def test_analyze_column_non_range():
    result = analyze_column("a", _rc("rng"))
    assert result.is_affine is False
    assert result.references_range is False


# ---------------------------------------------------------------------------
# analyze_literal
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value, expected_offset",
    [(5, 5), (-3, -3), (0, 0)],
    ids=["positive", "negative", "zero"],
)
def test_analyze_literal_int(value, expected_offset):
    result = analyze_literal(value)
    assert result.is_affine is True
    assert result.references_range is False
    assert result.scale == 0
    assert result.offset == expected_offset


@pytest.mark.parametrize(
    "value", [1.5, "foo", True, None], ids=["float", "str", "bool", "none"]
)
def test_analyze_literal_non_int(value):
    result = analyze_literal(value)
    assert result.is_affine is False
    assert result.references_range is False


@pytest.mark.parametrize(
    "value, expected_offset",
    [(np.int64(5), 5), (np.int32(-3), -3), (np.uint16(7), 7)],
    ids=["int64", "int32", "uint16"],
)
def test_analyze_literal_numpy_int(value, expected_offset):
    result = analyze_literal(value)
    assert result.is_affine is True
    assert result.references_range is False
    assert result.scale == 0
    assert result.offset == expected_offset
    assert isinstance(result.offset, int)


# ---------------------------------------------------------------------------
# combine_add / combine_sub / combine_mul
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "left_fn, right_fn, expected_scale, expected_offset",
    [
        (lambda: analyze_column("rng", _rc("rng")), lambda: analyze_literal(5), 1, 5),
        (lambda: analyze_literal(5), lambda: analyze_column("rng", _rc("rng")), 1, 5),
    ],
    ids=["range_plus_int", "int_plus_range"],
)
def test_combine_add_range_and_int(left_fn, right_fn, expected_scale, expected_offset):
    result = combine_add(left_fn(), right_fn())
    assert result.is_affine is True
    assert result.scale == expected_scale
    assert result.offset == expected_offset
    assert result.col_name == "rng"


def test_combine_add_non_affine_propagates():
    rng = analyze_column("rng", _rc("rng"))
    non_affine = RangeAnalysisResult(is_affine=False, references_range=True)
    result = combine_add(rng, non_affine)
    assert result.is_affine is False
    assert result.references_range is True


@pytest.mark.parametrize(
    "left_fn, right_fn, expected_scale, expected_offset",
    [
        (lambda: analyze_column("rng", _rc("rng")), lambda: analyze_literal(3), 1, -3),
        (
            lambda: analyze_literal(10),
            lambda: analyze_column("rng", _rc("rng")),
            -1,
            10,
        ),
    ],
    ids=["range_minus_int", "int_minus_range"],
)
def test_combine_sub(left_fn, right_fn, expected_scale, expected_offset):
    result = combine_sub(left_fn(), right_fn())
    assert result.is_affine is True
    assert result.scale == expected_scale
    assert result.offset == expected_offset
    assert result.col_name == "rng"


def test_combine_sub_range_minus_range_same_col():
    rng = analyze_column("rng", _rc("rng"))
    result = combine_sub(rng, rng)
    assert result.is_affine is False
    assert result.references_range is True


@pytest.mark.parametrize(
    "left_fn, right_fn, expected_scale, expected_offset",
    [
        (lambda: analyze_column("rng", _rc("rng")), lambda: analyze_literal(2), 2, 0),
        (lambda: analyze_literal(3), lambda: analyze_column("rng", _rc("rng")), 3, 0),
    ],
    ids=["range_times_int", "int_times_range"],
)
def test_combine_mul(left_fn, right_fn, expected_scale, expected_offset):
    result = combine_mul(left_fn(), right_fn())
    assert result.is_affine is True
    assert result.scale == expected_scale
    assert result.offset == expected_offset
    assert result.col_name == "rng"


def test_combine_mul_two_range_columns():
    rng1 = analyze_column("rng1", _rc("rng1", "rng2"))
    rng2 = analyze_column("rng2", _rc("rng1", "rng2"))
    result = combine_mul(rng1, rng2)
    assert result.is_affine is False
    assert result.references_range is True


def test_combine_mul_two_constants():
    lit1 = analyze_literal(2)
    lit2 = analyze_literal(3)
    result = combine_mul(lit1, lit2)
    assert result.is_affine is False
    assert result.references_range is False


# ---------------------------------------------------------------------------
# combine_neg
# ---------------------------------------------------------------------------


def test_combine_neg_range():
    rng = analyze_column("rng", _rc("rng"))
    result = combine_neg(rng)
    assert result.is_affine is True
    assert result.scale == -1
    assert result.offset == 0
    assert result.col_name == "rng"


def test_combine_neg_constant():
    lit = analyze_literal(5)
    result = combine_neg(lit)
    assert result.is_affine is True
    assert result.references_range is False
    assert result.scale == 0
    assert result.offset == -5


# ---------------------------------------------------------------------------
# Two different range columns
# ---------------------------------------------------------------------------


def test_two_range_columns_add():
    rng1 = analyze_column("rng1", _rc("rng1", "rng2"))
    rng2 = analyze_column("rng2", _rc("rng1", "rng2"))
    result = combine_add(rng1, rng2)
    assert result.is_affine is False
    assert result.references_range is True


# ---------------------------------------------------------------------------
# analyze_range_affine (tree walk)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "expr_builder, expected_affine, expected_scale, expected_offset, expected_refs",
    [
        (lambda: _rng_col(), True, 1, 0, True),
        (
            lambda: LiteFrameAdd(left=_rng_col(), right=_int_lit(5), dtype=int64),
            True,
            1,
            5,
            True,
        ),
        (
            lambda: LiteFrameAdd(left=_int_lit(5), right=_rng_col(), dtype=int64),
            True,
            1,
            5,
            True,
        ),
        (
            lambda: LiteFrameSub(left=_rng_col(), right=_int_lit(3), dtype=int64),
            True,
            1,
            -3,
            True,
        ),
        (
            lambda: LiteFrameSub(left=_int_lit(10), right=_rng_col(), dtype=int64),
            True,
            -1,
            10,
            True,
        ),
        (
            lambda: LiteFrameMul(left=_rng_col(), right=_int_lit(2), dtype=int64),
            True,
            2,
            0,
            True,
        ),
        (
            lambda: LiteFrameMul(left=_int_lit(3), right=_rng_col(), dtype=int64),
            True,
            3,
            0,
            True,
        ),
        (lambda: LiteFrameNeg(operand=_rng_col(), dtype=int64), True, -1, 0, True),
    ],
    ids=[
        "column",
        "add",
        "scalar_left_add",
        "sub",
        "scalar_left_sub",
        "mul",
        "scalar_left_mul",
        "neg",
    ],
)
def test_analyze_range_affine_simple(
    expr_builder, expected_affine, expected_scale, expected_offset, expected_refs
):
    rc = {"rng": RangeInfo(start=0, stop=10, step=1)}
    result = analyze_range_affine(expr_builder(), rc)
    assert result.is_affine is expected_affine
    assert result.scale == expected_scale
    assert result.offset == expected_offset
    assert result.references_range is expected_refs


@pytest.mark.parametrize(
    "expr_builder, expected_scale, expected_offset",
    [
        # rng * 2 + 5
        (
            lambda: LiteFrameAdd(
                left=LiteFrameMul(left=_rng_col(), right=_int_lit(2), dtype=int64),
                right=_int_lit(5),
                dtype=int64,
            ),
            2,
            5,
        ),
        # (rng + 5) * 2
        (
            lambda: LiteFrameMul(
                left=LiteFrameAdd(left=_rng_col(), right=_int_lit(5), dtype=int64),
                right=_int_lit(2),
                dtype=int64,
            ),
            2,
            10,
        ),
        # 5 - rng + 3 = -rng + 8
        (
            lambda: LiteFrameAdd(
                left=LiteFrameSub(left=_int_lit(5), right=_rng_col(), dtype=int64),
                right=_int_lit(3),
                dtype=int64,
            ),
            -1,
            8,
        ),
    ],
    ids=["mul_then_add", "add_then_mul", "scalar_sub_then_add"],
)
def test_analyze_range_affine_chained(expr_builder, expected_scale, expected_offset):
    rc = {"rng": RangeInfo(start=0, stop=10, step=1)}
    result = analyze_range_affine(expr_builder(), rc)
    assert result.is_affine is True
    assert result.scale == expected_scale
    assert result.offset == expected_offset


def test_analyze_range_affine_non_int_literal():
    rc = {"rng": RangeInfo(start=0, stop=10, step=1)}
    expr = LiteFrameAdd(
        left=_rng_col(),
        right=LiteFrameLiteral(value=1.5, dtype=wrap_arrow_dtype(pa.float64())),
        dtype=wrap_arrow_dtype(pa.float64()),
    )
    result = analyze_range_affine(expr, rc)
    assert result.is_affine is False
    assert result.references_range is True


def test_analyze_range_affine_non_range_column():
    rc = {"rng": RangeInfo(start=0, stop=10, step=1)}
    expr = LiteFrameAdd(
        left=LiteFrameColumn(name="a", dtype=int64),
        right=_int_lit(5),
        dtype=int64,
    )
    result = analyze_range_affine(expr, rc)
    assert result.is_affine is False
    assert result.references_range is False


def test_analyze_range_affine_two_range_columns():
    rc = {
        "rng1": RangeInfo(start=0, stop=10, step=1),
        "rng2": RangeInfo(start=0, stop=10, step=1),
    }
    expr = LiteFrameAdd(
        left=LiteFrameColumn(name="rng1", dtype=int64),
        right=LiteFrameColumn(name="rng2", dtype=int64),
        dtype=int64,
    )
    result = analyze_range_affine(expr, rc)
    assert result.is_affine is False
    assert result.references_range is True


@pytest.mark.parametrize(
    "expr_cls", [LiteFrameDiv, LiteFrameFloorDiv], ids=["div", "floordiv"]
)
def test_analyze_range_affine_division_not_affine(expr_cls):
    rc = {"rng": RangeInfo(start=0, stop=10, step=1)}
    expr = expr_cls(left=_rng_col(), right=_int_lit(2), dtype=int64)
    result = analyze_range_affine(expr, rc)
    assert result.is_affine is False
    assert result.references_range is True


# ---------------------------------------------------------------------------
# transform_range_info
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "scale, offset, expected_start, expected_stop, expected_step",
    [
        (1, 5, 5, 15, 1),
        (2, 0, 0, 20, 2),
        (2, 5, 5, 25, 2),
        (-1, 10, 10, 0, -1),
    ],
    ids=["add_offset", "mul_scale", "combined", "neg"],
)
def test_transform_range_info(
    scale, offset, expected_start, expected_stop, expected_step
):
    info = RangeInfo(start=0, stop=10, step=1)
    result = transform_range_info(info, scale=scale, offset=offset)
    assert result.start == expected_start
    assert result.stop == expected_stop
    assert result.step == expected_step


def test_transform_range_info_start_none():
    info = RangeInfo(start=None, stop=10, step=1)
    result = transform_range_info(info, scale=2, offset=5)
    assert result.start is None
    assert result.stop == 25
    assert result.step == 2


def test_transform_range_info_stop_none():
    info = RangeInfo(start=0, stop=None, step=1)
    result = transform_range_info(info, scale=2, offset=5)
    assert result.start == 5
    assert result.stop is None
    assert result.step == 2


def test_transform_range_info_step_none():
    info = RangeInfo(start=0, stop=10, step=None)
    result = transform_range_info(info, scale=3, offset=0)
    assert result.start == 0
    assert result.stop == 30
    assert result.step == 3


# ---------------------------------------------------------------------------
# Integration: projection op wiring (range_analysis + frame_metadata)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "expr_builder, expected_scale, expected_offset, "
    "expected_start, expected_stop, expected_step",
    [
        (lambda: col(RANGE_COL) + 5, 1, 5, 5, 15, 1),
        (lambda: col(RANGE_COL) * 2, 2, 0, 0, 20, 2),
        (lambda: 10 - col(RANGE_COL), -1, 10, 10, 0, -1),
    ],
    ids=["add", "mul", "scalar_sub"],
)
def test_affine_projection_stays_virtual(
    expr_builder,
    expected_scale,
    expected_offset,
    expected_start,
    expected_stop,
    expected_step,
):
    """In-place affine transform keeps the range column virtual with
    transformed scale/offset and propagated (start, stop, step)."""
    lf = _make_range_lf(10)
    result = lf.assign(**{RANGE_COL: expr_builder()})
    op = result.data.op
    assert isinstance(op, LiteFrameProjection)
    ra = op.range_analysis[RANGE_COL]
    assert ra.is_affine is True
    assert ra.scale == expected_scale
    assert ra.offset == expected_offset
    assert ra.col_name == RANGE_COL
    info = result.frame_metadata.range_columns[RANGE_COL]
    assert info.start == expected_start
    assert info.stop == expected_stop
    assert info.step == expected_step


def test_chained_in_place_fuses():
    """lf.assign(rng=col+5).assign(rng=col*2) fuses to scale=2, offset=10."""
    lf = _make_range_lf(10)
    result = lf.assign(**{RANGE_COL: col(RANGE_COL) + 5}).assign(
        **{RANGE_COL: col(RANGE_COL) * 2}
    )
    op = result.data.op
    assert isinstance(op, LiteFrameProjection)
    assert result.data.inputs[0] is lf._data
    ra = op.range_analysis[RANGE_COL]
    assert ra.is_affine is True
    assert ra.scale == 2
    assert ra.offset == 10


def test_non_prefix_range_column_materializes():
    """A new range-affine column appended after physical columns must
    materialize all range columns (contiguous prefix rule)."""
    lf = _make_range_lf(10)
    result = lf.assign(new_idx=col(RANGE_COL) + 5)
    op = result.data.op
    ra = op.range_analysis["new_idx"]
    assert ra.is_affine is False
    assert ra.references_range is True
    ra_idx = op.range_analysis[RANGE_COL]
    assert ra_idx.is_affine is False
    assert ra_idx.references_range is True


def test_isna_on_range_materializes():
    lf = _make_range_lf(10)
    result = lf.isna()
    op = result.data.op
    ra = op.range_analysis[RANGE_COL]
    assert ra.is_affine is False
    assert ra.references_range is True
    assert not result.frame_metadata or not result.frame_metadata.range_columns


def test_astype_on_range_materializes():
    lf = _make_range_lf(10)
    result = lf.astype({RANGE_COL: "int32"})
    op = result.data.op
    ra = op.range_analysis[RANGE_COL]
    assert ra.is_affine is False
    assert ra.references_range is True


@pytest.mark.parametrize(
    "expr_builder, slc, expected_start, expected_stop, expected_step",
    [
        (lambda: col(RANGE_COL), slice(1, 3), 1, 3, 1),
        (lambda: col(RANGE_COL) * 2, slice(1, 3), 2, 6, 2),
        (lambda: 10 - col(RANGE_COL), slice(1, 3), 9, 7, -1),
    ],
    ids=["unit_step", "affine_mul", "affine_sub"],
)
def test_iloc_slice_propagates_affine_step(
    expr_builder, slc, expected_start, expected_stop, expected_step
):
    """iloc contiguous slice propagates the affine (start, stop, step)."""
    lf = _make_range_lf(10)
    transformed = lf.assign(**{RANGE_COL: expr_builder()})
    result = transformed.iloc[slc]
    info = result.frame_metadata.range_columns[RANGE_COL]
    assert info.start == expected_start
    assert info.stop == expected_stop
    assert info.step == expected_step


@pytest.mark.parametrize(
    "expr_builder, idx, expected_start, expected_stop, expected_step",
    [
        (lambda: col(RANGE_COL), 1, 1, 2, 1),
        (lambda: col(RANGE_COL) * 2, 1, 2, 4, 2),
        (lambda: 10 - col(RANGE_COL), 1, 9, 8, -1),
    ],
    ids=["int_unit_step", "int_affine_mul", "int_affine_sub"],
)
def test_iloc_int_propagates_affine_step(
    expr_builder, idx, expected_start, expected_stop, expected_step
):
    """iloc scalar int access propagates the affine (start, stop, step)."""
    lf = _make_range_lf(10)
    transformed = lf.assign(**{RANGE_COL: expr_builder()})
    result = transformed.iloc[idx]
    info = result.frame_metadata.range_columns[RANGE_COL]
    assert info.start == expected_start
    assert info.stop == expected_stop
    assert info.step == expected_step


def test_udf_expr_referencing_range_requests_materialization():
    """A map-mode UDF over a virtual range column is non-affine but must
    report ``references_range=True`` so the planner materializes the range.
    """
    rng_col = _rng_col(RANGE_COL).bind(_rc(RANGE_COL))
    udf = LiteFrameUDFExpr(
        func=lambda s: s,
        mode="map",
        operands=[rng_col],
        output_names=[RANGE_COL],
        dtypes={RANGE_COL: int64},
    )
    result = analyze_range_affine(udf, _rc(RANGE_COL))
    assert result.is_affine is False
    assert result.references_range is True


def test_udf_expr_not_referencing_range_is_safe():
    """A map-mode UDF over a non-range column must report
    ``references_range=False`` (no spurious materialization)."""
    other = LiteFrameColumn(name="a", dtype=int64).bind({"a": RangeInfo(0, 10, 1)})
    udf = LiteFrameUDFExpr(
        func=lambda s: s,
        mode="map",
        operands=[other],
        output_names=["a"],
        dtypes={"a": int64},
    )
    result = analyze_range_affine(udf, _rc(RANGE_COL))
    assert result.is_affine is False
    assert result.references_range is False


def test_multi_column_sub_expr_referencing_range_requests_materialization():
    """``LiteFrameMultiColumnSubExpr`` wrapping a UDF that reads the virtual
    range must propagate ``references_range=True``."""
    rng_col = _rng_col(RANGE_COL).bind(_rc(RANGE_COL))
    udf = LiteFrameUDFExpr(
        func=lambda s: s,
        mode="transform_chunk",
        operands=[rng_col],
        output_names=[RANGE_COL],
        dtypes={RANGE_COL: int64},
    )
    sub = LiteFrameMultiColumnSubExpr(
        multi_expr=udf, output_name=RANGE_COL, dtype=int64
    )
    result = analyze_range_affine(sub, _rc(RANGE_COL))
    assert result.is_affine is False
    assert result.references_range is True
