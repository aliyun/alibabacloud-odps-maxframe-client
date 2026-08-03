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

from maxframe.liteframe.core import RANGE_COL_NAME, FrameMetadata, LiteFrameData
from maxframe.liteframe.datasource.from_local import from_local_df
from maxframe.liteframe.expressions import LiteFrameColumn
from maxframe.liteframe.indexing.select import drop
from maxframe.liteframe.operators.groupby import LiteFrameGroupByOp
from maxframe.liteframe.operators.project import LiteFrameProjection
from maxframe.liteframe.operators.source import LiteFrameReadODPSTable
from maxframe.protocol import DefaultIndexType
from maxframe.utils import wrap_arrow_dtype


@pytest.mark.parametrize(
    "default_index_type, expect_range, expect_index_in_columns, expect_index_in_dtypes",
    [
        pytest.param(None, False, False, False, id="no_range_by_default"),
        pytest.param(
            DefaultIndexType.range,
            True,
            True,
            True,
            id="with_explicit_range",
        ),
    ],
)
def test_from_local_range_behavior(
    default_index_type,
    expect_range,
    expect_index_in_columns,
    expect_index_in_dtypes,
):
    pdf = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
    kwargs = {}
    if default_index_type is not None:
        kwargs["default_index_type"] = default_index_type
    lf = from_local_df(pdf, **kwargs)

    if expect_range:
        assert lf.frame_metadata is not None
        assert RANGE_COL_NAME in lf.frame_metadata.range_columns
        ri = lf.frame_metadata.range_columns[RANGE_COL_NAME]
        assert ri.start == 0
        assert ri.stop == 3
        assert ri.step == 1
    else:
        assert lf.frame_metadata is None
        assert RANGE_COL_NAME not in lf.dtypes.index

    if expect_index_in_columns:
        assert list(lf.columns) == [RANGE_COL_NAME, "a", "b"]
    else:
        assert list(lf.columns) == ["a", "b"]

    if expect_index_in_dtypes:
        assert RANGE_COL_NAME in lf.dtypes.index
        assert lf.dtypes[RANGE_COL_NAME] == wrap_arrow_dtype(pa.int64())
    else:
        assert RANGE_COL_NAME not in lf.dtypes.index


def test_physical_dtypes_and_columns_hidden():
    physical_dtypes = pd.Series(
        {
            "a": wrap_arrow_dtype(pa.int64()),
            "_hc": wrap_arrow_dtype(pa.float64()),
        }
    )
    fm = FrameMetadata(hidden_columns=["_hc"])
    data = LiteFrameData(
        shape=(3, 2), physical_dtypes=physical_dtypes, frame_metadata=fm
    )
    # _physical_dtypes includes hidden column
    assert "_hc" in data._physical_dtypes.index
    # columns and dtypes exclude hidden column
    assert "_hc" not in data.dtypes.index
    assert list(data.dtypes.index) == ["a"]
    assert list(data.columns) == ["a"]


def test_projection_derives_dtypes_from_projections():
    pdf = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
    lf = from_local_df(pdf)
    projections = [
        LiteFrameColumn(name="a", dtype=lf.dtypes["a"]),
    ]
    # dtypes param should NOT be needed
    op = LiteFrameProjection(projections=projections)
    result = op(lf)
    assert list(result.dtypes.index) == ["a"]


def test_projection_shape_with_range_column_at_position_0():
    """Range column at position 0 stays virtual; shape should not double-count it."""
    dtypes = pd.Series(
        {
            "a": wrap_arrow_dtype(pa.int64()),
            "b": wrap_arrow_dtype(pa.float64()),
        }
    )
    op_src = LiteFrameReadODPSTable(
        table_name="t",
        dtypes=dtypes,
        default_index_type=DefaultIndexType.range,
    )
    lf = op_src._new_liteframe_from_source(shape=(10, 2))
    # Source shape: 10 rows, 2 physical + 1 range = 3 columns
    assert lf.shape == (10, 3)

    # Project: __index__ (range) at position 0, plus "a"
    projections = [
        LiteFrameColumn(name=RANGE_COL_NAME, dtype=wrap_arrow_dtype(pa.int64())),
        LiteFrameColumn(name="a", dtype=wrap_arrow_dtype(pa.int64())),
    ]
    op_proj = LiteFrameProjection(projections=projections)
    result = op_proj(lf)

    # __index__ stays virtual (position 0), "a" is physical → shape = (10, 2)
    assert result.shape == (10, 2)
    # _physical_dtypes should only contain physical column "a"
    assert RANGE_COL_NAME not in result._physical_dtypes.index
    assert list(result._physical_dtypes.index) == ["a"]
    # User-visible dtypes should include both
    assert list(result.dtypes.index) == [RANGE_COL_NAME, "a"]
    # Range column still tracked in frame_metadata
    assert result.frame_metadata is not None
    assert RANGE_COL_NAME in result.frame_metadata.range_columns


def test_projection_shape_with_range_column_materialized():
    """Range column not at position 0 gets materialized; shape counts it once."""
    dtypes = pd.Series(
        {
            "a": wrap_arrow_dtype(pa.int64()),
            "b": wrap_arrow_dtype(pa.float64()),
        }
    )
    op_src = LiteFrameReadODPSTable(
        table_name="t",
        dtypes=dtypes,
        default_index_type=DefaultIndexType.range,
    )
    lf = op_src._new_liteframe_from_source(shape=(10, 2))

    # Project: "a" first, then __index__ (not position 0 → materialized)
    projections = [
        LiteFrameColumn(name="a", dtype=wrap_arrow_dtype(pa.int64())),
        LiteFrameColumn(name=RANGE_COL_NAME, dtype=wrap_arrow_dtype(pa.int64())),
    ]
    op_proj = LiteFrameProjection(projections=projections)
    result = op_proj(lf)

    # __index__ materialized → both are physical columns, no range, shape = (10, 2)
    assert result.shape == (10, 2)
    # _physical_dtypes contains both as physical
    assert list(result._physical_dtypes.index) == ["a", RANGE_COL_NAME]
    # No range columns in metadata
    assert result.frame_metadata is None or result.frame_metadata.range_columns is None


def test_hidden_columns_in_source():
    dtypes = pd.Series(
        {
            "a": wrap_arrow_dtype(pa.int64()),
            "_hc": wrap_arrow_dtype(pa.float64()),
        }
    )
    op = LiteFrameReadODPSTable(
        table_name="test_table",
        dtypes=dtypes,
        hidden_columns=["_hc"],
    )
    lf = op._new_liteframe_from_source(shape=(10, 2))
    # Hidden column not in visible columns/dtypes
    assert "_hc" not in lf.columns
    assert "_hc" not in lf.dtypes.index
    # Hidden column in _physical_dtypes and _hidden_columns
    assert "_hc" in lf._physical_dtypes.index
    assert "_hc" in lf._hidden_columns


@pytest.mark.parametrize(
    "pruned_columns, expect_hc_present",
    [
        pytest.param(["a", "_hc"], True, id="pruning_preserves_hidden"),
        pytest.param(["a"], False, id="pruning_drops_hidden"),
    ],
)
def test_column_pruning_hidden(pruned_columns, expect_hc_present):
    dtypes = pd.Series(
        {
            "a": wrap_arrow_dtype(pa.int64()),
            "b": wrap_arrow_dtype(pa.float64()),
            "_hc": wrap_arrow_dtype(pa.float64()),
        }
    )
    op = LiteFrameReadODPSTable(
        table_name="test_table",
        dtypes=dtypes,
        columns=["a", "b", "_hc"],
        hidden_columns=["_hc"],
    )
    op.set_pruned_columns(pruned_columns)
    if expect_hc_present:
        assert "_hc" in op.dtypes.index
        assert "_hc" in op.hidden_columns
    else:
        assert "_hc" not in op.dtypes.index
        assert "_hc" not in (op.hidden_columns or [])


@pytest.mark.parametrize(
    "pruned_columns, expect_range, expect_physical_cols",
    [
        pytest.param(
            ["a", RANGE_COL_NAME],
            True,
            ["a"],
            id="pruning_keeps_range",
        ),
        pytest.param(
            ["a"],
            False,
            ["a"],
            id="pruning_drops_range",
        ),
    ],
)
def test_column_pruning_range(pruned_columns, expect_range, expect_physical_cols):
    """set_pruned_columns should handle range column (__index__) correctly."""
    dtypes = pd.Series(
        {
            "a": wrap_arrow_dtype(pa.int64()),
            "b": wrap_arrow_dtype(pa.float64()),
        }
    )
    op = LiteFrameReadODPSTable(
        table_name="test_table",
        dtypes=dtypes,
        columns=["a", "b"],
        default_index_type=DefaultIndexType.range,
    )
    op.set_pruned_columns(pruned_columns)
    if expect_range:
        assert op.default_index_type == DefaultIndexType.range
    else:
        assert op.default_index_type is None
    assert list(op.dtypes.index) == expect_physical_cols


def _make_hidden_lf(pdf_data):
    """Helper: create a LiteFrame with hidden columns set up."""
    pdf = pd.DataFrame(pdf_data)
    hc_name = [c for c in pdf.columns if c.startswith("_hc")]
    lf = from_local_df(pdf)
    fm = FrameMetadata(hidden_columns=hc_name)
    lf._data.frame_metadata = fm
    lf._data.__dict__.pop("dtypes", None)
    lf._data.__dict__.pop("columns", None)
    return lf


@pytest.mark.parametrize(
    "operation",
    [
        pytest.param("filter", id="filter_preserves_hidden"),
        pytest.param("sort", id="sort_preserves_hidden"),
    ],
)
def test_filter_sort_preserves_hidden_columns(operation):
    if operation == "filter":
        lf = _make_hidden_lf({"a": [1, 2, 3], "_hc": [10.0, 20.0, 30.0]})
        result = lf[lf["a"] > 1]
    else:
        lf = _make_hidden_lf({"a": [3, 1, 2], "_hc": [30.0, 10.0, 20.0]})
        result = lf.sort_values("a")
    assert "_hc" in result._hidden_columns


@pytest.mark.parametrize(
    "operation, expected_check",
    [
        pytest.param(
            "agg",
            lambda r: len(r._hidden_columns) == 0,
            id="agg_drops_hidden",
        ),
        pytest.param(
            "groupby",
            lambda r: "_hc" in r._hidden_columns,
            id="groupby_preserves_hidden",
        ),
        pytest.param(
            "merge_suffixes",
            lambda r: len(r._hidden_columns) == 2,
            id="merge_suffixes_hidden",
        ),
        pytest.param(
            "to_odps",
            lambda r: len(r._hidden_columns) == 0,
            id="to_odps_drops_hidden",
        ),
    ],
)
def test_operation_hidden_columns_behavior(operation, expected_check):
    if operation == "agg":
        lf = _make_hidden_lf({"a": [1, 2, 3], "_hc": [10.0, 20.0, 30.0]})
        result = lf.agg("sum")
    elif operation == "groupby":
        lf = _make_hidden_lf(
            {"a": [1, 1, 2], "b": [4, 5, 6], "_hc": [10.0, 20.0, 30.0]}
        )
        op = LiteFrameGroupByOp(
            groupby_params={"by": "a", "sort": False, "dropna": True}
        )
        result = op(lf)
    elif operation == "merge_suffixes":
        pdf1 = pd.DataFrame({"key": [1, 2], "_hc": [10.0, 20.0]})
        pdf2 = pd.DataFrame({"key": [2, 3], "_hc": [30.0, 40.0]})
        left = from_local_df(pdf1)
        right = from_local_df(pdf2)
        left._data.frame_metadata = FrameMetadata(hidden_columns=["_hc"])
        left._data.__dict__.pop("dtypes", None)
        left._data.__dict__.pop("columns", None)
        right._data.frame_metadata = FrameMetadata(hidden_columns=["_hc"])
        right._data.__dict__.pop("dtypes", None)
        right._data.__dict__.pop("columns", None)
        result = left.merge(right, on="key")
    elif operation == "to_odps":
        lf = _make_hidden_lf({"a": [1, 2, 3], "_hc": [10.0, 20.0, 30.0]})
        result = drop(lf, ["_hc"])

    assert expected_check(result)


def test_hidden_columns_through_pipeline():
    """Hidden columns survive through filter, sort, arithmetic, and projection."""
    lf = _make_hidden_lf(
        {"a": [3, 1, 2], "b": [4.0, 5.0, 6.0], "_hc": [30.0, 10.0, 20.0]}
    )

    # Arithmetic should preserve hidden columns
    result = lf + 1
    assert "_hc" in result._hidden_columns

    # Filter should preserve hidden columns
    filtered = lf[lf["a"] > 1]
    assert "_hc" in filtered._hidden_columns

    # Sort should preserve hidden columns
    sorted_lf = lf.sort_values("a")
    assert "_hc" in sorted_lf._hidden_columns

    # Select visible columns should preserve hidden columns
    selected = lf[["a"]]
    assert "_hc" in selected._hidden_columns

    # Drop hidden column
    dropped = drop(lf, ["_hc"])
    assert len(dropped._hidden_columns) == 0


def test_hidden_columns_not_in_user_api():
    """Hidden columns are invisible in columns and dtypes."""
    lf = _make_hidden_lf({"a": [1, 2, 3], "_hc": [10.0, 20.0, 30.0]})

    assert "_hc" not in lf.columns
    assert "_hc" not in lf.dtypes.index
    assert "_hc" in lf._physical_dtypes.index
    assert "_hc" in lf._hidden_columns
