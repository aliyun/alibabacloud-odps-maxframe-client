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

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from odps import ODPS
from odps import types as odps_types

from maxframe.config import options
from maxframe.liteframe.core import LITEFRAME_TYPE, RANGE_COL_NAME
from maxframe.liteframe.datasource.from_local import from_local_df
from maxframe.liteframe.datasource.read_odps_table import read_odps_table
from maxframe.liteframe.operators.source import (
    LiteFrameFromLocal,
    LiteFrameReadODPSTable,
)
from maxframe.protocol import DefaultIndexType
from maxframe.tests.utils import tn


def _make_mock_col(name, odps_type=None):
    """Create a mock column with a real ODPS type so schema conversion works."""
    col = MagicMock()
    col.name = name
    col.type = odps_type or odps_types.string
    return col


@pytest.mark.parametrize(
    "default_index_type, expected_shape, expected_columns, expect_range",
    [
        (None, (3, 2), ["a", "b"], False),
        (DefaultIndexType.range, (3, 3), [RANGE_COL_NAME, "a", "b"], True),
    ],
)
def test_from_local_df(
    default_index_type, expected_shape, expected_columns, expect_range
):
    pdf = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
    kwargs = {}
    if default_index_type is not None:
        kwargs["default_index_type"] = default_index_type
    lf = from_local_df(pdf, **kwargs)
    assert isinstance(lf, LITEFRAME_TYPE)
    assert lf.shape == expected_shape
    assert list(lf.columns) == expected_columns

    # Verify operator
    op = lf.data.op
    assert isinstance(op, LiteFrameFromLocal)
    assert op.data is pdf
    # op.dtypes contains only physical columns (range column is virtual)
    expected_physical = [c for c in expected_columns if c != RANGE_COL_NAME]
    assert len(op.dtypes) == len(expected_physical)

    if expect_range:
        assert lf.frame_metadata is not None
        assert RANGE_COL_NAME in lf.frame_metadata.range_columns
    else:
        assert lf.frame_metadata is None


def test_from_local_df_with_chunk_size():
    pdf = pd.DataFrame({"a": list(range(20))})
    lf = from_local_df(pdf, chunk_size=10)
    assert isinstance(lf, LITEFRAME_TYPE)
    # Default: no range columns
    assert lf.shape == (20, 1)

    # Verify chunk_size is stored in extra_params
    op = lf.data.op
    assert isinstance(op, LiteFrameFromLocal)
    assert op.data is pdf
    assert lf.data.extra_params.raw_chunk_size == 10
    assert len(op.dtypes) == 1


def test_from_local_df_without_chunk_size():
    pdf = pd.DataFrame({"a": list(range(20))})
    lf = from_local_df(pdf)
    assert isinstance(lf, LITEFRAME_TYPE)

    # Verify no raw_chunk_size in extra_params when not specified
    op = lf.data.op
    assert isinstance(op, LiteFrameFromLocal)
    assert lf.data.extra_params.get("raw_chunk_size", None) is None


@pytest.fixture
def odps_test_table():
    """Create a real ODPS test table for read_odps_table tests."""
    odps_entry = ODPS.from_environments()
    table_name = tn("test_liteframe_read_odps_table")
    odps_entry.delete_table(table_name, if_exists=True)
    test_table = odps_entry.create_table(
        table_name, "col1 string, col2 bigint, col3 double", lifecycle=1
    )
    yield test_table
    test_table.drop()


@pytest.fixture
def odps_partitioned_table():
    """Create a real ODPS partitioned test table."""
    odps_entry = ODPS.from_environments()
    table_name = tn("test_liteframe_partitioned_table")
    odps_entry.delete_table(table_name, if_exists=True)
    test_table = odps_entry.create_table(
        table_name,
        ("col1 string, col2 bigint", "pt string"),
        lifecycle=1,
    )
    yield test_table
    test_table.drop()


def test_read_odps_table(odps_test_table):
    lf = read_odps_table(odps_test_table.full_table_name)
    assert isinstance(lf, LITEFRAME_TYPE)

    # Verify operator
    op = lf.data.op
    assert isinstance(op, LiteFrameReadODPSTable)
    assert op.table_name == odps_test_table.full_table_name
    assert op.partitions is None
    assert op.columns is None
    # Default: no frame_metadata with range columns
    assert lf.frame_metadata is None


def test_read_odps_table_with_range(odps_test_table):
    lf = read_odps_table(
        odps_test_table.full_table_name, default_index_type=DefaultIndexType.range
    )
    assert isinstance(lf, LITEFRAME_TYPE)
    assert lf.frame_metadata is not None
    assert RANGE_COL_NAME in lf.frame_metadata.range_columns


def test_read_odps_table_with_partitions(odps_partitioned_table):
    lf = read_odps_table(
        odps_partitioned_table.full_table_name, partitions=["pt=20200101"]
    )
    assert isinstance(lf, LITEFRAME_TYPE)

    # Verify operator
    op = lf.data.op
    assert isinstance(op, LiteFrameReadODPSTable)
    assert op.table_name == odps_partitioned_table.full_table_name
    assert op.partitions == ["pt=20200101"]


def test_read_odps_table_schema_resolution():
    """Verify read_odps_table passes the resolved schema to get_table,
    aligning with DataFrame's schema resolution logic."""
    mock_col = _make_mock_col("col1")
    mock_table = MagicMock()
    mock_table.full_table_name = "test_project.test_table"
    mock_table.table_schema.columns = [mock_col]
    mock_table.table_schema.simple_columns = [mock_col]
    mock_table.table_schema.partitions = []
    mock_table.table_schema.names = ["col1"]

    mock_odps = MagicMock()
    mock_odps.get_table.return_value = mock_table
    mock_odps.schema = "default_project"

    with patch.object(ODPS, "from_global", return_value=mock_odps), patch.object(
        ODPS, "from_environments", return_value=None
    ):
        # Case 1: No default_schema set → falls back to odps_entry.schema
        options.session.default_schema = None
        read_odps_table("test_table")
        mock_odps.get_table.assert_called_with("test_table", schema="default_project")

        # Case 2: default_schema is explicitly set → uses it
        options.session.default_schema = "my_schema"
        try:
            read_odps_table("test_table")
            mock_odps.get_table.assert_called_with("test_table", schema="my_schema")
        finally:
            options.session.default_schema = None


def test_read_odps_table_missing_odps_entry():
    """Verify read_odps_table raises ValueError when no ODPS entry is available."""
    with patch.object(ODPS, "from_global", return_value=None), patch.object(
        ODPS, "from_environments", return_value=None
    ):
        with pytest.raises(ValueError, match="Missing odps_entry"):
            read_odps_table("test_table")


def test_read_odps_table_missing_columns():
    """Verify read_odps_table raises ValueError when requested columns
    don't exist in the table, instead of silently filtering them out."""
    mock_col = _make_mock_col("col1")

    mock_table = MagicMock()
    mock_table.full_table_name = "test_project.test_table"
    mock_table.table_schema.simple_columns = [mock_col]
    mock_table.table_schema.columns = [mock_col]
    mock_table.table_schema.partitions = []
    mock_table.table_schema.names = ["col1"]

    mock_odps = MagicMock()
    mock_odps.get_table.return_value = mock_table
    mock_odps.schema = "default_project"

    with patch.object(ODPS, "from_global", return_value=mock_odps), patch.object(
        ODPS, "from_environments", return_value=None
    ):
        # Single missing column
        with pytest.raises(ValueError, match="not found in ODPS table"):
            read_odps_table("test_table", columns=["col1", "missing_col"])

        # All columns missing
        with pytest.raises(ValueError, match="not found in ODPS table"):
            read_odps_table("test_table", columns=["nonexistent"])


def test_read_odps_table_string_partitions_normalized():
    """Verify that a single string partition is normalized to a list,
    preventing the DPE tiler from iterating over individual characters."""
    mock_col = _make_mock_col("col1")
    mock_pt = _make_mock_col("pt")

    mock_table = MagicMock()
    mock_table.full_table_name = "test_project.test_table"
    mock_table.table_schema.simple_columns = [mock_col]
    mock_table.table_schema.columns = [mock_col, mock_pt]
    mock_table.table_schema.partitions = [mock_pt]
    mock_table.table_schema.names = ["col1", "pt"]

    mock_odps = MagicMock()
    mock_odps.get_table.return_value = mock_table
    mock_odps.schema = "default_project"

    with patch.object(ODPS, "from_global", return_value=mock_odps), patch.object(
        ODPS, "from_environments", return_value=None
    ):
        lf = read_odps_table("test_table", partitions="pt=00")
        op = lf.data.op
        assert isinstance(op, LiteFrameReadODPSTable)
        assert op.partitions == ["pt=00"]
        assert isinstance(op.partitions, list)
