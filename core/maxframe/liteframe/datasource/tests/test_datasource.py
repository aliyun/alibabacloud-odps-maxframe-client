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

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from odps import ODPS
from odps import types as odps_types

from maxframe import opcodes
from maxframe.config import options
from maxframe.lib.dtypes_extension.blob import ExternalBlobDtype
from maxframe.liteframe.core import LITEFRAME_TYPE, RANGE_COL_NAME
from maxframe.liteframe.datasource.from_local import from_local_df
from maxframe.liteframe.datasource.read_csv import read_csv
from maxframe.liteframe.datasource.read_json import read_json
from maxframe.liteframe.datasource.read_lance import read_lance
from maxframe.liteframe.datasource.read_odps_table import read_odps_table
from maxframe.liteframe.datasource.read_parquet import read_parquet
from maxframe.liteframe.datatypes import is_blob_dtype, to_arrow_dtypes
from maxframe.liteframe.operators.core import LiteFrameDataFormat
from maxframe.liteframe.operators.datasource import (
    LiteFrameFromLocal,
    LiteFrameLakeDataSource,
    LiteFrameReadLance,
    LiteFrameReadODPSTable,
)
from maxframe.protocol import DefaultIndexType
from maxframe.tests.utils import tn
from maxframe.utils import wrap_arrow_dtype


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


@pytest.mark.parametrize(
    "lines, usecols, compression",
    [
        (True, None, "infer"),
        (True, ["a", "b"], "gzip"),
        (False, None, "infer"),
    ],
)
def test_read_json(lines, usecols, compression):
    kwargs = {"lines": lines, "compression": compression, "_local_test_mode": True}
    if usecols is not None:
        kwargs["usecols"] = usecols
    lf = read_json("/tmp/test.jsonl", **kwargs)
    assert isinstance(lf, LITEFRAME_TYPE)

    # Verify operator
    op = lf.data.op
    assert isinstance(op, LiteFrameLakeDataSource)
    assert op.data_format == LiteFrameDataFormat.JSON
    assert op.path == "/tmp/test.jsonl"
    assert op.read_kwargs["lines"] == lines
    assert op.columns == usecols
    assert op.read_kwargs["compression"] == compression
    assert op.chunk_bytes == "64M"
    assert op.merge_small_files is True
    assert op.read_kwargs["head_bytes"] == "100k"
    # Default: no frame_metadata with range columns
    assert lf.frame_metadata is None


def test_read_parquet_new_params():
    """Test that new params (chunk_bytes, merge_small_files, groups_as_chunks) work."""
    lf = read_parquet(
        "/tmp/test.parquet",
        chunk_bytes="128M",
        merge_small_files=False,
        groups_as_chunks=False,
        _local_test_mode=True,
    )
    assert isinstance(lf, LITEFRAME_TYPE)
    op = lf.data.op
    assert op.data_format == LiteFrameDataFormat.PARQUET
    assert op.chunk_bytes == "128M"
    assert op.merge_small_files is False
    assert op.groups_as_chunks is False


def test_read_csv_new_params():
    """Test that new params (chunk_bytes, merge_small_files, compression, head_bytes) work."""
    lf = read_csv(
        "/tmp/test.csv",
        chunk_bytes="32M",
        merge_small_files=False,
        compression="gzip",
        head_bytes="200k",
        _local_test_mode=True,
    )
    assert isinstance(lf, LITEFRAME_TYPE)
    op = lf.data.op
    assert op.data_format == LiteFrameDataFormat.CSV
    assert op.chunk_bytes == "32M"
    assert op.merge_small_files is False
    assert op.read_kwargs["compression"] == "gzip"
    assert op.read_kwargs["head_bytes"] == "200k"


def test_from_local_df_range_col_name_collision():
    """Source with a physical column named RANGE_COL_NAME should raise ValueError."""
    pdf = pd.DataFrame({RANGE_COL_NAME: [1, 2, 3], "a": [4, 5, 6]})
    with pytest.raises(ValueError, match="Column name collision"):
        from_local_df(pdf, default_index_type=DefaultIndexType.range)


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


@pytest.mark.parametrize(
    "initial_columns, pruned_columns, expected_columns",
    [
        pytest.param(None, ["col1"], ["col1"], id="plain_column_none_source"),
        pytest.param(None, ["pt"], ["pt"], id="partition_column_none_source"),
        pytest.param(["col1", "col2"], ["col1"], ["col1"], id="explicit_source"),
    ],
)
def test_read_odps_table_set_pruned_columns_none_source(
    initial_columns, pruned_columns, expected_columns
):
    """set_pruned_columns must not collapse a columns=None source to [].

    Regression: when the source was created without an explicit ``columns=``
    argument (``self.columns is None``), the pruning intersection
    ``[c for c in (self.columns or []) ...]`` produced ``[]``, which the ODPS
    tiler misreads as an explicit zero-column request.  This breaks any
    projection onto a columns=None source -- plain or partition column alike.
    """
    dtypes = pd.Series(
        {
            "col1": pd.api.types.pandas_dtype("int64"),
            "col2": pd.api.types.pandas_dtype("float64"),
            "pt": pd.api.types.pandas_dtype("object"),
        }
    )
    op = LiteFrameReadODPSTable(
        table_name="test_table",
        dtypes=dtypes,
        columns=initial_columns,
    )
    op.set_pruned_columns(pruned_columns)
    assert op.columns == expected_columns
    assert list(op.dtypes.index) == expected_columns


@pytest.mark.parametrize(
    "initial_columns, table_column_names, pruned_columns, "
    "expected_columns, expected_table_column_names",
    [
        pytest.param(
            None,
            ["_column_0", "_column_1", "c"],
            ["1", "c"],
            ["1", "c"],
            ["_column_1", "c"],
            id="non_sequential_prune_none_source",
        ),
        pytest.param(
            None,
            ["_column_0", "_column_1", "c"],
            ["0", "1"],
            ["0", "1"],
            ["_column_0", "_column_1"],
            id="suffix_prune_none_source",
        ),
        pytest.param(
            ["0", "1", "c"],
            ["_column_0", "_column_1", "c"],
            ["0", "c"],
            ["0", "c"],
            ["_column_0", "c"],
            id="explicit_source",
        ),
    ],
)
def test_read_odps_table_set_pruned_columns_table_column_names(
    initial_columns,
    table_column_names,
    pruned_columns,
    expected_columns,
    expected_table_column_names,
):
    """Pruned table_column_names stay positionally aligned with dtypes.index.

    Regression: the length-mismatch fallback took ``table_column_names[:N]``,
    assuming the surviving columns are a prefix of the table.  Non-sequential
    pruning (e.g. keeping indices [1, 2] of ["0", "1", "c"]) silently
    misaligned physical names with logical columns.  Each kept column must be
    mapped back through its position in the full column list.
    """
    dtypes = pd.Series(
        {
            "0": pd.api.types.pandas_dtype("int64"),
            "1": pd.api.types.pandas_dtype("float64"),
            "c": pd.api.types.pandas_dtype("object"),
        }
    )
    op = LiteFrameReadODPSTable(
        table_name="test_table",
        dtypes=dtypes,
        columns=initial_columns,
        table_column_names=table_column_names,
    )
    op.set_pruned_columns(pruned_columns)
    assert op.columns == expected_columns
    assert list(op.dtypes.index) == expected_columns
    assert op.table_column_names == expected_table_column_names


def test_read_lance_creates_operator():
    lf = read_lance("oss://endpoint/bucket/path", _local_test_mode=True)
    assert isinstance(lf, LITEFRAME_TYPE)

    op = lf.data.op
    assert isinstance(op, LiteFrameReadLance)
    assert op._op_type_ == opcodes.READ_LANCE
    assert op.path == "oss://endpoint/bucket/path"
    assert op.columns is None
    assert op.version is None
    assert op.asof is None
    assert op.filters is None
    assert op.with_row_address is False
    assert lf.frame_metadata is None


def test_read_lance_with_row_address():
    lf = read_lance("oss://endpoint/bucket/path", with_row_address=True)
    op = lf.data.op
    assert isinstance(op, LiteFrameReadLance)
    assert op.with_row_address is True
    assert op.hidden_columns == ["_rowaddr"]
    assert lf.frame_metadata is not None
    assert lf.frame_metadata.hidden_columns == ["_rowaddr"]


def test_read_lance_column_pruning():
    op = LiteFrameReadLance(
        path="/tmp/test.lance",
        dtypes=pd.Series(
            {"a": wrap_arrow_dtype(pa.int64()), "b": wrap_arrow_dtype(pa.string())}
        ),
        columns=["a", "b"],
    )
    op.set_pruned_columns(["a"])
    assert list(op.dtypes.index) == ["a"]
    assert op.columns == ["a"]


@pytest.mark.parametrize(
    "source_cls,dtypes,initial_columns,default_index,pruned_columns,"
    "expect_dtypes,expect_columns,expect_range",
    [
        pytest.param(
            LiteFrameLakeDataSource,
            {"a": pa.int64(), "b": pa.float64()},
            ["a", "b"],
            DefaultIndexType.range,
            ["a", RANGE_COL_NAME],
            ["a"],
            ["a"],
            DefaultIndexType.range,
            id="keep_range",
        ),
        pytest.param(
            LiteFrameLakeDataSource,
            {"a": pa.int64(), "b": pa.float64()},
            ["a", "b"],
            DefaultIndexType.range,
            ["a"],
            ["a"],
            ["a"],
            None,
            id="drop_range",
        ),
        pytest.param(
            LiteFrameLakeDataSource,
            {RANGE_COL_NAME: pa.int64(), "b": pa.float64()},
            [RANGE_COL_NAME, "b"],
            None,
            [RANGE_COL_NAME],
            [RANGE_COL_NAME],
            [RANGE_COL_NAME],
            None,
            id="physical_range_name",
        ),
        pytest.param(
            LiteFrameLakeDataSource,
            {"a": pa.int64(), "b": pa.float64()},
            None,
            DefaultIndexType.range,
            [RANGE_COL_NAME],
            [],
            [],
            DefaultIndexType.range,
            id="range_only_empty",
        ),
        pytest.param(
            LiteFrameReadLance,
            {"a": pa.int64(), "b": pa.float64()},
            ["a", "b"],
            DefaultIndexType.range,
            ["a", RANGE_COL_NAME],
            ["a"],
            ["a"],
            DefaultIndexType.range,
            id="lance_keep_range",
        ),
        pytest.param(
            LiteFrameReadLance,
            {"a": pa.int64(), "b": pa.float64()},
            ["a", "b"],
            DefaultIndexType.range,
            ["a"],
            ["a"],
            ["a"],
            None,
            id="lance_drop_range",
        ),
        pytest.param(
            LiteFrameReadLance,
            {RANGE_COL_NAME: pa.int64(), "b": pa.float64()},
            [RANGE_COL_NAME, "b"],
            None,
            [RANGE_COL_NAME],
            [RANGE_COL_NAME],
            [RANGE_COL_NAME],
            None,
            id="lance_physical_range_name",
        ),
        pytest.param(
            LiteFrameReadLance,
            {"a": pa.int64(), "b": pa.float64()},
            None,
            DefaultIndexType.range,
            [RANGE_COL_NAME],
            [],
            [],
            DefaultIndexType.range,
            id="lance_range_only_empty",
        ),
    ],
)
def test_lake_lance_set_pruned_columns(
    source_cls,
    dtypes,
    initial_columns,
    default_index,
    pruned_columns,
    expect_dtypes,
    expect_columns,
    expect_range,
):
    """set_pruned_columns handles the virtual range column and empty projections.

    keep_range/drop_range: RANGE_COL_NAME is virtual (not in dtypes), so it
    must not leak into dtypes; dropping it clears default_index_type.
    physical_range_name: with range off, a physical column named RANGE_COL_NAME
    survives pruning (the name is reserved only while range is enabled).
    range_only_empty: projecting only RANGE_COL_NAME onto a columns=None source
    materializes [] so the reader requests zero physical columns instead of
    restoring the full file width.
    """
    op = source_cls(
        path="/tmp/test",
        dtypes=pd.Series({k: wrap_arrow_dtype(v) for k, v in dtypes.items()}),
        columns=initial_columns,
        default_index_type=default_index,
    )
    op.set_pruned_columns(pruned_columns)
    assert list(op.dtypes.index) == expect_dtypes
    assert op.columns == expect_columns
    assert op.default_index_type == expect_range


def test_read_lance_validates_version_asof():
    with pytest.raises(ValueError, match="Cannot specify both"):
        read_lance("oss://endpoint/bucket/path", version=1, asof="2024-01-01")


def test_is_blob_dtype_external_blob():
    """is_blob_dtype returns True for ExternalBlobDtype."""
    assert is_blob_dtype(ExternalBlobDtype()) is True


def test_is_blob_dtype_non_blob():
    """is_blob_dtype returns False for non-blob dtypes."""
    assert is_blob_dtype(wrap_arrow_dtype(pa.int64())) is False
    assert is_blob_dtype(np.dtype("int64")) is False


def test_to_arrow_dtypes_preserves_blob_dtype():
    """ExternalBlobDtype is preserved through to_arrow_dtypes."""
    dtypes = pd.Series({"a": ExternalBlobDtype(), "b": wrap_arrow_dtype(pa.int64())})
    result = to_arrow_dtypes(dtypes)
    assert isinstance(result["a"], ExternalBlobDtype)
    assert result["b"] == wrap_arrow_dtype(pa.int64())
