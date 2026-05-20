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

import os
import tempfile

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from maxframe.dataframe.datasource.read_lance import DataFrameReadLance, read_lance

# Skip all tests in this module if lance is not installed or not compatible
try:
    import lance
except (ImportError, AttributeError):  # pragma: no cover
    # lance not installed or not compatible
    lance = None
    pytestmark = pytest.mark.skip(reason="lance not installed or not compatible")


@pytest.fixture
def sample_df():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame(
        {
            "id": np.arange(20, dtype=np.int64),
            "value": np.random.rand(20),
            "category": ["A"] * 10 + ["B"] * 10,
        }
    )


@pytest.fixture
def lance_dataset(sample_df):
    """Create a temporary Lance dataset and return its path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_path = f"{tmpdir}/test_dataset"
        lance.write_dataset(sample_df, dataset_path)
        yield dataset_path, sample_df


@pytest.fixture
def versioned_lance_dataset():
    """Create a Lance dataset with multiple versions."""
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_path = f"{tmpdir}/versioned_dataset"

        # Write first version
        df_v1 = pd.DataFrame(
            {
                "id": np.arange(10, dtype=np.int64),
                "value": np.arange(10, dtype=np.int64),
            }
        )
        lance.write_dataset(df_v1, dataset_path)

        # Write second version (append)
        df_v2 = pd.DataFrame(
            {
                "id": np.arange(10, 20, dtype=np.int64),
                "value": np.arange(10, 20, dtype=np.int64),
            }
        )
        lance.write_dataset(df_v2, dataset_path, mode="append")

        yield dataset_path, df_v1, df_v2


def test_read_lance_validates_read_stage_and_version_args():
    result = DataFrameReadLance(path="oss://bucket/dataset", read_stage="get_dtypes")()
    assert result.dtype == np.dtype("O")

    with pytest.raises(ValueError, match="Cannot specify both"):
        read_lance("oss://bucket/dataset", version=1, asof="2024-01-01")


def test_read_lance_local_with_filter_and_columns(lance_dataset):
    """Test reading with both filter and columns in local mode."""
    dataset_path, _ = lance_dataset
    result = read_lance(dataset_path, columns=["id", "value"], filters="`id` >= 15")
    result_df = result.op.data

    assert len(result_df) == 5
    assert list(result_df.columns) == ["id", "value"]
    assert all(result_df["id"] >= 15)


def test_read_lance_local_filter_alias(lance_dataset):
    """Test that filter= is accepted as an alias of filters=."""
    dataset_path, _ = lance_dataset
    result = read_lance(dataset_path, columns=["id"], filter="`id` >= 18")
    result_df = result.op.data

    assert len(result_df) == 2
    assert list(result_df.columns) == ["id"]
    assert list(result_df["id"]) == [18, 19]


def test_normalize_lance_filters_passthrough():
    """Native Lance filter objects should pass through unchanged."""
    from maxframe.dataframe.datasource.read_lance import normalize_lance_filters

    marker = object()
    assert normalize_lance_filters(marker) is marker


def test_read_lance_local_version(versioned_lance_dataset):
    """Test reading specific version in local mode."""
    dataset_path, df_v1, df_v2 = versioned_lance_dataset

    # Verify we have 2 versions
    ds = lance.dataset(dataset_path)
    assert ds.version == 2

    # Read version 1 - should only have first 10 rows
    result_v1 = read_lance(dataset_path, version=1)
    result_v1_df = result_v1.op.data
    assert len(result_v1_df) == 10
    assert result_v1_df["id"].max() < 10

    # Read latest version - should have all 20 rows
    result_latest = read_lance(dataset_path)
    result_latest_df = result_latest.op.data
    assert len(result_latest_df) == 20


def test_read_lance_op_index_col():
    """Test DataFrameReadLance operator with index_col parameter."""
    # index_col with string
    op = DataFrameReadLance(
        path="oss://bucket/dataset",
        index_col="user_id",
    )
    assert op.index_col == "user_id"

    # index_col with int
    op_int = DataFrameReadLance(
        path="oss://bucket/dataset",
        index_col=0,
    )
    assert op_int.index_col == 0

    # index_col with list (MultiIndex)
    op_multi = DataFrameReadLance(
        path="oss://bucket/dataset",
        index_col=["region", "date"],
    )
    assert op_multi.index_col == ["region", "date"]

    # index_col=False (explicitly no index)
    op_false = DataFrameReadLance(
        path="oss://bucket/dataset",
        index_col=False,
    )
    assert op_false.index_col is False


def test_read_lance_local_index_col_string(lance_dataset):
    """Test reading with index_col as string column name."""
    dataset_path, sample_df = lance_dataset
    result = read_lance(dataset_path, index_col="id")
    result_df = result.op.data

    # Check that 'id' is now the index
    assert result_df.index.name == "id"
    assert "id" not in result_df.columns
    assert list(result_df.columns) == ["value", "category"]
    assert len(result_df) == 20


def test_read_lance_local_index_col_int(lance_dataset):
    """Test reading with index_col as column index."""
    dataset_path, sample_df = lance_dataset
    # First column (index 0) is 'id'
    result = read_lance(dataset_path, index_col=0)
    result_df = result.op.data

    # Check that 'id' (first column) is now the index
    assert result_df.index.name == "id"
    assert "id" not in result_df.columns
    assert len(result_df) == 20


def test_read_lance_local_index_col_list(lance_dataset):
    """Test reading with index_col as list (MultiIndex)."""
    dataset_path, sample_df = lance_dataset
    result = read_lance(dataset_path, index_col=["id", "category"])
    result_df = result.op.data

    # Check that MultiIndex is created
    assert isinstance(result_df.index, pd.MultiIndex)
    assert result_df.index.names == ["id", "category"]
    assert "id" not in result_df.columns
    assert "category" not in result_df.columns
    assert list(result_df.columns) == ["value"]


def test_read_lance_local_index_col_false(lance_dataset):
    """Test reading with index_col=False (don't use data column as index).

    index_col=False means "don't use any data column as index", but
    preserved index from pandas metadata should still be used.
    Since sample_df has RangeIndex, the result should also have RangeIndex.
    """
    dataset_path, sample_df = lance_dataset
    result = read_lance(dataset_path, index_col=False)
    result_df = result.op.data

    # index_col=False lets pandas metadata restore the original index
    # Since sample_df has RangeIndex, result should also have RangeIndex
    assert isinstance(result_df.index, pd.RangeIndex)
    # All original columns should be present (no column used as index)
    assert list(result_df.columns) == ["id", "value", "category"]


def test_read_lance_local_index_col_conflict():
    """Test that conflicting index_col raises clear error."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create dataset with named index (preserved in pandas metadata)
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        df = df.set_index("a")  # 'a' becomes the index
        df.index.name = "a"
        path = os.path.join(tmpdir, "indexed_dataset")
        lance.write_dataset(pa.Table.from_pandas(df), path)

        # Reading with index_col=None should work (uses preserved index)
        result = read_lance(path, index_col=None)
        assert result.op.data.index.name == "a"

        # Reading with matching index_col should work
        result2 = read_lance(path, index_col="a")
        assert result2.op.data.index.name == "a"

        # Reading with conflicting index_col should raise error
        with pytest.raises(ValueError) as exc_info:
            read_lance(path, index_col="b")
        assert "conflicts with preserved index" in str(exc_info.value)
        assert "['a']" in str(exc_info.value)


def test_read_lance_local_index_col_matches_preserved_multiindex():
    """When index_col matches preserved MultiIndex, should work without KeyError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create dataset with MultiIndex (preserved in pandas metadata)
        tuples = [("A", "X"), ("A", "Y"), ("B", "X")]
        idx = pd.MultiIndex.from_tuples(tuples, names=["first", "second"])
        df = pd.DataFrame({"value": [1, 2, 3]}, index=idx)
        path = os.path.join(tmpdir, "multiindex_dataset")
        lance.write_dataset(pa.Table.from_pandas(df), path)

        # Reading with matching index_col should work (no KeyError)
        result = read_lance(path, index_col=["first", "second"])
        result_df = result.op.data
        assert isinstance(result_df.index, pd.MultiIndex)
        assert list(result_df.index.names) == ["first", "second"]
        pd.testing.assert_frame_equal(result_df, df)


def test_read_lance_local_index_col_with_columns_success(lance_dataset):
    """Index_col works when required index column is kept in columns list."""
    dataset_path, _ = lance_dataset
    result = read_lance(dataset_path, index_col="id", columns=["id", "value"])
    df = result.op.data
    assert df.index.name == "id"
    assert list(df.columns) == ["value"]


def test_read_lance_local_index_col_with_columns_missing(lance_dataset):
    """Index_col missing from selected columns should raise a clear error."""
    dataset_path, _ = lance_dataset
    with pytest.raises(ValueError) as exc_info:
        read_lance(dataset_path, index_col="id", columns=["value", "category"])
    msg = str(exc_info.value)
    assert "index_col requires columns" in msg
    assert "id" in msg


def test_read_lance_index_col_invalid_type(lance_dataset):
    """index_col must be int/str or list/tuple; dict should raise."""
    dataset_path, _ = lance_dataset
    with pytest.raises(TypeError):
        read_lance(dataset_path, index_col={})


def test_read_lance_index_col_empty_sequence_uses_preserved(lance_dataset):
    """Empty list/tuple should be treated as None (use preserved index)."""
    dataset_path, _ = lance_dataset
    # write has RangeIndex; empty index_col should not fail
    result = read_lance(dataset_path, index_col=[])
    assert len(result.op.data) == 20


def test_read_lance_local_preserved_multiindex_restored():
    """When preserved_index is MultiIndex and index_col=None, restore it."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tuples = [("A", "X"), ("A", "Y"), ("B", "X")]
        index = pd.MultiIndex.from_tuples(tuples, names=["lvl0", "lvl1"])
        df = pd.DataFrame({"v1": [1, 2, 3], "v2": [4, 5, 6]}, index=index)
        path = os.path.join(tmpdir, "mi_dataset")
        lance.write_dataset(pa.Table.from_pandas(df), path)

        result = read_lance(path, index_col=None)
        out_df = result.op.data
        assert isinstance(out_df.index, pd.MultiIndex)
        assert list(out_df.index.names) == ["lvl0", "lvl1"]
        pd.testing.assert_frame_equal(out_df, df)
