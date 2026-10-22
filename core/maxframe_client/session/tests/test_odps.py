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

import io

import mock
import numpy as np
import pyarrow as pa
import pytest

import maxframe.tensor as mt
from maxframe.io.odpsio.schema import (
    arrow_schema_to_odps_schema,
    decode_blob_table_comment,
    encode_blob_table_comment,
)
from maxframe.lib.dtypes_extension.blob import ArrowBlobType
from maxframe.tests.utils import tn
from maxframe.utils import sync_pyodps_options
from maxframe_client.session.odps import MaxFrameSession

try:
    from odps.maxstorage import (
        MaxStorageClient,
        RowRangeInputSplit,
        SplitMode,
        SplitOptions,
    )
except ImportError:
    MaxStorageClient = None


@pytest.mark.parametrize(
    "shape,chunk_size",
    [
        ((10,), None),
        ((10,), 3),
        ((6, 4), None),
        ((6, 4), 3),
    ],
)
def test_upload_blob_table_tensor_nsplits(shape, chunk_size):
    """Test that _upload_and_get_blob_table_read_tileable handles None nsplits.

    When a tensor is created from local data (ArrayDataSource), nsplits is None
    because tiling hasn't happened yet. The upload path should treat the entire
    array as a single chunk per dimension.
    """
    raw = np.random.rand(*shape)
    t = mt.tensor(raw, chunk_size=chunk_size)
    assert t.nsplits is None

    mock_odps = mock.Mock()
    mock_table = mock.MagicMock()
    mock_table.odps = mock_odps
    mock_odps.create_table.return_value = mock_table
    mock_odps.get_table.return_value = mock_table

    # Mock MaxStorageClient via patching the module-level name
    mock_client = mock.MagicMock()
    mock_write_session = mock.MagicMock()
    mock_client.create_table_write_session.return_value = mock_write_session

    # Capture row batches written so we can verify idx columns and format
    written_batches = []

    class _FakeWriter:
        def write_batch(self, batch):
            written_batches.append(batch)

        def flush(self):
            pass

        def close(self):
            pass

    mock_write_session.open_arrow_writer.return_value = _FakeWriter()

    with mock.patch(
        "maxframe_client.session.odps.MaxStorageClient", return_value=mock_client
    ):
        session = MaxFrameSession.__new__(MaxFrameSession)
        session._odps_entry = mock_odps
        session._session_id = "test_session"

        session._upload_and_get_blob_table_read_tileable(t.data, raw)

    # nsplits should be set to single-chunk-per-dimension
    expected_nsplits = tuple((s,) for s in shape)
    assert t.data.nsplits == expected_nsplits

    # Table comment should contain the correct nsplits
    create_kwargs = mock_odps.create_table.call_args
    comment = create_kwargs.kwargs["comment"]
    meta = decode_blob_table_comment(comment)
    assert meta["nsplits"] == [list(sp) for sp in expected_nsplits]
    assert meta["shape"] == list(shape)

    # Blob tables require append delta hints and table properties
    hints = create_kwargs.kwargs["hints"]
    assert hints["odps.table.append2.enable"] is True
    table_props = create_kwargs.kwargs["table_properties"]
    assert table_props["table.format.version"] == "2"
    assert table_props["transactional"] == "true"

    # Single write session and writer
    assert mock_client.create_table_write_session.call_count == 1
    assert mock_write_session.open_arrow_writer.call_count == 1
    assert mock_write_session.commit.call_count == 1

    # Verify the actual blob content was written correctly.
    # With auto_upload_blobs, raw serialized bytes go directly into the
    # content column; verify they decode to the expected tensor.
    assert len(written_batches) >= 1
    row_table = pa.Table.from_batches(written_batches)
    assert row_table.num_rows == 1
    content_bytes = row_table.column("content")[0].as_py()
    buf = io.BytesIO(content_bytes)
    npz = np.load(buf, allow_pickle=False)
    np.testing.assert_array_equal(npz["data"], raw)

    # Verify idx columns + format
    ndim = len(shape)
    for i in range(ndim):
        col_name = f"idx_{i}"
        assert col_name in row_table.column_names
        assert row_table.column(col_name)[0].as_py() == 0
    assert row_table.column("format")[0].as_py() == "npz"


@pytest.mark.parametrize(
    "shape",
    [(10,), (6, 4)],
    ids=["1d", "2d"],
)
def test_write_blob_table_roundtrip(odps_account, shape):
    """Write a tensor to a blob table via _write_blob_table and read it back.

    Verifies that the data actually lands in ODPS and can be read back
    using the odps.maxstorage read path.
    """
    if MaxStorageClient is None:
        pytest.skip("odps.maxstorage is not available")

    odps_entry = odps_account
    raw = np.random.rand(*shape)

    # Serialize tensor data the same way _upload_and_get_blob_table_read_tileable does
    buf = io.BytesIO()
    np.savez(buf, data=raw, allow_pickle=False)
    pickled_bytes = buf.getvalue()

    ndim = len(shape)
    table_name = tn(f"mf_test_write_blob_roundtrip_{ndim}d")
    odps_entry.delete_table(table_name, if_exists=True)

    # Build table schema
    arrow_fields = []
    for i in range(ndim):
        arrow_fields.append(pa.field(f"idx_{i}", pa.int64()))
    arrow_fields.append(pa.field("format", pa.string()))
    arrow_fields.append(pa.field("content", ArrowBlobType()))
    arrow_schema = pa.schema(arrow_fields)
    odps_schema = arrow_schema_to_odps_schema(arrow_schema)

    comment = encode_blob_table_comment(
        {
            "dtype": str(raw.dtype),
            "ndim": ndim,
            "shape": list(shape),
            "nsplits": [list((s,)) for s in shape],
        }
    )

    with sync_pyodps_options():
        table = odps_entry.create_table(
            table_name,
            odps_schema,
            comment=comment,
            lifecycle=1,
            hints={
                "odps.sql.type.system.odps2": "true",
                "odps.table.append2.enable": "true",
            },
            table_properties={
                "table.format.version": "2",
                "transactional": "true",
            },
        )

    # Write data through _write_blob_table
    idx_arrays = {f"idx_{i}": [0] for i in range(ndim)}
    MaxFrameSession._write_blob_table(
        odps_entry,
        table,
        blob_bytes_list=[pickled_bytes],
        idx_arrays=idx_arrays,
        format_array=["npz"],
        ndim=ndim,
    )

    # Read back and verify
    with sync_pyodps_options():
        client = MaxStorageClient(odps_entry)

        read_session = client.create_table_read_session(
            table,
            split_options=SplitOptions(
                split_mode=SplitMode.ROW_OFFSET,
                split_number=100,
            ),
        )
        reader = read_session.open_arrow_reader(
            RowRangeInputSplit(read_session.id, 0, 100)
        )
        batches = []
        while True:
            batch = reader.read()
            if batch is None:
                break
            batches.append(batch)
        all_data = pa.Table.from_batches(batches) if batches else pa.table({})

    assert all_data.num_rows == 1
    blob_ref = all_data.column("content")[0].as_py()
    blob_manager = client.open_blob_manager(table)
    blob_iter = blob_manager.read_blobs([blob_ref])
    record = next(iter(blob_iter))
    blob_bytes = record.data
    read_buf = io.BytesIO(blob_bytes)
    npz = np.load(read_buf, allow_pickle=False)
    np.testing.assert_array_equal(npz["data"], raw)

    # Verify idx columns
    for i in range(ndim):
        assert all_data.column(f"idx_{i}")[0].as_py() == 0
    assert all_data.column("format")[0].as_py() == "npz"

    odps_entry.delete_table(table_name, if_exists=True, wait=False)
