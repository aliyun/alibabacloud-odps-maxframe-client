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
import uuid

import mock
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from odps import ODPS
from odps.errors import ODPSError

import maxframe.dataframe as md
import maxframe.tensor as mt
from maxframe.config import options
from maxframe.io.odpsio import ODPSTableIO
from maxframe.io.odpsio.schema import (
    arrow_schema_to_odps_schema,
    encode_blob_table_comment,
)
from maxframe.lib import wrapped_pickle as pickle
from maxframe.lib.dtypes_extension.blob import ArrowBlobType
from maxframe.protocol import (
    LiteFrameTableMeta,
    ODPSTableResultInfo,
    ResultType,
    TensorTableMeta,
)
from maxframe.tests.utils import get_test_unique_name, tn
from maxframe.utils import call_with_retry, sync_pyodps_options
from maxframe_client.fetcher import ODPSTableFetcher

try:
    from odps.maxstorage import MaxStorageClient
except ImportError:
    MaxStorageClient = None

requires_storage_api = pytest.mark.skipif(
    MaxStorageClient is None,
    reason="Requires odps.maxstorage (PyODPS >= 0.13.0)",
)


@pytest.fixture
def switch_table_io(request):
    old_use_common_table = options.use_common_table
    try:
        options.use_common_table = request.param
        yield
    finally:
        options.use_common_table = old_use_common_table


@pytest.mark.parametrize("switch_table_io", [False, True], indirect=True)
async def test_fetch_table_data(switch_table_io):
    odps_entry = ODPS.from_environments()
    table_io = ODPSTableIO(odps_entry)
    fetcher = ODPSTableFetcher(odps_entry)

    data = pd.DataFrame(
        {
            "_idx_0": np.arange(1000),
            "a": np.random.rand(1000),
            "b": np.random.randint(0, 10, 1000),
            "c": np.random.choice(list("ABC"), 1000),
        }
    )

    table_name = tn("mf_test_fetch_table_data_" + str(uuid.uuid4().hex))
    odps_entry.delete_table(table_name, if_exists=True)

    odps_entry.create_table(
        table_name, "_idx_0 bigint, a double, b bigint, c string", lifecycle=1
    )
    with table_io.open_writer(table_name) as writer:
        writer.write(pa.Table.from_pandas(data, preserve_index=False))

    raw_data = data[list("abc")]

    tileable = md.read_pandas(data[list("abc")])
    result_info = ODPSTableResultInfo(ResultType.ODPS_TABLE, full_table_name=table_name)
    fetched = await fetcher.fetch(tileable, result_info, [None, None])
    assert len(fetched) == 1000
    pd.testing.assert_frame_equal(raw_data, fetched)

    result_info = ODPSTableResultInfo(ResultType.ODPS_TABLE, full_table_name=table_name)
    fetched = await fetcher.fetch(tileable, result_info, [slice(None, 2000), None])
    assert len(fetched) == 1000
    pd.testing.assert_frame_equal(raw_data, fetched)

    result_info = ODPSTableResultInfo(ResultType.ODPS_TABLE, full_table_name=table_name)
    fetched = await fetcher.fetch(tileable, result_info, [2, None])
    assert len(fetched) == 1
    pd.testing.assert_frame_equal(raw_data.iloc[2:3], fetched)

    result_info = ODPSTableResultInfo(ResultType.ODPS_TABLE, full_table_name=table_name)
    fetched = await fetcher.fetch(tileable, result_info, [slice(None, 10), None])
    assert len(fetched) == 10
    pd.testing.assert_frame_equal(raw_data.iloc[:10], fetched)

    result_info = ODPSTableResultInfo(ResultType.ODPS_TABLE, full_table_name=table_name)
    fetched = await fetcher.fetch(tileable, result_info, [slice(None, 10, 3), None])
    assert len(fetched) == 4
    pd.testing.assert_frame_equal(raw_data.iloc[:10:3], fetched)

    tileable = md.read_pandas(data[list("ab")])
    result_info = ODPSTableResultInfo(ResultType.ODPS_TABLE, full_table_name=table_name)
    fetched = await fetcher.fetch(tileable, result_info, [slice(-5, None), slice(2)])
    assert len(fetched) == 5
    pd.testing.assert_frame_equal(raw_data.iloc[-5:, :2], fetched)

    tileable = md.read_pandas(data[list("a")])
    result_info = ODPSTableResultInfo(ResultType.ODPS_TABLE, full_table_name=table_name)
    fetched = await fetcher.fetch(tileable, result_info, [slice(-1, -6, -1), 0])
    assert len(fetched) == 5
    assert len(fetched.columns) == 1
    pd.testing.assert_frame_equal(raw_data.iloc[-1:-6:-1, :1], fetched)

    odps_entry.delete_table(table_name, if_exists=True)


@pytest.mark.parametrize("switch_table_io", [False, True], indirect=True)
async def test_fetch_int_cols_with_nones(switch_table_io):
    odps_entry = ODPS.from_environments()
    table_io = ODPSTableIO(odps_entry)
    fetcher = ODPSTableFetcher(odps_entry)

    data = pd.DataFrame(
        {
            "_idx_0": np.arange(1000),
            "a": np.random.randint(0, 10, 1000),
            "b": np.random.choice(list("ABC"), 1000),
        }
    )

    table_name = tn("mf_test_fetch_int_cols_with_nones_" + str(uuid.uuid4().hex))
    odps_entry.delete_table(table_name, if_exists=True)

    odps_entry.create_table(
        table_name, "_idx_0 bigint, a bigint, b string", lifecycle=1
    )
    new_data = data.copy()
    new_data.iloc[10, 1] = None
    new_data = new_data.astype({"a": pd.Int64Dtype()})
    with table_io.open_writer(table_name) as writer:
        writer.write(pa.Table.from_pandas(new_data, preserve_index=False))

    tileable = md.DataFrame(data[list("ab")])
    result_info = ODPSTableResultInfo(ResultType.ODPS_TABLE, full_table_name=table_name)
    fetched = await fetcher.fetch(tileable, result_info, [None, None])
    assert len(fetched) == 1000
    pd.testing.assert_frame_equal(
        new_data.drop("_idx_0", axis=1), fetched, check_dtype=False
    )

    odps_entry.delete_table(table_name, if_exists=True)


async def test_fetch_data_shape():
    odps_entry = ODPS.from_environments()
    table_io = ODPSTableIO(odps_entry)
    fetcher = ODPSTableFetcher(odps_entry)

    data = pd.DataFrame(
        {
            "_idx_0": np.arange(1000),
            "a": np.random.rand(1000),
            "b": np.random.randint(0, 10, 1000),
            "c": np.random.choice(list("ABC"), 1000),
        }
    )

    table_names = [
        tn("mf_test_groupby_table_" + str(uuid.uuid4().hex)) for _ in range(3)
    ]
    pt_specs = [None, None, ["pt='2020-01-01'"]]
    for tn_to_check in table_names:
        odps_entry.delete_table(tn_to_check, if_exists=True)

    odps_entry.create_table(
        table_names[0], "_idx_0 bigint, a double, b bigint, c string", lifecycle=1
    )
    odps_entry.create_table(
        table_names[1], "_idx_0 bigint, a double, b bigint, c string", lifecycle=1
    )
    odps_entry.create_table(
        table_names[2],
        ("_idx_0 bigint, a double, b bigint, c string", "pt string"),
        lifecycle=1,
    )
    with table_io.open_writer(table_names[0]) as writer:
        writer.write(pa.Table.from_pandas(data, preserve_index=False))

    odps_entry.execute_sql(
        f"insert overwrite table {table_names[1]} select * from {table_names[0]}"
    )
    odps_entry.execute_sql(
        f"insert overwrite table {table_names[2]} partition ({pt_specs[-1][0]}) "
        f"select * from {table_names[0]}"
    )

    for tn_to_check, pt_spec in zip(table_names, pt_specs):
        tileable = md.read_pandas(data[list("abc")])
        tileable._shape = (np.nan, 3)
        result_info = ODPSTableResultInfo(
            ResultType.ODPS_TABLE, full_table_name=tn_to_check, partition_specs=pt_spec
        )
        await fetcher.update_tileable_meta(tileable, result_info)
        assert tileable.shape == (1000, 3)

    def _create_session_with_error(*_, **__):
        raise ODPSError("StatusConflict: mock")

    with mock.patch(
        "maxframe.io.odpsio.tableio.TunnelTableIO.create_download_sessions",
        new=_create_session_with_error,
    ):
        tileable = md.read_pandas(data[list("abc")])
        tileable._shape = (np.nan, 3)
        result_info = ODPSTableResultInfo(
            ResultType.ODPS_TABLE, full_table_name=table_names[0]
        )
        await fetcher.update_tileable_meta(tileable, result_info)
        assert tileable.shape == (np.nan, 3)

    for tn_to_check in table_names:
        odps_entry.delete_table(tn_to_check, if_exists=True, wait=False)


def _write_tensor_blob_table(odps_entry, table_name, raw, nsplits):
    """Write a tensor into an ODPS blob table using odps.maxstorage.

    Uses "npz" format for numerical dtypes and "pickle" for object dtypes.
    """
    ndim = raw.ndim
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
            "shape": list(raw.shape),
            "nsplits": [list(sp) for sp in nsplits],
        }
    )

    use_npz = raw.dtype != object and not raw.dtype.names

    with sync_pyodps_options():
        odps_entry.delete_table(table_name, if_exists=True)
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

        client = MaxStorageClient(odps_entry)
        write_session = client.create_table_write_session(table)

        stream_id = "stream-1"
        stream_version = 1
        writer = write_session.open_arrow_writer(
            stream_id,
            stream_version,
            auto_upload_blobs=True,
        )

        # Serialize chunks: raw bytes go into the content column; the
        # writer auto-uploads blob cells and replaces them with refs.
        idx_arrays = {f"idx_{i}": [] for i in range(ndim)}
        format_array = []
        content_array = []

        for chunk_idx in np.ndindex(*[len(sp) for sp in nsplits]):
            slices = []
            for dim, idx in enumerate(chunk_idx):
                start = sum(nsplits[dim][:idx])
                end = start + nsplits[dim][idx]
                slices.append(slice(start, end))

            chunk_data = raw[tuple(slices)]
            buf = io.BytesIO()
            if use_npz:
                np.savez(buf, data=chunk_data, allow_pickle=False)
                fmt = "npz"
            else:
                pickle.dump(chunk_data, buf)
                fmt = "pickle"
            content_array.append(buf.getvalue())

            for i in range(ndim):
                idx_arrays[f"idx_{i}"].append(int(chunk_idx[i]))
            format_array.append(fmt)

        arrow_table = pa.table(
            {
                **{
                    f"idx_{i}": pa.array(idx_arrays[f"idx_{i}"], type=pa.int64())
                    for i in range(ndim)
                },
                "format": pa.array(format_array, type=pa.string()),
                "content": pa.array(content_array, type=pa.binary()),
            }
        )

        for batch in arrow_table.to_batches():
            writer.write_batch(batch)
        writer.flush()
        writer.close()
        write_session.commit(
            stream_ids=[stream_id],
            stream_versions=[stream_version],
        )


def _write_object_blob_table(odps_entry, table_name, value):
    """Write an object into an ODPS blob table using odps.maxstorage."""

    arrow_fields = [
        pa.field("format", pa.string()),
        pa.field("content", ArrowBlobType()),
    ]
    arrow_schema = pa.schema(arrow_fields)
    odps_schema = arrow_schema_to_odps_schema(arrow_schema)

    buf = io.BytesIO()
    pickle.dump(value, buf)
    pickled_bytes = buf.getvalue()

    with sync_pyodps_options():
        odps_entry.delete_table(table_name, if_exists=True)
        table = odps_entry.create_table(
            table_name,
            odps_schema,
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

        client = MaxStorageClient(odps_entry)
        write_session = client.create_table_write_session(table)

        stream_id = "stream-1"
        stream_version = 0

        def _upload_batch():
            nonlocal stream_version
            stream_version += 1
            writer = write_session.open_arrow_writer(
                stream_id,
                stream_version,
                auto_upload_blobs=True,
            )

            arrow_table = pa.table(
                {
                    "format": pa.array(["pickle"], type=pa.string()),
                    "content": pa.array([pickled_bytes], type=pa.binary()),
                }
            )

            for batch in arrow_table.to_batches():
                writer.write_batch(batch)
            writer.flush()
            writer.close()
            return stream_version

        stream_version = call_with_retry(_upload_batch)
        write_session.commit(
            stream_ids=[stream_id],
            stream_versions=[stream_version],
        )


@requires_storage_api
@pytest.mark.parametrize(
    "dtype",
    [np.float64, object],
    ids=["numerical", "object"],
)
async def test_fetch_blob_table_tensor_1d(dtype, odps_account):
    odps_entry = odps_account
    fetcher = ODPSTableFetcher(odps_entry)

    if dtype == object:
        raw = np.array([f"val_{i}" for i in range(10)], dtype=object)
    else:
        raw = np.random.rand(10).astype(dtype)
    nsplits = ((5, 5),)

    table_name = tn(
        f"mf_test_fetch_blob_tensor_1d_{dtype.__name__}_{get_test_unique_name(5)}"
    )
    odps_entry.delete_table(table_name, if_exists=True)
    _write_tensor_blob_table(odps_entry, table_name, raw, nsplits)

    tileable = mt.tensor(raw, chunk_size=5)
    table_meta = TensorTableMeta(
        table_name=table_name,
        dtype=str(raw.dtype),
        ndim=1,
        shape=list(raw.shape),
        nsplits=[list(sp) for sp in nsplits],
    )
    result_info = ODPSTableResultInfo(
        ResultType.ODPS_TABLE,
        full_table_name=table_name,
        table_meta=table_meta,
    )

    # Full fetch
    fetched = await fetcher.fetch(tileable, result_info, [None])
    np.testing.assert_array_equal(fetched, raw)

    # Slice: [2:8]
    fetched = await fetcher.fetch(tileable, result_info, [slice(2, 8)])
    np.testing.assert_array_equal(fetched, raw[2:8])

    # Integer index: [5]
    fetched = await fetcher.fetch(tileable, result_info, [5])
    np.testing.assert_array_equal(fetched, raw[5])

    odps_entry.delete_table(table_name, if_exists=True, wait=False)


@requires_storage_api
@pytest.mark.parametrize(
    "dtype",
    [np.float64, object],
    ids=["numerical", "object"],
)
async def test_fetch_blob_table_tensor_2d(dtype, odps_account):
    odps_entry = odps_account
    fetcher = ODPSTableFetcher(odps_entry)

    if dtype == object:
        raw = np.array([f"val_{i}" for i in range(24)], dtype=object).reshape(6, 4)
    else:
        raw = np.random.rand(6, 4).astype(dtype)
    nsplits = ((3, 3), (3, 1))

    table_name = tn(
        f"mf_test_fetch_blob_tensor_2d_{dtype.__name__}_{get_test_unique_name(5)}"
    )
    odps_entry.delete_table(table_name, if_exists=True)
    _write_tensor_blob_table(odps_entry, table_name, raw, nsplits)

    tileable = mt.tensor(raw, chunk_size=3)
    table_meta = TensorTableMeta(
        table_name=table_name,
        dtype=str(raw.dtype),
        ndim=2,
        shape=list(raw.shape),
        nsplits=[list(sp) for sp in nsplits],
    )
    result_info = ODPSTableResultInfo(
        ResultType.ODPS_TABLE,
        full_table_name=table_name,
        table_meta=table_meta,
    )

    # Full fetch
    fetched = await fetcher.fetch(tileable, result_info, [None, None])
    np.testing.assert_array_equal(fetched, raw)

    # Slice rows, all cols: [2:5, :]
    fetched = await fetcher.fetch(tileable, result_info, [slice(2, 5), slice(None)])
    np.testing.assert_array_equal(fetched, raw[2:5, :])

    # All rows, integer col: [:, 2] -> 1D
    fetched = await fetcher.fetch(tileable, result_info, [slice(None), 2])
    np.testing.assert_array_equal(fetched, raw[:, 2])

    # Integer row, integer col: [2, 3] -> scalar
    fetched = await fetcher.fetch(tileable, result_info, [2, 3])
    assert fetched == raw[2, 3]

    # Negative slice rows, all cols: [-3:, :]
    fetched = await fetcher.fetch(tileable, result_info, [slice(-3, None), slice(None)])
    np.testing.assert_array_equal(fetched, raw[-3:, :])

    # Step slice: [:10:3, :]
    fetched = await fetcher.fetch(
        tileable, result_info, [slice(None, 10, 3), slice(None)]
    )
    np.testing.assert_array_equal(fetched, raw[:10:3, :])

    # Reverse slice on rows: [::-1, :]
    fetched = await fetcher.fetch(
        tileable, result_info, [slice(None, None, -1), slice(None)]
    )
    np.testing.assert_array_equal(fetched, raw[::-1, :])

    # Reverse slice on cols: [:, ::-1]
    fetched = await fetcher.fetch(
        tileable, result_info, [slice(None), slice(None, None, -1)]
    )
    np.testing.assert_array_equal(fetched, raw[:, ::-1])

    # Reverse slice on both dims: [::-1, ::-1]
    fetched = await fetcher.fetch(
        tileable, result_info, [slice(None, None, -1), slice(None, None, -1)]
    )
    np.testing.assert_array_equal(fetched, raw[::-1, ::-1])

    odps_entry.delete_table(table_name, if_exists=True, wait=False)


@requires_storage_api
async def test_fetch_blob_table_object(odps_account):
    odps_entry = odps_account
    fetcher = ODPSTableFetcher(odps_entry)

    raw = {"key": "value", "number": 42}

    table_name = tn("mf_test_fetch_blob_object_" + str(uuid.uuid4().hex))
    odps_entry.delete_table(table_name, if_exists=True)
    _write_object_blob_table(odps_entry, table_name, raw)

    result_info = ODPSTableResultInfo(
        ResultType.ODPS_TABLE,
        full_table_name=table_name,
    )
    fetched = await fetcher.to_thread(fetcher._fetch_blob_table_object, result_info)
    assert fetched == raw

    odps_entry.delete_table(table_name, if_exists=True, wait=False)


def test_liteframe_physical_columns_maps_sanitized_names():
    """``_liteframe_physical_columns`` maps logical LiteFrame column names to
    their sanitized physical ODPS table names.

    An int-string logical column (``"0"``) is stored as ``_column_0`` at
    table-creation time; the tunnel read must select the physical names and
    the logical names are restored afterwards. Without the mapping, the
    read would select ``"0"`` against a schema that only has ``_column_0``.
    """
    dtypes = pd.Series([np.dtype("int64"), np.dtype("float64")], index=["0", "b"])
    table_meta = LiteFrameTableMeta(
        table_name="t", dtypes=dtypes, table_column_names=["_column_0", "b"]
    )
    assert ODPSTableFetcher._liteframe_physical_columns(table_meta, ["0", "b"]) == [
        "_column_0",
        "b",
    ]
    # No mapping carried -> logical names pass through unchanged.
    assert ODPSTableFetcher._liteframe_physical_columns(
        LiteFrameTableMeta(table_name="t", dtypes=dtypes), ["0"]
    ) == ["0"]
