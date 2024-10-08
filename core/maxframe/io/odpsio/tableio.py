# Copyright 1999-2024 Alibaba Group Holding Ltd.
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
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Dict, List, Optional, Union

import numpy as np
import pyarrow as pa
from odps import ODPS
from odps import __version__ as pyodps_version
from odps.apis.storage_api import (
    StorageApiArrowClient,
    TableBatchScanResponse,
    TableBatchWriteResponse,
)
from odps.tunnel import TableTunnel
from odps.types import OdpsSchema, PartitionSpec, timestamp_ntz

try:
    import pyarrow.compute as pac
except ImportError:
    pac = None

from ...config import options
from ...env import ODPS_STORAGE_API_ENDPOINT
from ...lib.version import Version
from ...utils import sync_pyodps_options
from .schema import odps_schema_to_arrow_schema

PartitionsType = Union[List[str], str, None]

_DEFAULT_ROW_BATCH_SIZE = 4096
_need_patch_batch = Version(pyodps_version) < Version("0.12.0")


class ODPSTableIO(ABC):
    def __new__(cls, odps: ODPS):
        if cls is ODPSTableIO:
            if options.use_common_table:
                return HaloTableIO(odps)
            else:
                return TunnelTableIO(odps)
        return super().__new__(cls)

    def __init__(self, odps: ODPS):
        self._odps = odps

    @classmethod
    def _get_reader_schema(
        cls,
        table_schema: OdpsSchema,
        columns: Optional[List[str]] = None,
        partition_columns: Union[None, bool, List[str]] = None,
    ) -> OdpsSchema:
        final_cols = []

        columns = columns or [col.name for col in table_schema.simple_columns]
        if partition_columns is True:
            partition_columns = [c.name for c in table_schema.partitions]
        else:
            partition_columns = partition_columns or []

        for col_name in columns + partition_columns:
            final_cols.append(table_schema[col_name])
        return OdpsSchema(final_cols)

    @abstractmethod
    def open_reader(
        self,
        full_table_name: str,
        partitions: PartitionsType = None,
        columns: Optional[List[str]] = None,
        partition_columns: Union[None, bool, List[str]] = None,
        start: Optional[int] = None,
        stop: Optional[int] = None,
        reverse_range: bool = False,
        row_batch_size: int = _DEFAULT_ROW_BATCH_SIZE,
    ):
        raise NotImplementedError

    @abstractmethod
    def open_writer(
        self,
        full_table_name: str,
        partition: Optional[str] = None,
        overwrite: bool = True,
    ):
        raise NotImplementedError


class TunnelMultiPartitionReader:
    def __init__(
        self,
        odps_entry: ODPS,
        table_name: str,
        partitions: PartitionsType,
        columns: Optional[List[str]] = None,
        partition_columns: Optional[List[str]] = None,
        start: Optional[int] = None,
        count: Optional[int] = None,
        partition_to_download_ids: Dict[str, str] = None,
    ):
        self._odps_entry = odps_entry
        self._table = odps_entry.get_table(table_name)
        self._columns = columns

        odps_schema = ODPSTableIO._get_reader_schema(
            self._table.table_schema, columns, partition_columns
        )
        self._schema = odps_schema_to_arrow_schema(odps_schema)

        self._start = start or 0
        self._count = count
        self._row_left = count

        self._cur_reader = None
        self._reader_iter = None
        self._cur_partition_id = -1
        self._reader_start_pos = 0

        if partitions is None or isinstance(partitions, str):
            self._partitions = [partitions]
        else:
            self._partitions = partitions

        self._partition_cols = partition_columns
        self._partition_to_download_ids = partition_to_download_ids or dict()

    @property
    def count(self) -> Optional[int]:
        if len(self._partitions) > 1:
            return None
        return self._count

    def _open_next_reader(self):
        if self._cur_reader is not None:
            self._reader_start_pos += self._cur_reader.count

        if (
            self._row_left is not None and self._row_left <= 0
        ) or 1 + self._cur_partition_id >= len(self._partitions):
            self._cur_reader = None
            return

        while 1 + self._cur_partition_id < len(self._partitions):
            self._cur_partition_id += 1

            part_str = self._partitions[self._cur_partition_id]

            # todo make this more formal when PyODPS 0.12.0 is released
            req_columns = self._columns
            if not _need_patch_batch:
                req_columns = self._schema.names
            with sync_pyodps_options():
                self._cur_reader = self._table.open_reader(
                    part_str,
                    columns=req_columns,
                    arrow=True,
                    download_id=self._partition_to_download_ids.get(part_str),
                )
            if self._cur_reader.count + self._reader_start_pos > self._start:
                start = self._start - self._reader_start_pos
                if self._count is None:
                    count = None
                else:
                    count = min(self._count, self._cur_reader.count - start)

                with sync_pyodps_options():
                    self._reader_iter = self._cur_reader.read(start, count)
                break
            self._reader_start_pos += self._cur_reader.count
        else:
            self._cur_reader = None

    def _fill_batch_partition(self, batch: pa.RecordBatch) -> pa.RecordBatch:
        pt_spec = PartitionSpec(self._partitions[self._cur_partition_id])

        names = list(batch.schema.names)
        arrays = []
        for idx in range(batch.num_columns):
            col = batch.column(idx)
            if isinstance(col.type, pa.TimestampType):
                if col.type.tz is not None:
                    target_type = pa.timestamp(
                        self._schema.types[idx].unit, col.type.tz
                    )
                    arrays.append(col.cast(target_type))
                else:
                    target_type = pa.timestamp(
                        self._schema.types[idx].unit, options.local_timezone
                    )
                    pd_col = col.to_pandas().dt.tz_localize(options.local_timezone)
                    arrays.append(pa.Array.from_pandas(pd_col).cast(target_type))
            else:
                arrays.append(batch.column(idx))

        for part_col in self._partition_cols or []:
            names.append(part_col)
            col_type = self._schema.field_by_name(part_col).type
            pt_col = np.repeat([pt_spec[part_col]], batch.num_rows)
            arrays.append(pa.array(pt_col).cast(col_type))
        return pa.RecordBatch.from_arrays(arrays, names)

    def read(self):
        with sync_pyodps_options():
            if self._cur_reader is None:
                self._open_next_reader()
                if self._cur_reader is None:
                    return None
            while self._cur_reader is not None:
                try:
                    batch = next(self._reader_iter)
                    if batch is not None:
                        if self._row_left is not None:
                            self._row_left -= batch.num_rows
                        if _need_patch_batch:
                            return self._fill_batch_partition(batch)
                        else:
                            return batch
                except StopIteration:
                    self._open_next_reader()
            return None

    def read_all(self) -> pa.Table:
        batches = []
        while True:
            batch = self.read()
            if batch is None:
                break
            batches.append(batch)
        if not batches:
            return self._schema.empty_table()
        return pa.Table.from_batches(batches)


class TunnelWrappedWriter:
    def __init__(self, nested_writer):
        self._writer = nested_writer

    def write(self, data: Union[pa.RecordBatch, pa.Table]):
        if not any(isinstance(tp, pa.TimestampType) for tp in data.schema.types):
            self._writer.write(data)
            return
        pa_type = type(data)
        arrays = []
        for idx in range(data.num_columns):
            name = data.schema.names[idx]
            col = data.column(idx)
            if not isinstance(col.type, pa.TimestampType):
                arrays.append(col)
                continue
            if self._writer.schema[name].type == timestamp_ntz:
                col = HaloTableArrowWriter._localize_timezone(col, "UTC")
            else:
                col = HaloTableArrowWriter._localize_timezone(col)
            arrays.append(col)
        data = pa_type.from_arrays(arrays, names=data.schema.names)
        self._writer.write(data)

    def __getattr__(self, item):
        return getattr(self._writer, item)


class TunnelTableIO(ODPSTableIO):
    @contextmanager
    def open_reader(
        self,
        full_table_name: str,
        partitions: PartitionsType = None,
        columns: Optional[List[str]] = None,
        partition_columns: Union[None, bool, List[str]] = None,
        start: Optional[int] = None,
        stop: Optional[int] = None,
        reverse_range: bool = False,
        row_batch_size: int = _DEFAULT_ROW_BATCH_SIZE,
    ):
        with sync_pyodps_options():
            table = self._odps.get_table(full_table_name)

        if partition_columns is True:
            partition_columns = [c.name for c in table.table_schema.partitions]

        total_records = None
        part_to_down_id = None
        if (
            (start is not None and start < 0)
            or (stop is not None and stop < 0)
            or (reverse_range and start is None)
        ):
            with sync_pyodps_options():
                table = self._odps.get_table(full_table_name)
                tunnel = TableTunnel(self._odps)
                parts = (
                    [partitions]
                    if partitions is None or isinstance(partitions, str)
                    else partitions
                )
                part_to_down_id = dict()
                total_records = 0
                for part in parts:
                    down_session = tunnel.create_download_session(
                        table, async_mode=True, partition_spec=part
                    )
                    part_to_down_id[part] = down_session.id
                    total_records += down_session.count

        count = None
        if start is not None or stop is not None:
            if reverse_range:
                start = start if start is not None else total_records - 1
                stop = stop if stop is not None else -1
            else:
                start = start if start is not None else 0
                stop = stop if stop is not None else None
            start = start if start >= 0 else total_records + start
            stop = stop if stop is None or stop >= 0 else total_records + stop
            if reverse_range:
                count = start - stop
                start = stop + 1
            else:
                count = stop - start if stop is not None and start is not None else None

        yield TunnelMultiPartitionReader(
            self._odps,
            full_table_name,
            partitions=partitions,
            columns=columns,
            partition_columns=partition_columns,
            start=start,
            count=count,
            partition_to_download_ids=part_to_down_id,
        )

    @contextmanager
    def open_writer(
        self,
        full_table_name: str,
        partition: Optional[str] = None,
        overwrite: bool = True,
    ):
        table = self._odps.get_table(full_table_name)
        with sync_pyodps_options():
            with table.open_writer(
                partition=partition,
                arrow=True,
                create_partition=partition is not None,
                overwrite=overwrite,
            ) as writer:
                # fixme should yield writer directly once pyodps fixes
                #  related arrow timestamp bug when provided schema and
                #  table schema is identical.
                if _need_patch_batch:
                    yield TunnelWrappedWriter(writer)
                else:
                    yield writer


class HaloTableArrowReader:
    def __init__(
        self,
        client: StorageApiArrowClient,
        scan_info: TableBatchScanResponse,
        odps_schema: OdpsSchema,
        start: Optional[int] = None,
        count: Optional[int] = None,
        row_batch_size: Optional[int] = None,
    ):
        self._client = client
        self._scan_info = scan_info

        self._cur_split_id = -1
        self._cur_reader = None

        self._odps_schema = odps_schema
        self._arrow_schema = odps_schema_to_arrow_schema(odps_schema)

        self._start = start
        self._count = count
        self._cursor = 0
        self._row_batch_size = row_batch_size

    @property
    def count(self) -> int:
        return self._count

    def _open_next_reader(self):
        from odps.apis.storage_api import ReadRowsRequest

        if 0 <= self._scan_info.split_count <= self._cur_split_id + 1:
            # scan by split
            self._cur_reader = None
            return
        elif self._count is not None and self._cursor >= self._count:
            # scan by range
            self._cur_reader = None
            return

        read_rows_kw = {}
        if self._start is not None:
            read_rows_kw["row_index"] = self._start + self._cursor
            read_rows_kw["row_count"] = min(
                self._row_batch_size, self._count - self._cursor
            )
            self._cursor = min(self._count, self._cursor + self._row_batch_size)

        req = ReadRowsRequest(
            session_id=self._scan_info.session_id,
            split_index=self._cur_split_id + 1,
            **read_rows_kw,
        )
        self._cur_reader = self._client.read_rows_arrow(req)
        self._cur_split_id += 1

    def _convert_timezone(self, batch: pa.RecordBatch) -> pa.RecordBatch:
        timezone = options.local_timezone
        if not any(isinstance(tp, pa.TimestampType) for tp in batch.schema.types):
            return batch

        cols = []
        for idx in range(batch.num_columns):
            col = batch.column(idx)
            name = batch.schema.names[idx]
            if not isinstance(col.type, pa.TimestampType):
                cols.append(col)
                continue
            if self._odps_schema[name].type == timestamp_ntz:
                col = col.cast(pa.timestamp(col.type.unit))
                cols.append(col)
                continue

            if hasattr(pac, "local_timestamp"):
                col = col.cast(pa.timestamp(col.type.unit, timezone))
            else:
                pd_col = col.to_pandas().dt.tz_convert(timezone)
                col = pa.Array.from_pandas(pd_col).cast(
                    pa.timestamp(col.type.unit, timezone)
                )
            cols.append(col)

        return pa.RecordBatch.from_arrays(cols, names=batch.schema.names)

    def read(self):
        if self._cur_reader is None:
            self._open_next_reader()
            if self._cur_reader is None:
                return None
        while self._cur_reader is not None:
            batch = self._cur_reader.read()
            if batch is not None:
                return self._convert_timezone(batch)
            self._open_next_reader()
        return None

    def read_all(self) -> pa.Table:
        batches = []
        while True:
            batch = self.read()
            if batch is None:
                break
            batches.append(batch)
        if not batches:
            return self._arrow_schema.empty_table()
        return pa.Table.from_batches(batches)


class HaloTableArrowWriter:
    def __init__(
        self,
        client: StorageApiArrowClient,
        write_info: TableBatchWriteResponse,
        odps_schema: OdpsSchema,
    ):
        self._client = client
        self._write_info = write_info
        self._odps_schema = odps_schema
        self._arrow_schema = odps_schema_to_arrow_schema(odps_schema)

        self._writer = None

    def open(self):
        from odps.apis.storage_api import WriteRowsRequest

        self._writer = self._client.write_rows_arrow(
            WriteRowsRequest(self._write_info.session_id)
        )

    @classmethod
    def _localize_timezone(cls, col, tz=None):
        from odps.lib import tzlocal

        if tz is None:
            if options.local_timezone is None:
                tz = str(tzlocal.get_localzone())
            else:
                tz = str(options.local_timezone)

        if col.type.tz is not None:
            return col
        if hasattr(pac, "assume_timezone"):
            col = pac.assume_timezone(col, tz)
            return col
        else:
            col = col.to_pandas()
            return pa.Array.from_pandas(col.dt.tz_localize(tz))

    def _convert_schema(self, batch: pa.RecordBatch):
        if batch.schema == self._arrow_schema and not any(
            isinstance(tp, pa.TimestampType) for tp in self._arrow_schema.types
        ):
            return batch
        cols = []
        for idx in range(batch.num_columns):
            col = batch.column(idx)
            name = batch.schema.names[idx]

            if isinstance(col.type, pa.TimestampType):
                if self._odps_schema[name].type == timestamp_ntz:
                    col = self._localize_timezone(col, "UTC")
                else:
                    col = self._localize_timezone(col)

            if col.type != self._arrow_schema.types[idx]:
                col = col.cast(self._arrow_schema.types[idx])
            cols.append(col)
        return pa.RecordBatch.from_arrays(cols, names=batch.schema.names)

    def write(self, batch):
        if isinstance(batch, pa.Table):
            for b in batch.to_batches():
                self._writer.write(self._convert_schema(b))
        else:
            self._writer.write(self._convert_schema(batch))

    def close(self):
        commit_msg, is_success = self._writer.finish()
        if not is_success:
            raise IOError(commit_msg)
        return commit_msg


class HaloTableIO(ODPSTableIO):
    _storage_api_endpoint = os.getenv(ODPS_STORAGE_API_ENDPOINT)

    @staticmethod
    def _convert_partitions(partitions: PartitionsType) -> Optional[List[str]]:
        if partitions is None:
            return []
        elif isinstance(partitions, (str, PartitionSpec)):
            partitions = [partitions]
        return [
            "/".join(f"{k}={v}" for k, v in PartitionSpec(pt).items())
            for pt in partitions
        ]

    def get_table_record_count(
        self, full_table_name: str, partitions: PartitionsType = None
    ):
        from odps.apis.storage_api import SplitOptions, TableBatchScanRequest

        table = self._odps.get_table(full_table_name)
        client = StorageApiArrowClient(
            self._odps, table, rest_endpoint=self._storage_api_endpoint
        )

        split_option = SplitOptions.SplitMode.SIZE

        scan_kw = {
            "required_partitions": self._convert_partitions(partitions),
            "split_options": SplitOptions.get_default_options(split_option),
        }

        # todo add more options for partition column handling
        req = TableBatchScanRequest(**scan_kw)
        resp = client.create_read_session(req)
        return resp.record_count

    @contextmanager
    def open_reader(
        self,
        full_table_name: str,
        partitions: PartitionsType = None,
        columns: Optional[List[str]] = None,
        partition_columns: Union[None, bool, List[str]] = None,
        start: Optional[int] = None,
        stop: Optional[int] = None,
        reverse_range: bool = False,
        row_batch_size: int = _DEFAULT_ROW_BATCH_SIZE,
    ):
        from odps.apis.storage_api import (
            SessionRequest,
            SessionStatus,
            SplitOptions,
            TableBatchScanRequest,
        )

        table = self._odps.get_table(full_table_name)
        client = StorageApiArrowClient(
            self._odps, table, rest_endpoint=self._storage_api_endpoint
        )

        split_option = SplitOptions.SplitMode.SIZE
        if start is not None or stop is not None:
            split_option = SplitOptions.SplitMode.ROW_OFFSET

        scan_kw = {
            "required_partitions": self._convert_partitions(partitions),
            "split_options": SplitOptions.get_default_options(split_option),
        }
        columns = columns or [c.name for c in table.table_schema.simple_columns]
        scan_kw["required_data_columns"] = columns
        if partition_columns is True:
            scan_kw["required_partition_columns"] = [
                c.name for c in table.table_schema.partitions
            ]
        else:
            scan_kw["required_partition_columns"] = partition_columns

        # todo add more options for partition column handling
        req = TableBatchScanRequest(**scan_kw)
        resp = client.create_read_session(req)

        session_id = resp.session_id
        status = resp.session_status
        while status == SessionStatus.INIT:
            resp = client.get_read_session(SessionRequest(session_id))
            status = resp.session_status
            time.sleep(1.0)

        assert status == SessionStatus.NORMAL

        count = None
        if start is not None or stop is not None:
            if reverse_range:
                start = start if start is not None else resp.record_count - 1
                stop = stop if stop is not None else -1
            else:
                start = start if start is not None else 0
                stop = stop if stop is not None else resp.record_count
            start = start if start >= 0 else resp.record_count + start
            stop = stop if stop >= 0 else resp.record_count + stop
            if reverse_range:
                count = start - stop
                start = stop + 1
            else:
                count = stop - start

        reader_schema = self._get_reader_schema(
            table.table_schema, columns, partition_columns
        )
        yield HaloTableArrowReader(
            client,
            resp,
            odps_schema=reader_schema,
            start=start,
            count=count,
            row_batch_size=row_batch_size,
        )

    @contextmanager
    def open_writer(
        self,
        full_table_name: str,
        partition: Optional[str] = None,
        overwrite: bool = True,
    ):
        from odps.apis.storage_api import (
            SessionRequest,
            SessionStatus,
            TableBatchWriteRequest,
        )

        table = self._odps.get_table(full_table_name)
        client = StorageApiArrowClient(
            self._odps, table, rest_endpoint=self._storage_api_endpoint
        )

        part_strs = self._convert_partitions(partition)
        part_str = part_strs[0] if part_strs else None
        req = TableBatchWriteRequest(partition_spec=part_str, overwrite=overwrite)
        resp = client.create_write_session(req)

        session_id = resp.session_id
        writer = HaloTableArrowWriter(client, resp, table.table_schema)
        writer.open()

        yield writer

        commit_msg = writer.close()
        resp = client.commit_write_session(
            SessionRequest(session_id=session_id), [commit_msg]
        )
        while resp.session_status == SessionStatus.COMMITTING:
            resp = client.get_write_session(SessionRequest(session_id=session_id))
        assert resp.session_status == SessionStatus.COMMITTED
