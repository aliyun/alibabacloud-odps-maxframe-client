# Copyright 1999-2025 Alibaba Group Holding Ltd.
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
from collections import OrderedDict
from contextlib import contextmanager
from typing import Dict, List, Optional, Union

import pyarrow as pa
from odps import ODPS
from odps.apis.storage_api import (
    StorageApiArrowClient,
    TableBatchScanResponse,
    TableBatchWriteResponse,
)
from odps.errors import TableModified
from odps.tunnel import TableDownloadSession, TableDownloadStatus, TableTunnel
from odps.types import OdpsSchema, PartitionSpec, timestamp_ntz

try:
    import pyarrow.compute as pac
except ImportError:
    pac = None

from ...config import options
from ...env import ODPS_STORAGE_API_ENDPOINT
from ...utils import call_with_retry, is_empty, sync_pyodps_options
from .schema import odps_schema_to_arrow_schema

PartitionsType = Union[List[str], str, None]

_DEFAULT_ROW_BATCH_SIZE = 4096
_DOWNLOAD_ID_CACHE_SIZE = 100


class ODPSTableIO(ABC):
    def __new__(cls, odps: ODPS):
        if cls is ODPSTableIO:
            if options.use_common_table or ODPS_STORAGE_API_ENDPOINT in os.environ:
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

        columns = (
            columns
            if not is_empty(columns)
            else [col.name for col in table_schema.simple_columns]
        )
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

        if partitions is None:
            if not self._table.table_schema.partitions:
                self._partitions = [None]
            else:
                self._partitions = [str(pt) for pt in self._table.partitions]
        elif isinstance(partitions, str):
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

    def _open_table_reader(self, partition: Optional[str], columns: List[str]):
        attempts = 2
        for trial in range(attempts):
            try:
                return self._table.open_reader(
                    partition,
                    columns=columns,
                    arrow=True,
                    download_id=self._partition_to_download_ids.get(partition),
                    append_partitions=True,
                )
            except TableModified:
                if trial == attempts - 1:
                    raise
                pt_to_session = TunnelTableIO.create_download_sessions(
                    self._odps_entry,
                    self._table.full_table_name,
                    partition,
                    reopen=True,
                )
                assert partition in pt_to_session
                self._partition_to_download_ids[partition] = pt_to_session[partition].id
        raise RuntimeError(
            "Unexpected condition: all trial of open reader done and not raised"
        )

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
            req_columns = self._schema.names
            with sync_pyodps_options():
                self._cur_reader = self._open_table_reader(
                    part_str, columns=req_columns
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

    def read(self):
        with sync_pyodps_options():
            is_first_batch = False
            if self._cur_reader is None:
                is_first_batch = True
                self._open_next_reader()
                if self._cur_reader is None:
                    return None
            while self._cur_reader is not None:
                try:
                    try:
                        batch = next(self._reader_iter)
                    except TableModified:
                        if not is_first_batch:
                            raise
                        # clear download id cache to create new sessions
                        self._partition_to_download_ids = dict()
                        self._cur_reader = None
                        self._open_next_reader()
                        if self._cur_reader is None:
                            return None
                        batch = next(self._reader_iter)

                    if batch is not None:
                        if self._row_left is not None:
                            self._row_left -= batch.num_rows
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


class TunnelTableIO(ODPSTableIO):
    _down_session_ids = OrderedDict()
    _down_modified_time = dict()

    @classmethod
    def _get_modified_time(cls, odps_entry: ODPS, full_table_name, partition):
        data_src = odps_entry.get_table(full_table_name)
        if partition is not None:
            data_src = data_src.partitions[partition]
        return data_src.last_data_modified_time

    @classmethod
    def create_download_sessions(
        cls,
        odps_entry: ODPS,
        full_table_name: str,
        partitions: List[Optional[str]] = None,
        reopen: bool = False,
        timeout: Optional[float] = None,
    ) -> Dict[Optional[str], TableDownloadSession]:
        table = odps_entry.get_table(full_table_name)
        tunnel = TableTunnel(odps_entry, quota_name=options.tunnel_quota_name)
        parts = (
            [partitions]
            if partitions is None or isinstance(partitions, str)
            else partitions
        )
        part_to_session = dict()
        for part in parts:
            part_key = (full_table_name, part)
            modified_time = cls._get_modified_time(odps_entry, full_table_name, part)
            down_session = None

            if (
                not reopen
                and part_key in cls._down_session_ids
                and cls._down_modified_time.get(part_key) == modified_time
            ):
                down_id = cls._down_session_ids[part_key]
                down_session = tunnel.create_download_session(
                    table,
                    async_mode=True,
                    partition_spec=part,
                    download_id=down_id,
                    timeout=timeout,
                )
                if down_session.status != TableDownloadStatus.Normal:
                    down_session = None

            if down_session is None:
                down_session = tunnel.create_download_session(
                    table, async_mode=True, partition_spec=part, timeout=timeout
                )

            while len(cls._down_session_ids) >= _DOWNLOAD_ID_CACHE_SIZE:
                k, _ = cls._down_session_ids.popitem(False)
                cls._down_modified_time.pop(k)
            cls._down_session_ids[part_key] = down_session.id
            cls._down_modified_time[part_key] = modified_time
            part_to_session[part] = down_session
        return part_to_session

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
                tunnel_sessions = self.create_download_sessions(
                    self._odps, full_table_name, partitions
                )
                part_to_down_id = {
                    pt: session.id for (pt, session) in tunnel_sessions.items()
                }
                total_records = sum(
                    session.count for session in tunnel_sessions.values()
                )

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
        self._cur_reader = call_with_retry(self._client.read_rows_arrow, req)
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

        self._writer = call_with_retry(
            self._client.write_rows_arrow,
            WriteRowsRequest(self._write_info.session_id),
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
            self._odps,
            table,
            rest_endpoint=self._storage_api_endpoint,
            quota_name=options.tunnel_quota_name,
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
        resp = call_with_retry(client.create_read_session, req)

        session_id = resp.session_id
        status = resp.session_status
        while status == SessionStatus.INIT:
            resp = call_with_retry(client.get_read_session, SessionRequest(session_id))
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
            self._odps,
            table,
            rest_endpoint=self._storage_api_endpoint,
            quota_name=options.tunnel_quota_name,
        )

        part_strs = self._convert_partitions(partition)
        part_str = part_strs[0] if part_strs else None
        req = TableBatchWriteRequest(partition_spec=part_str, overwrite=overwrite)
        resp = call_with_retry(client.create_write_session, req)

        session_id = resp.session_id
        writer = HaloTableArrowWriter(client, resp, table.table_schema)
        writer.open()

        yield writer

        commit_msg = writer.close()
        resp = call_with_retry(
            client.commit_write_session,
            SessionRequest(session_id=session_id),
            [commit_msg],
        )
        while resp.session_status == SessionStatus.COMMITTING:
            resp = call_with_retry(
                client.get_write_session, SessionRequest(session_id=session_id)
            )
        assert resp.session_status == SessionStatus.COMMITTED
