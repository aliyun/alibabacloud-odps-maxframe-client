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
from typing import List, Optional, Union

import pyarrow as pa
from odps import ODPS
from odps.apis.storage_api import (
    StorageApiArrowClient,
    TableBatchScanResponse,
    TableBatchWriteResponse,
)
from odps.types import PartitionSpec

from ..env import ODPS_STORAGE_API_ENDPOINT
from .schema import odps_schema_to_arrow_schema

PartitionsType = Union[List[str], str, None]

_DEFAULT_ROW_BATCH_SIZE = 4096


class MCTableIO(ABC):
    def __init__(self, odps: ODPS):
        self._odps = odps

    @abstractmethod
    def open_reader(
        self,
        full_table_name: str,
        partitions: PartitionsType = None,
        columns: Optional[List[str]] = None,
        start: Optional[int] = None,
        count: Optional[int] = None,
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


class HaloTableArrowReader:
    def __init__(
        self,
        client: StorageApiArrowClient,
        scan_info: TableBatchScanResponse,
        schema: pa.Schema,
        start: Optional[int] = None,
        count: Optional[int] = None,
        row_batch_size: Optional[int] = None,
    ):
        self._client = client
        self._scan_info = scan_info

        self._cur_split_id = -1
        self._cur_reader = None

        self._schema = schema

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

    def read(self):
        if self._cur_reader is None:
            self._open_next_reader()
            if self._cur_reader is None:
                return None
        while self._cur_reader is not None:
            batch = self._cur_reader.read()
            if batch is not None:
                return batch
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


class HaloTableArrowWriter:
    def __init__(
        self, client: StorageApiArrowClient, write_info: TableBatchWriteResponse
    ):
        self._client = client
        self._write_info = write_info

        self._writer = None

    def open(self):
        from odps.apis.storage_api import WriteRowsRequest

        self._writer = self._client.write_rows_arrow(
            WriteRowsRequest(self._write_info.session_id)
        )

    def write(self, batch):
        if isinstance(batch, pa.Table):
            for b in batch.to_batches():
                self._writer.write(b)
        else:
            self._writer.write(batch)

    def close(self):
        commit_msg, is_success = self._writer.finish()
        if not is_success:
            raise IOError(commit_msg)
        return commit_msg


class HaloTableIO(MCTableIO):
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
            SplitOptions,
            Status,
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
        status = resp.status
        while status == Status.WAIT:
            resp = client.get_read_session(SessionRequest(session_id))
            status = resp.status
            time.sleep(1.0)

        assert status == Status.OK

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

        yield HaloTableArrowReader(
            client,
            resp,
            schema=odps_schema_to_arrow_schema(table.table_schema),
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
        writer = HaloTableArrowWriter(client, resp)
        writer.open()

        yield writer

        commit_msg = writer.close()
        resp = client.commit_write_session(
            SessionRequest(session_id=session_id), [commit_msg]
        )
        while resp.session_status == SessionStatus.COMMITTING:
            resp = client.get_write_session(SessionRequest(session_id=session_id))
        assert resp.session_status == SessionStatus.COMMITTED
