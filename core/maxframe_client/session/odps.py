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

import abc
import asyncio
import logging
import time
import weakref
from numbers import Integral
from typing import Dict, List, Mapping, Optional, Tuple, Union
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from odps import ODPS

from maxframe.config import options
from maxframe.core import Entity, TileableGraph, enter_mode
from maxframe.dataframe import read_odps_table
from maxframe.dataframe.core import DATAFRAME_TYPE, SERIES_TYPE
from maxframe.dataframe.datasource import PandasDataSourceOperator
from maxframe.dataframe.datasource.read_odps_table import DataFrameReadODPSTable
from maxframe.odpsio import HaloTableIO, pandas_to_arrow, pandas_to_odps_schema
from maxframe.protocol import (
    DagInfo,
    DagStatus,
    ODPSTableResultInfo,
    ResultInfo,
    SessionInfo,
)
from maxframe.session import (
    AbstractSession,
    ExecutionInfo,
    IsolatedAsyncSession,
    Profiling,
    Progress,
)
from maxframe.typing_ import TileableType
from maxframe.utils import ToThreadMixin, build_temp_table_name

from ..clients.framedriver import FrameDriverClient
from ..fetcher import get_fetcher_cls
from .consts import RESTFUL_SESSION_INSECURE_SCHEME, RESTFUL_SESSION_SECURE_SCHEME
from .graph import gen_submit_tileable_graph

logger = logging.getLogger(__name__)


class MaxFrameServiceCaller(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def create_session(self) -> SessionInfo:
        raise NotImplementedError

    @abc.abstractmethod
    def delete_session(self) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def submit_dag(
        self, dag: TileableGraph, managed_input_infos: Dict[str, ResultInfo]
    ) -> DagInfo:
        raise NotImplementedError

    @abc.abstractmethod
    def get_dag_info(self, dag_id: str) -> DagInfo:
        raise NotImplementedError

    @abc.abstractmethod
    def cancel_dag(self, dag_id: str) -> DagInfo:
        raise NotImplementedError

    @abc.abstractmethod
    def decref(self, tileable_keys: List[str]) -> None:
        raise NotImplementedError


class MaxFrameSession(ToThreadMixin, IsolatedAsyncSession):
    _odps_entry: Optional[ODPS]
    _tileable_to_infos: Mapping[TileableType, ResultInfo]

    @classmethod
    async def init(
        cls,
        address: str,
        session_id: Optional[str] = None,
        backend: str = None,
        odps_entry: Optional[ODPS] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> "AbstractSession":
        session_obj = cls(
            address, session_id, odps_entry=odps_entry, timeout=timeout, **kwargs
        )
        await session_obj._init(address)
        return session_obj

    def __init__(
        self,
        address: str,
        session_id: str,
        odps_entry: Optional[ODPS] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(address, session_id)
        self.timeout = timeout
        self._odps_entry = odps_entry or ODPS.from_environments()
        self._tileable_to_infos = weakref.WeakKeyDictionary()

        self._caller = self._create_caller(odps_entry, address, **kwargs)

    @classmethod
    def _create_caller(
        cls, odps_entry: ODPS, address: str, **kwargs
    ) -> MaxFrameServiceCaller:
        raise NotImplementedError

    async def _init(self, _address: str):
        session_info = await self.ensure_async_call(self._caller.create_session)
        self._session_id = session_info.session_id

    def _upload_and_get_read_tileable(self, t: TileableType) -> Optional[TileableType]:
        if (
            not isinstance(t.op, PandasDataSourceOperator)
            or t.op.get_data() is None
            or t.inputs
        ):
            return None

        schema, table_meta = pandas_to_odps_schema(t, unknown_as_string=True)
        if self._odps_entry.exist_table(table_meta.table_name):
            self._odps_entry.delete_table(table_meta.table_name)
        table_name = build_temp_table_name(self.session_id, t.key)
        table_obj = self._odps_entry.create_table(table_name, schema)

        data = t.op.get_data()
        batch_size = options.session.upload_batch_size

        halo_client = HaloTableIO(self._odps_entry)
        with halo_client.open_writer(table_obj.full_table_name) as writer:
            for batch_start in range(0, len(data), batch_size):
                if isinstance(data, pd.Index):
                    batch = data[batch_start : batch_start + batch_size]
                else:
                    batch = data.iloc[batch_start : batch_start + batch_size]
                arrow_batch, _ = pandas_to_arrow(batch)
                writer.write(arrow_batch)

        read_tileable = read_odps_table(
            table_obj.full_table_name,
            columns=table_meta.table_column_names,
            index_col=table_meta.table_index_column_names,
            output_type=table_meta.type,
        )
        if isinstance(read_tileable, DATAFRAME_TYPE):
            if list(read_tileable.dtypes.index) != list(t.dtypes.index):
                read_tileable.columns = list(t.dtypes.index)
        elif isinstance(read_tileable, SERIES_TYPE):
            if read_tileable.name != t.name:
                read_tileable.name = t.name
        else:  # INDEX_TYPE
            if list(read_tileable.names) != list(t.names):
                read_tileable.names = t.names
        read_tileable._key = t.key
        read_tileable.params = t.params
        return read_tileable.data

    @enter_mode(kernel=True, build=True)
    def _scan_and_replace_pandas_sources(
        self, graph: TileableGraph
    ) -> Dict[TileableType, TileableType]:
        """Replaces Pandas data sources with temp table sources in the graph"""
        replacements = dict()
        for t in graph:
            replaced = self._upload_and_get_read_tileable(t)
            if replaced is None:
                continue
            replacements[t] = replaced

        for src, replaced in replacements.items():
            successors = list(graph.successors(src))
            graph.remove_node(src)
            graph.add_node(replaced)
            for pred in replaced.inputs or ():
                graph.add_node(pred)
                graph.add_edge(pred, replaced)

            for succ in successors:
                graph.add_edge(replaced, succ)
                succ.inputs = [replacements.get(t, t) for t in succ.inputs]

        graph.results = [replacements.get(t, t) for t in graph.results]
        return replacements

    @enter_mode(kernel=True, build=True)
    def _get_input_infos(self, tileables: List[TileableType]) -> Dict[str, ResultInfo]:
        """Generate ResultInfo structs from generated temp tables"""
        infos = dict()
        for t in tileables:
            key = t.key
            if not isinstance(t.op, DataFrameReadODPSTable):
                if not isinstance(t.inputs[0].op, DataFrameReadODPSTable):
                    continue
                t = t.inputs[0]
            infos[key] = ODPSTableResultInfo(full_table_name=t.op.table_name)
        return infos

    async def execute(self, *tileables, **kwargs) -> ExecutionInfo:
        tileables = [
            tileable.data if isinstance(tileable, Entity) else tileable
            for tileable in tileables
        ]
        tileable_to_copied = dict()
        tileable_graph, to_execute_tileables = gen_submit_tileable_graph(
            self, tileables, tileable_to_copied
        )
        source_replacements = self._scan_and_replace_pandas_sources(tileable_graph)

        # we need to manage uploaded data sources with refcounting mechanism
        # as nodes in tileable_graph are copied, we need to use original nodes
        copied_to_tileable = {v: k for k, v in tileable_to_copied.items()}
        for replaced_src in source_replacements.keys():
            copied_to_tileable[replaced_src]._attach_session(self)

        replaced_infos = self._get_input_infos(list(source_replacements.values()))
        dag_info = await self.ensure_async_call(
            self._caller.submit_dag, tileable_graph, replaced_infos
        )

        progress = Progress()
        profiling = Profiling()
        aio_task = asyncio.create_task(
            self._run_in_background(dag_info, to_execute_tileables, progress)
        )
        return ExecutionInfo(
            aio_task,
            progress,
            profiling,
            asyncio.get_running_loop(),
            to_execute_tileables,
        )

    async def _run_in_background(
        self, dag_info: DagInfo, tileables: List, progress: Progress
    ):
        start_time = time.time()
        dag_id = dag_info.dag_id
        wait_timeout = 10
        with enter_mode(build=True, kernel=True):
            key_to_tileables = {t.key: t for t in tileables}

            try:
                while True:
                    elapsed_time = time.time() - start_time
                    timeout_val = (
                        min(self.timeout - elapsed_time, wait_timeout)
                        if self.timeout
                        else wait_timeout
                    )
                    if timeout_val <= 0:
                        raise TimeoutError("Running DAG timed out")

                    dag_info: DagInfo = await self.ensure_async_call(
                        self._caller.get_dag_info, dag_id
                    )
                    progress.value = dag_info.progress
                    if dag_info.status != DagStatus.RUNNING:
                        break
                    await asyncio.sleep(timeout_val)
            except asyncio.CancelledError:
                dag_info = await self.ensure_async_call(self._caller.cancel_dag, dag_id)
                if dag_info.status != DagStatus.CANCELLED:  # pragma: no cover
                    raise
            finally:
                if dag_info.status == DagStatus.SUCCEEDED:
                    progress.value = 1.0
                elif dag_info.status == DagStatus.FAILED:
                    dag_info.error_info.reraise()

            if dag_info.status in (DagStatus.RUNNING, DagStatus.CANCELLED):
                return

            for key, result_info in dag_info.tileable_to_result_infos.items():
                t = key_to_tileables[key]
                self._tileable_to_infos[t] = result_info

    def _get_data_tileable_and_indexes(
        self, tileable: TileableType
    ) -> Tuple[TileableType, List[Union[slice, Integral]]]:
        from maxframe.dataframe.indexing.iloc import (
            DataFrameIlocGetItem,
            SeriesIlocGetItem,
        )
        from maxframe.tensor.indexing import TensorIndex

        slice_op_types = TensorIndex, DataFrameIlocGetItem, SeriesIlocGetItem

        if isinstance(tileable, Entity):
            tileable = tileable.data

        indexes = None
        while tileable not in self._tileable_to_infos:
            # if tileable's op is slice, try to check input
            if isinstance(tileable.op, slice_op_types):
                indexes = tileable.op.indexes
                tileable = tileable.inputs[0]
                if not all(isinstance(index, (slice, Integral)) for index in indexes):
                    raise ValueError("Only support fetch data slices")
            else:
                raise ValueError(f"Cannot fetch unexecuted tileable: {tileable!r}")

        return tileable, indexes

    async def fetch(self, *tileables, **kwargs) -> list:
        results = []
        tileables = [
            tileable.data if isinstance(tileable, Entity) else tileable
            for tileable in tileables
        ]
        with enter_mode(build=True):
            for tileable in tileables:
                data_tileable, indexes = self._get_data_tileable_and_indexes(tileable)
                info = self._tileable_to_infos[data_tileable]
                fetcher = get_fetcher_cls(info.result_type)(self._odps_entry)
                results.append(await fetcher.fetch(tileable, info, indexes))
        return results

    async def decref(self, *tileable_keys):
        return await self.ensure_async_call(self._caller.decref, list(tileable_keys))

    async def destroy(self):
        await self.ensure_async_call(self._caller.delete_session)
        await super().destroy()

    async def _get_ref_counts(self) -> Dict[str, int]:
        pass

    async def fetch_tileable_op_logs(
        self,
        tileable_op_key: str,
        offsets: Union[Dict[str, List[int]], str, int],
        sizes: Union[Dict[str, List[int]], str, int],
    ) -> Dict:
        pass

    async def get_total_n_cpu(self):
        pass

    async def get_cluster_versions(self) -> List[str]:
        raise NotImplementedError

    async def get_web_endpoint(self) -> Optional[str]:
        raise NotImplementedError

    async def create_remote_object(
        self, session_id: str, name: str, object_cls, *args, **kwargs
    ):
        raise NotImplementedError

    async def get_remote_object(self, session_id: str, name: str):
        raise NotImplementedError

    async def destroy_remote_object(self, session_id: str, name: str):
        raise NotImplementedError

    async def create_mutable_tensor(
        self,
        shape: tuple,
        dtype: Union[np.dtype, str],
        name: str = None,
        default_value: Union[int, float] = 0,
        chunk_size: Union[int, Tuple] = None,
    ):
        raise NotImplementedError

    async def get_mutable_tensor(self, name: str):
        raise NotImplementedError


class MaxFrameRestCaller(MaxFrameServiceCaller):
    _client: FrameDriverClient
    _session_id: Optional[str]

    def __init__(self, client: FrameDriverClient):
        self._client = client
        self._session_id = None

    async def create_session(self) -> SessionInfo:
        info = await self._client.create_session(options.to_dict(remote_only=True))
        self._session_id = info.session_id
        return info

    async def delete_session(self) -> None:
        await self._client.delete_session(self._session_id)

    async def submit_dag(
        self, dag: TileableGraph, managed_input_infos: Dict[str, ResultInfo]
    ) -> DagInfo:
        return await self._client.submit_dag(self._session_id, dag, managed_input_infos)

    async def get_dag_info(self, dag_id: str) -> DagInfo:
        return await self._client.get_dag_info(self._session_id, dag_id)

    async def cancel_dag(self, dag_id: str) -> DagInfo:
        return await self._client.cancel_dag(self._session_id, dag_id)

    async def decref(self, tileable_keys: List[str]) -> None:
        return await self._client.decref(self._session_id, tileable_keys)


class MaxFrameRestSession(MaxFrameSession):
    schemes = [RESTFUL_SESSION_INSECURE_SCHEME, RESTFUL_SESSION_SECURE_SCHEME]

    def __init__(
        self,
        address: str,
        session_id: str,
        odps_entry: Optional[ODPS] = None,
        timeout: Optional[float] = None,
        new: bool = True,
        **kwargs,
    ):
        parsed_endpoint = urlparse(address)
        scheme = (
            "http"
            if parsed_endpoint.scheme == RESTFUL_SESSION_INSECURE_SCHEME
            else "https"
        )
        real_endpoint = address.replace(f"{parsed_endpoint.scheme}://", f"{scheme}://")

        super().__init__(
            real_endpoint, session_id, odps_entry=odps_entry, timeout=timeout
        )

    @classmethod
    def _create_caller(cls, odps_entry: ODPS, address: str, **kwargs):
        return MaxFrameRestCaller(FrameDriverClient(address))


def register_session_schemes(overwrite: bool = False):
    MaxFrameRestSession.register_schemes(overwrite=overwrite)
