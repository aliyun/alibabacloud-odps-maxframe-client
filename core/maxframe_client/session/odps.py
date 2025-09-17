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

import abc
import asyncio
import copy
import logging
import time
import weakref
from numbers import Integral
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from odps import ODPS
from odps import options as odps_options
from odps.config import option_context as odps_option_context
from odps.console import in_ipython_frontend

from maxframe.codegen import CodeGenResult
from maxframe.codegen.spe import SPECodeGenerator
from maxframe.config import options
from maxframe.core import Entity, TileableGraph, build_fetch, enter_mode
from maxframe.core.operator import Fetch, estimate_tileable_execution_size
from maxframe.dataframe import read_odps_table
from maxframe.dataframe.core import DATAFRAME_TYPE, INDEX_TYPE, SERIES_TYPE
from maxframe.dataframe.datasource import PandasDataSourceOperator
from maxframe.dataframe.datasource.read_odps_table import DataFrameReadODPSTable
from maxframe.errors import (
    MaxFrameError,
    NoTaskServerResponseError,
    SessionAlreadyClosedError,
    TileableNotExecutedError,
)
from maxframe.io.objects import get_object_io_handler
from maxframe.io.odpsio import (
    ODPSTableIO,
    ODPSVolumeWriter,
    pandas_to_arrow,
    pandas_to_odps_schema,
)
from maxframe.protocol import (
    ConstantResultInfo,
    DagInfo,
    ExecutionStatus,
    ODPSTableResultInfo,
    ODPSVolumeResultInfo,
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
from maxframe.sperunner import SPEDagRunner
from maxframe.tensor.datasource import ArrayDataSource
from maxframe.typing_ import TileableType
from maxframe.utils import (
    ToThreadMixin,
    build_session_volume_name,
    build_temp_table_name,
    estimate_pandas_size,
    get_default_table_properties,
    no_default,
    str_to_bool,
    sync_pyodps_options,
)

from ..clients.framedriver import FrameDriverClient
from ..fetcher import get_fetcher_cls
from .consts import RESTFUL_SESSION_INSECURE_SCHEME, RESTFUL_SESSION_SECURE_SCHEME
from .graph import gen_submit_tileable_graph

logger = logging.getLogger(__name__)


class MaxFrameServiceCaller(metaclass=abc.ABCMeta):
    def get_settings_to_upload(self) -> Dict[str, Any]:
        odps_entry = getattr(self, "_odps_entry", None)
        entry_quota_name = getattr(odps_entry, "quota_name", None)

        sql_settings = (odps_options.sql.settings or {}).copy()
        sql_settings.update(options.sql.settings or {})
        quota_name = (
            options.session.quota_name
            or entry_quota_name
            or getattr(odps_options, "quota_name", None)
        )
        quota_settings = {
            sql_settings.get("odps.task.wlm.quota", None),
            options.spe.task.settings.get("odps.task.wlm.quota", None),
            options.pythonpack.task.settings.get("odps.task.wlm.quota", None),
            options.dpe.task.settings.get("odps.task.wlm.quota", None),
            quota_name,
        }.difference([None])
        if len(quota_settings) >= 2:
            raise ValueError(
                "Quota settings are conflicting: %s" % ", ".join(sorted(quota_settings))
            )
        elif len(quota_settings) == 1:
            quota_name = quota_settings.pop()
        lifecycle = options.session.table_lifecycle or odps_options.lifecycle
        temp_lifecycle = (
            options.session.temp_table_lifecycle or odps_options.temp_lifecycle
        )

        enable_schema = options.session.enable_schema
        default_schema = options.session.default_schema
        if hasattr(self, "_odps_entry"):
            default_schema = default_schema or self._odps_entry.schema

        # use flags in sql settings
        if sql_settings.get("odps.default.schema"):
            default_schema = sql_settings["odps.default.schema"]
        if str_to_bool(
            sql_settings.get("odps.namespace.schema") or "false"
        ) or str_to_bool(
            sql_settings.get("odps.sql.allow.namespace.schema") or "false"
        ):
            enable_schema = True

        mf_settings = dict(options.to_dict(remote_only=True).items())
        mf_settings["sql.settings"] = sql_settings
        mf_settings["session.table_lifecycle"] = lifecycle
        mf_settings["session.temp_table_lifecycle"] = temp_lifecycle
        mf_settings["session.quota_name"] = quota_name
        if enable_schema is not None:
            mf_settings["session.enable_schema"] = enable_schema
        if options.session.enable_high_availability is None:
            mf_settings["session.enable_high_availability"] = not in_ipython_frontend()
        mf_settings["session.default_schema"] = default_schema or "default"
        return mf_settings

    @abc.abstractmethod
    def create_session(self) -> SessionInfo:
        raise NotImplementedError

    @abc.abstractmethod
    def delete_session(self) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def submit_dag(
        self,
        dag: TileableGraph,
        managed_input_infos: Dict[str, ResultInfo],
        new_settings: Dict[str, Any] = None,
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

    def get_logview_address(self, dag_id=None, hours=None) -> Optional[str]:
        return None


class LocalSPEDagRunner(SPEDagRunner):
    def __init__(
        self,
        session_id: str,
        subdag_id: str,
        subdag: TileableGraph,
        generated: CodeGenResult,
        settings: Dict[str, Any],
        odps_entry: Optional[ODPS] = None,
        tileable_to_info: Mapping[TileableType, ResultInfo] = None,
        data_tileable_getter: Optional[
            Callable[[TileableType], Tuple[TileableType, Any]]
        ] = None,
        loop: asyncio.AbstractEventLoop = None,
    ):
        super().__init__(session_id, subdag_id, subdag, generated, settings)
        self._odps = self._odps or odps_entry
        self._tileable_key_to_info = {
            t.key: v for t, v in ({} or tileable_to_info).items()
        }
        self._data_tileable_getter = data_tileable_getter or (lambda x: (x, None))
        self._loop = loop

    def fetch_data_by_tileable(self, t: TileableType) -> Any:
        tileable, index = self._data_tileable_getter(t)
        result_info = self._tileable_key_to_info[tileable.key]
        fetcher = get_fetcher_cls(result_info.result_type)(self._odps)
        return asyncio.run_coroutine_threadsafe(
            fetcher.fetch(t, result_info, index), self._loop
        ).result()

    def store_data(self, key: str, value: Any) -> ResultInfo:
        return ConstantResultInfo(data=value)


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
        self._odps_entry = odps_entry or ODPS.from_global() or ODPS.from_environments()
        self._tileable_to_infos = weakref.WeakKeyDictionary()

        self._caller = self._create_caller(odps_entry, address, **kwargs)
        self._last_settings = None
        self._pull_interval = 1 if in_ipython_frontend() else 3
        self._replace_internal_host = kwargs.get("replace_internal_host", True)

    @classmethod
    def _create_caller(
        cls, odps_entry: ODPS, address: str, **kwargs
    ) -> MaxFrameServiceCaller:
        raise NotImplementedError

    async def _init(self, _address: str):
        session_info = await self.ensure_async_call(self._caller.create_session)
        self._last_settings = copy.deepcopy(self._caller.get_settings_to_upload())
        self._session_id = session_info.session_id
        await self._show_logview_address()

    def _upload_and_get_table_read_tileable(
        self, t: TileableType, data: Any
    ) -> Optional[TileableType]:
        table_schema, table_meta = pandas_to_odps_schema(t, unknown_as_string=True)
        if self._odps_entry.exist_table(table_meta.table_name):
            self._odps_entry.delete_table(
                table_meta.table_name, hints=options.sql.settings
            )
        table_name = build_temp_table_name(self.session_id, t.key)
        schema = (
            self._last_settings.get("session.default_schema", None)
            if self._last_settings.get("session.enable_schema", False)
            else None
        )
        table_obj = self._odps_entry.create_table(
            table_name,
            table_schema,
            schema=schema,
            lifecycle=options.session.temp_table_lifecycle,
            hints=options.sql.settings,
            if_not_exists=True,
            table_properties=options.session.temp_table_properties
            or get_default_table_properties(),
        )

        batch_size = options.session.upload_batch_size

        if len(data):
            table_client = ODPSTableIO(self._odps_entry)
            with table_client.open_writer(table_obj.full_table_name) as writer:
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
                read_tileable.rename(t.names, inplace=True)
        read_tileable._key = t.key
        read_tileable.params = t.params
        return read_tileable.data

    def _upload_and_get_vol_read_tileable(
        self, t: TileableType, data: Any
    ) -> Optional[TileableType]:
        vol_name = build_session_volume_name(self.session_id)
        writer = ODPSVolumeWriter(
            self._odps_entry,
            vol_name,
            t.key,
            replace_internal_host=self._replace_internal_host,
        )
        io_handler = get_object_io_handler(t)
        io_handler().write_object(writer, t, data)
        return build_fetch(t).data

    def _get_local_data(self, t: TileableType) -> Any:
        if isinstance(t.op, (ArrayDataSource, PandasDataSourceOperator)):
            # scenario 1: tensor or DataFrame input
            if t.inputs:
                return no_default
            return t.op.get_data()
        if isinstance(t.op, Fetch):
            # scenario 2: local data
            key_to_tileables = {t.key: t for t in self._tileable_to_infos.keys()}
            if t.key not in key_to_tileables:
                return no_default
            src_info = self._tileable_to_infos[key_to_tileables[t.key]]
            if not isinstance(src_info, ConstantResultInfo):
                return no_default
            return src_info.data
        return no_default

    def _upload_and_get_read_tileable(self, t: TileableType) -> Optional[TileableType]:
        local_data = self._get_local_data(t)
        if local_data is no_default:
            return
        with sync_pyodps_options():
            if isinstance(t, DATAFRAME_TYPE + SERIES_TYPE + INDEX_TYPE):
                return self._upload_and_get_table_read_tileable(t, local_data)
            else:
                return self._upload_and_get_vol_read_tileable(t, local_data)

    @enter_mode(kernel=True, build=True)
    def _scan_and_replace_local_sources(
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
                succ.op.inputs = [replacements.get(t, t) for t in succ.inputs]

        graph.results = [replacements.get(t, t) for t in graph.results]
        return replacements

    @enter_mode(kernel=True, build=True)
    def _get_input_infos(self, tileables: List[TileableType]) -> Dict[str, ResultInfo]:
        """Generate ResultInfo structs from generated temp tables"""
        vol_name = build_session_volume_name(self.session_id)

        infos = dict()
        for t in tileables:
            key = t.key
            if isinstance(t.op, DataFrameReadODPSTable):
                infos[key] = ODPSTableResultInfo(full_table_name=t.op.table_name)
            else:
                if isinstance(t.op, Fetch):
                    infos[key] = ODPSVolumeResultInfo(
                        volume_name=vol_name, volume_path=t.key
                    )
                elif t.inputs and isinstance(t.inputs[0].op, DataFrameReadODPSTable):
                    t = t.inputs[0]
                    infos[key] = ODPSTableResultInfo(full_table_name=t.op.table_name)
        return infos

    def _get_diff_settings(self) -> Dict[str, Any]:
        new_settings = self._caller.get_settings_to_upload()
        if not self._last_settings:  # pragma: no cover
            self._last_settings = copy.deepcopy(new_settings)
            return new_settings

        if self._last_settings.get("session.quota_name", None) != new_settings.get(
            "session.quota_name", None
        ):
            raise ValueError("Quota name cannot be changed after sessions are created")

        update = dict()
        for k in new_settings.keys():
            old_item = self._last_settings.get(k)
            new_item = new_settings.get(k)
            try:
                if old_item != new_item:
                    update[k] = new_item
            except:  # noqa: E722  # nosec  # pylint: disable=bare-except
                update[k] = new_item
        self._last_settings = copy.deepcopy(new_settings)
        return update

    def _is_local_executable(self, tileable_graph: TileableGraph) -> bool:
        from maxframe.codegen.spe.core import get_op_adapter

        local_exec_size_limit = options.local_execution.size_limit
        if not options.local_execution.enabled or not local_exec_size_limit:
            return False

        # make sure all ops registered in SPE
        try:
            for tileable in tileable_graph:
                get_op_adapter(type(tileable.op))
        except KeyError:
            return False

        fetch_sizes = dict()
        for inp in tileable_graph.iter_indep():
            if not isinstance(inp.op, Fetch):
                continue
            local_data = self._get_local_data(inp)
            # todo add resolution of sizes of tensor data type here
            if isinstance(local_data, (pd.DataFrame, pd.Series, pd.Index)):
                fetch_sizes[inp.key] = estimate_pandas_size(local_data)

        est_exec_size = estimate_tileable_execution_size(
            tileable_graph, fetch_sizes=fetch_sizes
        )
        return local_exec_size_limit and est_exec_size < local_exec_size_limit

    async def execute(self, *tileables, **kwargs) -> ExecutionInfo:
        tileables = [
            tileable.data if isinstance(tileable, Entity) else tileable
            for tileable in tileables
        ]
        tileable_to_copied = dict()
        tileable_graph, to_execute_tileables = gen_submit_tileable_graph(
            self, tileables, tileable_to_copied
        )

        if self._is_local_executable(tileable_graph):
            return await self._execute_locally(tileable_graph, to_execute_tileables)
        else:
            return await self._execute_in_service(
                tileable_graph, to_execute_tileables, tileable_to_copied
            )

    async def _execute_in_service(
        self,
        tileable_graph: TileableGraph,
        to_execute_tileables: List[TileableType],
        tileable_to_copied: Dict[TileableType, TileableType],
    ) -> ExecutionInfo:
        source_replacements = self._scan_and_replace_local_sources(tileable_graph)

        # we need to manage uploaded data sources with refcounting mechanism
        # as nodes in tileable_graph are copied, we need to use original nodes
        copied_to_tileable = {v: k for k, v in tileable_to_copied.items()}
        for replaced_src in source_replacements.keys():
            copied_to_tileable[replaced_src]._attach_session(self)

        replaced_infos = self._get_input_infos(list(source_replacements.values()))
        dag_info = await self.ensure_async_call(
            self._caller.submit_dag,
            tileable_graph,
            replaced_infos,
            self._get_diff_settings(),
        )

        await self._show_logview_address(dag_info.dag_id)

        progress = Progress()
        profiling = Profiling()
        aio_task = asyncio.create_task(
            self._run_remotely_in_background(dag_info, to_execute_tileables, progress)
        )
        return ExecutionInfo(
            aio_task,
            progress,
            profiling,
            asyncio.get_running_loop(),
            to_execute_tileables,
        )

    async def _run_remotely_in_background(
        self, dag_info: DagInfo, tileables: List, progress: Progress
    ):
        start_time = time.time()
        session_id = dag_info.session_id
        dag_id = dag_info.dag_id
        server_no_response_time = None
        with enter_mode(build=True, kernel=True):
            key_to_tileables = {t.key: t for t in tileables}
            timeout_val = 0.1
            try:
                while True:
                    elapsed_time = time.time() - start_time
                    next_timeout_val = min(timeout_val * 2, self._pull_interval)
                    timeout_val = (
                        min(self.timeout - elapsed_time, next_timeout_val)
                        if self.timeout
                        else next_timeout_val
                    )
                    if timeout_val <= 0:
                        raise TimeoutError("Running DAG timed out")

                    try:
                        dag_info: DagInfo = await self.ensure_async_call(
                            self._caller.get_dag_info, dag_id
                        )
                        server_no_response_time = None
                    except (NoTaskServerResponseError, SessionAlreadyClosedError) as ex:
                        # when we receive SessionAlreadyClosedError after NoTaskServerResponseError
                        #  is received, it is possible that task server is restarted and
                        #  SessionAlreadyClosedError might be flaky. Otherwise, the error
                        #  should be raised.
                        if (
                            isinstance(ex, SessionAlreadyClosedError)
                            and not server_no_response_time
                        ):
                            raise
                        server_no_response_time = server_no_response_time or time.time()
                        if (
                            time.time() - server_no_response_time
                            > options.client.task_restart_timeout
                        ):
                            raise MaxFrameError(
                                "Failed to get valid response from service. "
                                f"Session {self._session_id}."
                            ) from None
                        await asyncio.sleep(timeout_val)
                        continue

                    if dag_info is None:
                        raise SystemError(
                            f"Cannot find DAG with ID {dag_id} in session {session_id}"
                        )
                    progress.value = dag_info.progress
                    if dag_info.status != ExecutionStatus.RUNNING:
                        break
                    await asyncio.sleep(timeout_val)
            except asyncio.CancelledError:
                dag_info = await self.ensure_async_call(self._caller.cancel_dag, dag_id)
                if dag_info.status != ExecutionStatus.CANCELLED:  # pragma: no cover
                    raise
            finally:
                if dag_info.status == ExecutionStatus.SUCCEEDED:
                    progress.value = 1.0
                elif dag_info.status == ExecutionStatus.FAILED:
                    dag_info.error_info.reraise()

            if dag_info.status in (ExecutionStatus.RUNNING, ExecutionStatus.CANCELLED):
                return

            for key, result_info in dag_info.tileable_to_result_infos.items():
                t = key_to_tileables[key]
                fetcher = get_fetcher_cls(result_info.result_type)(self._odps_entry)
                await fetcher.update_tileable_meta(t, result_info)
                self._tileable_to_infos[t] = result_info

    async def _execute_locally(
        self,
        tileable_graph: TileableGraph,
        to_execute_tileables: List[TileableType],
    ):
        cur_loop = asyncio.get_running_loop()

        def run_sync(subdag_id):
            codegen = SPECodeGenerator(self.session_id, subdag_id)
            generated_code = codegen.generate(tileable_graph)

            runner = LocalSPEDagRunner(
                self._session_id,
                subdag_id,
                tileable_graph,
                generated_code,
                settings={},
                odps_entry=self._odps_entry,
                tileable_to_info=self._tileable_to_infos,
                loop=cur_loop,
            )
            with odps_option_context():
                self._odps_entry.to_global()
                key_to_info = runner.run()
            for tileable in to_execute_tileables:
                self._tileable_to_infos[tileable] = key_to_info[tileable.key]

        mock_subdag_id = f"subdag_local_{int(time.time())}"
        progress = Progress()
        profiling = Profiling()
        aio_task = asyncio.create_task(self.to_thread(run_sync, mock_subdag_id))
        return ExecutionInfo(
            aio_task,
            progress,
            profiling,
            asyncio.get_running_loop(),
            to_execute_tileables,
        )

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
                raise TileableNotExecutedError(
                    f"Cannot fetch unexecuted tileable: {tileable!r}"
                )

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
                results.append(await fetcher.fetch(data_tileable, info, indexes))
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

    async def get_logview_address(self, hours=None) -> Optional[str]:
        return await self.get_dag_logview_address(None, hours)

    async def get_dag_logview_address(self, dag_id=None, hours=None) -> Optional[str]:
        return await self.ensure_async_call(
            self._caller.get_logview_address, dag_id, hours
        )

    async def _show_logview_address(self, dag_id=None, hours=None):
        identity = f"Session ID: {self._session_id}"
        if dag_id:
            identity += f", DAG ID: {dag_id}"

        logview_addr = await self.get_dag_logview_address(dag_id, hours)
        if logview_addr:
            logger.info("%s, Logview: %s", identity, logview_addr)
        else:
            logger.info("%s, Logview address does not exist", identity)


class MaxFrameRestCaller(MaxFrameServiceCaller):
    _client: FrameDriverClient
    _session_id: Optional[str]

    def __init__(self, odps_entry: ODPS, client: FrameDriverClient):
        self._odps_entry = odps_entry
        self._client = client
        self._session_id = None

    async def create_session(self) -> SessionInfo:
        info = await self._client.create_session(self.get_settings_to_upload())
        self._session_id = info.session_id
        return info

    async def delete_session(self) -> None:
        await self._client.delete_session(self._session_id)

    async def submit_dag(
        self,
        dag: TileableGraph,
        managed_input_infos: Dict[str, ResultInfo] = None,
        new_settings: Dict[str, Any] = None,
    ) -> DagInfo:
        return await self._client.submit_dag(
            self._session_id, dag, managed_input_infos, new_settings=new_settings
        )

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
            real_endpoint, session_id, odps_entry=odps_entry, timeout=timeout, **kwargs
        )

    @classmethod
    def _create_caller(cls, odps_entry: ODPS, address: str, **kwargs):
        return MaxFrameRestCaller(odps_entry, FrameDriverClient(address))


def register_session_schemes(overwrite: bool = False):
    MaxFrameRestSession.register_schemes(overwrite=overwrite)
