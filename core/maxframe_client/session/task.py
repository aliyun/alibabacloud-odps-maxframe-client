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

import base64
import json
import logging
import time
from typing import Dict, List, Optional, Type, Union

import msgpack
from odps import ODPS
from odps import options as odps_options
from odps import serializers
from odps.errors import parse_instance_error
from odps.models import Instance, Task

from maxframe.config import options
from maxframe.core import TileableGraph
from maxframe.protocol import DagInfo, JsonSerializable, ResultInfo, SessionInfo
from maxframe.utils import deserialize_serializable, serialize_serializable, to_str

try:
    from maxframe import __version__ as mf_version
except ImportError:
    mf_version = None

from .consts import (
    MAXFRAME_DEFAULT_PROTOCOL,
    MAXFRAME_OUTPUT_JSON_FORMAT,
    MAXFRAME_OUTPUT_MAXFRAME_FORMAT,
    MAXFRAME_OUTPUT_MSGPACK_FORMAT,
    MAXFRAME_TASK_CANCEL_DAG_METHOD,
    MAXFRAME_TASK_CREATE_SESSION_METHOD,
    MAXFRAME_TASK_DECREF_METHOD,
    MAXFRAME_TASK_DELETE_SESSION_METHOD,
    MAXFRAME_TASK_GET_DAG_INFO_METHOD,
    MAXFRAME_TASK_GET_SESSION_METHOD,
    MAXFRAME_TASK_SUBMIT_DAG_METHOD,
    ODPS_SESSION_INSECURE_SCHEME,
    ODPS_SESSION_SECURE_SCHEME,
)
from .odps import MaxFrameServiceCaller, MaxFrameSession

logger = logging.getLogger(__name__)


class MaxFrameTask(Task):
    __slots__ = ("_output_format", "_major_version", "_service_endpoint")
    _root = "MaxFrame"
    _anonymous_task_name = "AnonymousMaxFrameTask"

    command = serializers.XMLNodeField("Command", default="CREATE_SESSION")

    def __init__(self, **kwargs):
        kwargs["name"] = kwargs.get("name") or self._anonymous_task_name
        self._output_format = kwargs.pop(
            "output_format", MAXFRAME_OUTPUT_MSGPACK_FORMAT
        )
        self._major_version = kwargs.pop("major_version", None)
        self._service_endpoint = kwargs.pop("service_endpoint", None)
        super().__init__(**kwargs)

    def serial(self):
        if self.properties is None:
            self.properties = dict()

        if odps_options.default_task_settings:
            settings = odps_options.default_task_settings
        else:
            settings = dict()

        if self._major_version is not None:
            settings["odps.task.major.version"] = self._major_version

        if "settings" in self.properties:
            settings.update(json.loads(self.properties["settings"]))

        # merge sql options
        sql_settings = (odps_options.sql.settings or {}).copy()
        sql_settings.update(options.sql.settings or {})

        mf_settings = dict(options.to_dict(remote_only=True).items())
        mf_settings["sql.settings"] = sql_settings
        mf_opts = {
            "odps.maxframe.settings": json.dumps(mf_settings),
            "odps.maxframe.output_format": self._output_format,
            "odps.service.endpoint": self._service_endpoint,
        }
        if mf_version:
            mf_opts["odps.maxframe.client_version"] = mf_version
        settings.update(mf_opts)
        self.properties["settings"] = json.dumps(settings)
        return super().serial()


class MaxFrameInstanceCaller(MaxFrameServiceCaller):
    _instance: Optional[Instance]

    def __init__(
        self,
        odps_entry: ODPS,
        task_name: Optional[str] = None,
        project: Optional[str] = None,
        priority: Optional[str] = None,
        running_cluster: Optional[str] = None,
        nested_instance_id: Optional[str] = None,
        major_version: Optional[str] = None,
        output_format: Optional[str] = None,
        **kwargs,
    ):
        self._odps_entry = odps_entry
        self._task_name = task_name
        self._project = project
        self._priority = priority
        self._running_cluster = running_cluster
        self._major_version = major_version
        self._output_format = output_format or MAXFRAME_OUTPUT_MSGPACK_FORMAT
        if nested_instance_id is None:
            self._nested = False
            self._instance = None
        else:
            self._nested = True
            self._instance = odps_entry.get_instance(nested_instance_id)

    def _deserial_task_info_result(
        self, content: Union[bytes, str, dict], target_cls: Type[JsonSerializable]
    ):
        if isinstance(content, (str, bytes)):
            json_data = json.loads(to_str(content))
        else:
            json_data = content
        result_data = base64.b64decode(json_data["result"])
        if self._output_format == MAXFRAME_OUTPUT_MAXFRAME_FORMAT:
            return deserialize_serializable(result_data)
        elif self._output_format == MAXFRAME_OUTPUT_JSON_FORMAT:
            return target_cls.from_json(json.loads(result_data))
        elif self._output_format == MAXFRAME_OUTPUT_MSGPACK_FORMAT:
            return target_cls.from_json(msgpack.loads(result_data))
        else:
            raise ValueError(
                f"Serialization format {self._output_format} not supported"
            )

    def create_session(self) -> SessionInfo:
        task = MaxFrameTask(
            name=self._task_name,
            major_version=self._major_version,
            output_format=self._output_format,
            service_endpoint=self._odps_entry.endpoint,
        )
        if not self._nested:
            self._task_name = task.name
            project = self._odps_entry.get_project(self._project)
            self._instance = project.instances.create(
                task=task,
                priority=self._priority,
                running_cluster=self._running_cluster,
            )
            self._wait_instance_task_ready()
            return self.get_session()
        else:
            result = self._instance.put_task_info(
                self._task_name,
                MAXFRAME_TASK_CREATE_SESSION_METHOD,
                task.properties["settings"],
            )
            return self._deserial_task_info_result(result, SessionInfo)

    def _parse_instance_result_error(self):
        result_data = self._instance.get_task_result(self._task_name)
        try:
            info = self._deserial_task_info_result({"result": result_data}, SessionInfo)
        except:
            raise parse_instance_error(result_data)
        info.error_info.reraise()

    def _wait_instance_task_ready(
        self, interval: float = 0.1, max_interval: float = 5.0, timeout: int = None
    ):
        check_time = time.time()
        timeout = timeout or options.client.task_start_timeout
        while True:
            if self._instance.is_terminated(retry=True):
                self._parse_instance_result_error()
            status_json = json.loads(
                self._instance.get_task_info(self._task_name, "status") or "{}"
            )
            if status_json.get("status") == "Running":
                break
            if time.time() - check_time > timeout:
                raise TimeoutError("Check session startup time out")
            time.sleep(interval)
            interval = min(max_interval, interval * 2)

    def get_session(self) -> SessionInfo:
        req_data = {"output_format": self._output_format}
        serialized = self._instance.put_task_info(
            self._task_name, MAXFRAME_TASK_GET_SESSION_METHOD, json.dumps(req_data)
        )
        info: SessionInfo = self._deserial_task_info_result(serialized, SessionInfo)
        info.session_id = self._instance.id
        return info

    def delete_session(self) -> None:
        if not self._nested:
            self._instance.stop()
        else:
            req_data = {"output_format": self._output_format}
            self._instance.put_task_info(
                self._task_name,
                MAXFRAME_TASK_DELETE_SESSION_METHOD,
                json.dumps(req_data),
            )

    def submit_dag(
        self,
        dag: TileableGraph,
        managed_input_infos: Optional[Dict[str, ResultInfo]] = None,
    ) -> DagInfo:
        req_data = {
            "protocol": MAXFRAME_DEFAULT_PROTOCOL,
            "dag": base64.b64encode(serialize_serializable(dag)).decode(),
            "managed_input_infos": base64.b64encode(
                serialize_serializable(managed_input_infos)
            ).decode(),
            "output_format": self._output_format,
        }
        res = self._instance.put_task_info(
            self._task_name, MAXFRAME_TASK_SUBMIT_DAG_METHOD, json.dumps(req_data)
        )
        return self._deserial_task_info_result(res, DagInfo)

    def get_dag_info(self, dag_id: str) -> DagInfo:
        req_data = {
            "protocol": MAXFRAME_DEFAULT_PROTOCOL,
            "dag_id": dag_id,
            "output_format": self._output_format,
        }
        res = self._instance.put_task_info(
            self._task_name, MAXFRAME_TASK_GET_DAG_INFO_METHOD, json.dumps(req_data)
        )
        return self._deserial_task_info_result(res, DagInfo)

    def cancel_dag(self, dag_id: str) -> DagInfo:
        req_data = {
            "protocol": MAXFRAME_DEFAULT_PROTOCOL,
            "dag_id": dag_id,
            "output_format": self._output_format,
        }
        res = self._instance.put_task_info(
            self._task_name, MAXFRAME_TASK_CANCEL_DAG_METHOD, json.dumps(req_data)
        )
        return self._deserial_task_info_result(res, DagInfo)

    def decref(self, tileable_keys: List[str]) -> None:
        req_data = {
            "tileable_keys": ",".join(tileable_keys),
        }
        self._instance.put_task_info(
            self._task_name, MAXFRAME_TASK_DECREF_METHOD, json.dumps(req_data)
        )


class MaxFrameTaskSession(MaxFrameSession):
    schemes = [ODPS_SESSION_INSECURE_SCHEME, ODPS_SESSION_SECURE_SCHEME]

    _instance: Instance

    @classmethod
    def _create_caller(
        cls,
        odps_entry: ODPS,
        address: str,
        priority: Optional[int] = None,
        project: Optional[str] = None,
        running_cluster: Optional[str] = None,
        **kwargs,
    ) -> MaxFrameServiceCaller:
        return MaxFrameInstanceCaller(
            odps_entry,
            priority=priority,
            running_cluster=running_cluster,
            project=project,
            **kwargs,
        )


def register_session_schemes(overwrite: bool = False):
    MaxFrameTaskSession.register_schemes(overwrite=overwrite)
