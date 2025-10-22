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

import base64
import json
import logging
import time
from typing import Any, Dict, List, Optional, Type, Union

import msgpack
from odps import ODPS
from odps import options as odps_options
from odps.errors import EmptyTaskInfoError, parse_instance_error
from odps.models import Instance, MaxFrameTask

from maxframe.config import options
from maxframe.core import TileableGraph
from maxframe.errors import NoTaskServerResponseError, SessionAlreadyClosedError
from maxframe.protocol import DagInfo, JsonSerializable, ResultInfo, SessionInfo
from maxframe.utils import deserialize_serializable, serialize_serializable, to_str

try:
    from maxframe import __version__ as mf_version
except ImportError:
    mf_version = None

from .consts import (
    EMPTY_RESPONSE_RETRY_COUNT,
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


class MaxFrameInstanceCaller(MaxFrameServiceCaller):
    _instance: Optional[Instance]

    def __init__(
        self,
        odps_entry: ODPS,
        task_name: Optional[str] = None,
        project: Optional[str] = None,
        priority: Optional[int] = None,
        running_cluster: Optional[str] = None,
        nested_instance_id: Optional[str] = None,
        major_version: Optional[str] = None,
        output_format: Optional[str] = None,
        **kwargs,
    ):
        if callable(odps_options.get_priority):
            default_priority = odps_options.get_priority(odps_entry)
        else:
            default_priority = odps_options.priority
        priority = priority if priority is not None else default_priority

        self._odps_entry = odps_entry
        self._task_name = task_name
        self._project = project
        self._priority = priority
        self._running_cluster = running_cluster
        self._major_version = major_version
        self._output_format = output_format or MAXFRAME_OUTPUT_MSGPACK_FORMAT
        self._deleted = False

        if nested_instance_id is None:
            self._nested = False
            self._instance = None
        else:
            self._nested = True
            self._instance = odps_entry.get_instance(nested_instance_id)

    @property
    def instance(self):
        return self._instance

    def _deserial_task_info_result(
        self, content: Union[bytes, str, dict], target_cls: Type[JsonSerializable]
    ):
        if isinstance(content, (str, bytes)):
            if len(content) == 0:
                content = "{}"
            json_data = json.loads(to_str(content))
        else:
            json_data = content
        encoded_result = json_data.get("result")
        if not encoded_result:
            if self._deleted:
                return None
            else:
                raise SessionAlreadyClosedError(self._instance.id)

        try:
            result_data = base64.b64decode(encoded_result)
        except:
            # todo change to a better logic when it is possible
            #  to judge if server side returns success or fail
            raise parse_instance_error(encoded_result)

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

    def _create_maxframe_task(self) -> MaxFrameTask:
        task = MaxFrameTask(name=self._task_name, major_version=self._major_version)
        mf_settings = self.get_settings_to_upload()
        mf_opts = {
            "odps.maxframe.settings": json.dumps(mf_settings),
            "odps.maxframe.output_format": self._output_format,
        }
        if mf_settings.get("session.quota_name", None):
            mf_opts["odps.task.wlm.quota"] = mf_settings["session.quota_name"]
        if mf_version:
            mf_opts["odps.maxframe.client_version"] = mf_version
        task.update_settings(mf_opts)
        return task

    def create_session(self) -> SessionInfo:
        task = self._create_maxframe_task()
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
                raise TimeoutError(
                    f"Check session startup time out. Instance ID: {self._instance.id}"
                )
            time.sleep(interval)
            interval = min(max_interval, interval * 2)

    def _put_task_info(self, method_name: str, json_data: dict):
        for trial in range(EMPTY_RESPONSE_RETRY_COUNT):
            try:
                return self._instance.put_task_info(
                    self._task_name,
                    method_name,
                    json.dumps(json_data),
                    raise_empty=True,
                )
            except EmptyTaskInfoError as ex:
                # retry when server returns HTTP 204, which is designed for retry
                if ex.code != 204 or trial >= EMPTY_RESPONSE_RETRY_COUNT - 1:
                    raise NoTaskServerResponseError(
                        f"No response for request {method_name}. "
                        f"Instance ID: {self._instance.id}. "
                        f"Request ID: {ex.request_id}"
                    ) from None
                time.sleep(0.5)

    def get_session(self) -> SessionInfo:
        req_data = {"output_format": self._output_format}
        serialized = self._put_task_info(MAXFRAME_TASK_GET_SESSION_METHOD, req_data)
        info: SessionInfo = self._deserial_task_info_result(serialized, SessionInfo)
        info.session_id = self._instance.id
        return info

    def delete_session(self) -> None:
        if not self._nested:
            self._instance.stop()
        else:
            req_data = {"output_format": self._output_format}
            self._put_task_info(MAXFRAME_TASK_DELETE_SESSION_METHOD, req_data)
        self._deleted = True

    def submit_dag(
        self,
        dag: TileableGraph,
        managed_input_infos: Optional[Dict[str, ResultInfo]] = None,
        new_settings: Dict[str, Any] = None,
    ) -> DagInfo:
        new_settings_value = {
            "odps.maxframe.settings": json.dumps(new_settings),
        }
        req_data = {
            "protocol": MAXFRAME_DEFAULT_PROTOCOL,
            "dag": base64.b64encode(serialize_serializable(dag)).decode(),
            "managed_input_infos": base64.b64encode(
                serialize_serializable(managed_input_infos)
            ).decode(),
            "new_settings": json.dumps(new_settings_value),
            "output_format": self._output_format,
        }
        res = self._put_task_info(MAXFRAME_TASK_SUBMIT_DAG_METHOD, req_data)
        return self._deserial_task_info_result(res, DagInfo)

    def get_dag_info(self, dag_id: str) -> DagInfo:
        req_data = {
            "protocol": MAXFRAME_DEFAULT_PROTOCOL,
            "dag_id": dag_id,
            "output_format": self._output_format,
        }
        res = self._put_task_info(MAXFRAME_TASK_GET_DAG_INFO_METHOD, req_data)
        return self._deserial_task_info_result(res, DagInfo)

    def cancel_dag(self, dag_id: str) -> DagInfo:
        req_data = {
            "protocol": MAXFRAME_DEFAULT_PROTOCOL,
            "dag_id": dag_id,
            "output_format": self._output_format,
        }
        res = self._put_task_info(MAXFRAME_TASK_CANCEL_DAG_METHOD, req_data)
        return self._deserial_task_info_result(res, DagInfo)

    def decref(self, tileable_keys: List[str]) -> None:
        req_data = {
            "tileable_keys": ",".join(tileable_keys),
        }
        self._put_task_info(MAXFRAME_TASK_DECREF_METHOD, req_data)

    def get_logview_address(self, dag_id=None, hours=None) -> Optional[str]:
        """
        Generate logview address

        Parameters
        ----------
            dag_id: id of dag for which dag logview detail page to access
            hours: hours of the logview address auth limit
        Returns
        -------
            Logview address
        """
        hours = hours or options.session.logview_hours
        # notice: maxframe can't reuse subQuery else will conflict with mcqa when fetch resource data,
        #         added dagId for maxframe so logview backend will return maxframe data format if
        #         instance and dagId is provided.
        dag_suffix = f"&dagId={dag_id}" if dag_id else ""
        return self._instance.get_logview_address(hours) + dag_suffix


class MaxFrameTaskSession(MaxFrameSession):
    schemes = [ODPS_SESSION_INSECURE_SCHEME, ODPS_SESSION_SECURE_SCHEME]

    _caller: MaxFrameInstanceCaller

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

    @property
    def closed(self) -> bool:
        if super().closed:
            return True
        if not self._caller or not self._caller.instance:
            # session not initialized yet
            return False
        return self._caller.instance.is_terminated()


def register_session_schemes(overwrite: bool = False):
    MaxFrameTaskSession.register_schemes(overwrite=overwrite)
